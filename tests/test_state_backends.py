import threading
import uuid
from concurrent.futures import ThreadPoolExecutor

import fakeredis
import pytest
import redis

from src.rate_limiter import MemoryRateLimiter, RedisRateLimiter
from src.session_store import MemorySessionStore, RedisSessionStore
from src.state_runtime import LeaseLost, SessionBusy, StateError, StateSettings


@pytest.fixture(params=["memory", "redis"])
def store(request):
    if request.param == "memory":
        return MemorySessionStore()
    return RedisSessionStore(fakeredis.FakeRedis(decode_responses=True))


def test_session_retention_isolation_and_idempotency(store):
    sid = str(uuid.uuid4())
    lease = store.acquire(sid)
    try:
        for i in range(8):
            store.save(lease, str(i), f"q{i}", f"a{i}")
        store.save(lease, "7", "q7", "a7")
        history = store.read(lease)
        assert len(history) == 12
        assert history[0]["content"] == "q2"
        if hasattr(store, "client"):
            assert 0 < store.client.ttl(lease.history_key) <= 86400
        with pytest.raises(SessionBusy):
            store.acquire(sid)
        other = store.acquire(str(uuid.uuid4()))
        try:
            assert store.read(other) == []
        finally:
            other.close()
        store.clear(lease)
        assert store.read(lease) == []
    finally:
        lease.close()


def test_redis_instances_share_history_and_prevent_stale_commits():
    server = fakeredis.FakeServer()
    a = RedisSessionStore(fakeredis.FakeRedis(server=server, decode_responses=True))
    b = RedisSessionStore(fakeredis.FakeRedis(server=server, decode_responses=True))
    sid = str(uuid.uuid4())
    lease = a.acquire(sid)
    a.save(lease, "first", "question", "answer")
    a.client.set(lease.key, "new-owner", px=60000)
    with pytest.raises(LeaseLost):
        a.save(lease, "second", "stale", "must not save")
    lease.close()
    assert a.client.get(lease.key) == "new-owner"
    a.client.delete(lease.key)
    other = b.acquire(sid)
    try:
        assert b.read(other)[0]["content"] == "question"
    finally:
        other.close()


def test_commit_retry_after_lost_ack_is_idempotent(monkeypatch):
    client = fakeredis.FakeRedis(decode_responses=True)
    store = RedisSessionStore(client)
    lease = store.acquire(str(uuid.uuid4()))
    original = client.eval
    calls = 0

    def lost_ack(script, *args):
        nonlocal calls
        result = original(script, *args)
        if "RPUSH" in script:
            calls += 1
            if calls == 1:
                raise redis.TimeoutError("credential-must-not-leak")
        return result

    monkeypatch.setattr(client, "eval", lost_ack)
    try:
        store.save(lease, "same-id", "q", "a")
        assert len(store.read(lease)) == 2
        assert calls == 2
    finally:
        lease.close()


@pytest.mark.parametrize("backend", ["memory", "redis"])
def test_dual_limit_is_atomic_and_shared(backend):
    if backend == "memory":
        a = b = MemoryRateLimiter()
    else:
        server = fakeredis.FakeServer()
        a, b = [RedisRateLimiter(fakeredis.FakeRedis(server=server, decode_responses=True)) for _ in range(2)]
    assert a.check([("ip", 2), ("session-a", 1)]).allowed
    assert not b.check([("ip", 2), ("session-a", 1)]).allowed
    assert b.check([("ip", 2), ("session-b", 1)]).allowed
    rejected = a.check([("ip", 2), ("session-c", 1)])
    assert not rejected.allowed and 1 <= rejected.retry_after <= 60


def test_memory_sliding_window_boundary():
    now = [0]
    limiter = MemoryRateLimiter(clock=lambda: now[0])
    assert limiter.check([("key", 1)]).allowed
    now[0] = 59.2
    assert limiter.check([("key", 1)]).retry_after == 1
    now[0] = 60
    assert limiter.check([("key", 1)]).allowed


def test_configuration_and_outage_are_safe(monkeypatch):
    monkeypatch.setenv("SESSION_BACKEND", "redis")
    monkeypatch.delenv("REDIS_URL", raising=False)
    with pytest.raises(ValueError, match="REDIS_URL"):
        StateSettings.from_env()
    server = fakeredis.FakeServer()
    server.connected = False
    client = fakeredis.FakeRedis(server=server, decode_responses=True)
    with pytest.raises(StateError):
        RedisSessionStore(client).acquire(str(uuid.uuid4()))
    with pytest.raises(StateError):
        RedisRateLimiter(client).check([("ip", 1)])


def test_watchdog_renews_lease_with_controlled_progress(monkeypatch):
    client = fakeredis.FakeRedis(decode_responses=True)
    settings = StateSettings(lease_seconds=0.3, renew_seconds=0.05)
    renewed = threading.Event()
    original = client.eval

    def observe(script, *args):
        result = original(script, *args)
        if "PEXPIRE" in script:
            renewed.set()
        return result

    monkeypatch.setattr(client, "eval", observe)
    store = RedisSessionStore(client, settings)
    lease = store.acquire(str(uuid.uuid4()))
    try:
        assert renewed.wait(3)
        assert client.pttl(lease.key) > 0
        lease.ensure_owned()
    finally:
        lease.close()


def test_memory_session_expires_without_evicting_an_active_lease():
    now = [0]
    store = MemorySessionStore(ttl=10, clock=lambda: now[0])
    sid = str(uuid.uuid4())
    lease = store.acquire(sid)
    store.save(lease, "id", "q", "a")
    now[0] = 20
    with pytest.raises(SessionBusy):
        store.acquire(sid)
    lease.close()
    now[0] = 31
    fresh = store.acquire(sid)
    try:
        assert store.read(fresh) == []
    finally:
        fresh.close()


def test_concurrent_redis_limit_never_over_admits():
    server = fakeredis.FakeServer()
    limiters = [RedisRateLimiter(fakeredis.FakeRedis(server=server, decode_responses=True)) for _ in range(2)]
    with ThreadPoolExecutor(max_workers=10) as pool:
        decisions = list(pool.map(lambda i: limiters[i % 2].check([("ip", 5)]).allowed, range(30)))
    assert sum(decisions) == 5


def test_wrong_type_limit_does_not_partially_charge_other_dimension():
    client = fakeredis.FakeRedis(decode_responses=True)
    limiter = RedisRateLimiter(client)
    import hashlib
    key = "rag:rate:{limits}:" + hashlib.sha256(b"broken").hexdigest()
    client.set(key, "wrong-type")
    with pytest.raises(StateError):
        limiter.check([("normal", 1), ("broken", 1)])
    assert limiter.check([("normal", 1)]).allowed
