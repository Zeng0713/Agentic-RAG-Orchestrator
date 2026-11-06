"""Six-turn session stores with fail-fast leases and idempotent commits."""

import json
import logging
import threading
import time
import uuid

import redis

from src.state_runtime import LeaseLost, SessionBusy, StateError, StateSettings

logger = logging.getLogger("rag_project")


def normalize_session_id(session_id):
    try:
        return str(uuid.UUID(session_id))
    except (ValueError, TypeError, AttributeError):
        raise ValueError("session_id 必须是有效的 UUID") from None


def flatten(turns):
    return [message for turn in turns for message in (
        {"role": "user", "content": turn["question"]},
        {"role": "assistant", "content": turn["answer"]},
    )]


class MemoryLease:
    def __init__(self, store, key, entry):
        self.store, self.key, self.entry = store, key, entry
        self.closed = False

    def ensure_owned(self):
        if self.closed:
            raise LeaseLost()

    def close(self):
        if not self.closed:
            with self.store._lock:
                self.closed = True
                self.entry["updated"] = self.store.clock()
                self.entry["lock"].release()


class MemorySessionStore:
    def __init__(self, ttl=86400, max_sessions=2048, clock=time.monotonic):
        self.ttl, self.max_sessions, self.clock = ttl, max_sessions, clock
        self._entries = {}
        self._lock = threading.Lock()

    def acquire(self, session_id):
        key = normalize_session_id(session_id)
        with self._lock:
            now = self.clock()
            for sid, entry in list(self._entries.items()):
                if not entry["lock"].locked() and now - entry["updated"] >= self.ttl:
                    del self._entries[sid]
            if key not in self._entries:
                if len(self._entries) >= self.max_sessions:
                    idle = [sid for sid, entry in self._entries.items() if not entry["lock"].locked()]
                    if not idle:
                        raise StateError("会话容量已满，请稍后再试")
                    del self._entries[min(idle, key=lambda sid: self._entries[sid]["updated"])]
                self._entries[key] = {"turns": [], "updated": now, "lock": threading.Lock()}
            entry = self._entries[key]
            if not entry["lock"].acquire(blocking=False):
                raise SessionBusy()
            return MemoryLease(self, key, entry)

    def read(self, lease):
        lease.ensure_owned()
        return flatten(lease.entry["turns"])

    def save(self, lease, request_id, question, answer):
        lease.ensure_owned()
        turns = lease.entry["turns"]
        if not any(turn["request_id"] == request_id for turn in turns):
            turns.append(dict(request_id=request_id, question=question, answer=answer))
            del turns[:-6]
        lease.entry["updated"] = self.clock()

    def clear(self, lease):
        lease.ensure_owned()
        lease.entry["turns"].clear()

    def ready(self):
        return True


RENEW = """
if redis.call('GET', KEYS[1]) == ARGV[1] then
    return redis.call('PEXPIRE', KEYS[1], ARGV[2])
end
return 0
"""
RELEASE = """
if redis.call('GET', KEYS[1]) == ARGV[1] then return redis.call('DEL', KEYS[1]) end
return 0
"""
READ = """
if redis.call('GET', KEYS[1]) ~= ARGV[1] then return {0} end
local turns = redis.call('LRANGE', KEYS[2], 0, -1)
redis.call('EXPIRE', KEYS[2], ARGV[2])
return {1, turns}
"""
COMMIT = """
local kind = redis.call('TYPE', KEYS[2]).ok
if kind ~= 'none' and kind ~= 'list' then return -2 end
local turn = cjson.decode(ARGV[2])
local ttl = tonumber(ARGV[3])
if not ttl or ttl <= 0 then return -2 end
for _, item in ipairs(redis.call('LRANGE', KEYS[2], 0, -1)) do
    if cjson.decode(item).request_id == turn.request_id then return 2 end
end
if redis.call('GET', KEYS[1]) ~= ARGV[1] then return 0 end
redis.call('RPUSH', KEYS[2], ARGV[2])
redis.call('LTRIM', KEYS[2], -6, -1)
redis.call('EXPIRE', KEYS[2], ttl)
return 1
"""
CLEAR = """
if redis.call('GET', KEYS[1]) ~= ARGV[1] then return 0 end
redis.call('DEL', KEYS[2])
return 1
"""


class RedisLease:
    def __init__(self, store, key, token):
        self.store, self.key, self.token = store, key, token
        self.history_key = key.removesuffix(':lock') + ':turns'
        self.lost = threading.Event()
        self.stopped = threading.Event()
        self.thread = threading.Thread(target=self._watchdog, daemon=True)
        self.thread.start()

    def _watchdog(self):
        while not self.stopped.wait(self.store.settings.renew_seconds):
            try:
                renewed = self.store.client.eval(RENEW, 1, self.key, self.token,
                                                 int(self.store.settings.lease_seconds * 1000))
                if not renewed:
                    self.lost.set()
            except redis.RedisError:
                self.lost.set()
            if self.lost.is_set():
                logger.warning("session_lease_lost")
                return

    def ensure_owned(self):
        if self.lost.is_set() or self.stopped.is_set():
            raise LeaseLost()

    def close(self):
        if self.stopped.is_set():
            return
        self.stopped.set()
        self.thread.join(timeout=1.2)
        try:
            self.store.client.eval(RELEASE, 1, self.key, self.token)
        except redis.RedisError as exc:
            logger.warning("session_release_failed error_type=%s", type(exc).__name__)


class RedisSessionStore:
    def __init__(self, client, settings=None):
        self.client = client
        self.settings = settings or StateSettings()

    def acquire(self, session_id):
        sid = normalize_session_id(session_id)
        key = f"{self.settings.prefix}:session:{{{sid}}}:lock"
        token = uuid.uuid4().hex
        started = time.monotonic()
        try:
            acquired = self.client.set(key, token, nx=True,
                                       px=int(self.settings.lease_seconds * 1000))
        except redis.RedisError as exc:
            logger.warning("session_acquire_failed error_type=%s", type(exc).__name__)
            raise StateError() from None
        logger.info("session_acquire duration_ms=%.1f acquired=%s",
                    (time.monotonic() - started) * 1000, bool(acquired))
        if not acquired:
            raise SessionBusy()
        return RedisLease(self, key, token)

    def read(self, lease):
        lease.ensure_owned()
        try:
            result = self.client.eval(READ, 2, lease.key, lease.history_key, lease.token,
                                      self.settings.session_ttl)
            if not result[0]:
                raise LeaseLost()
            return flatten([json.loads(item) for item in result[1]])
        except (redis.RedisError, ValueError, KeyError, TypeError):
            raise StateError() from None

    def save(self, lease, request_id, question, answer):
        lease.ensure_owned()
        turn = json.dumps(dict(request_id=request_id, question=question, answer=answer), ensure_ascii=False)
        started = time.monotonic()
        for attempt in range(3):
            lease.ensure_owned()
            try:
                result = self.client.eval(COMMIT, 2, lease.key, lease.history_key,
                                          lease.token, turn, self.settings.session_ttl)
                if result == 0:
                    raise LeaseLost()
                if result < 0:
                    raise StateError("会话存储格式异常")
                logger.info("session_commit duration_ms=%.1f attempts=%d",
                            (time.monotonic() - started) * 1000, attempt + 1)
                return
            except (redis.ConnectionError, redis.TimeoutError) as exc:
                logger.warning("session_commit_retry attempt=%d error_type=%s", attempt + 1, type(exc).__name__)
                if attempt < 2:
                    time.sleep(0.1 * (attempt + 1))
            except redis.RedisError:
                raise StateError() from None
        raise StateError("回答已生成，但无法确认本轮对话是否保存")

    def clear(self, lease):
        lease.ensure_owned()
        try:
            if not self.client.eval(CLEAR, 2, lease.key, lease.history_key, lease.token):
                raise LeaseLost()
        except redis.RedisError:
            raise StateError() from None

    def ready(self):
        try:
            return bool(self.client.ping())
        except redis.RedisError:
            raise StateError() from None
