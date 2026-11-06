"""Real Redis + independent uvicorn processes. Never calls a model provider."""

from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import uuid

import httpx
import pytest
import redis

pytestmark = pytest.mark.redis


def wait_until(predicate, timeout=20):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.05)
    raise AssertionError("Timed out waiting for local test service")


@pytest.fixture
def instances(tmp_path):
    url = os.getenv("TEST_REDIS_URL")
    if not url:
        pytest.skip("Set TEST_REDIS_URL to run real Redis tests")
    prefix = "rag-test-" + uuid.uuid4().hex
    env = {**os.environ, "SESSION_BACKEND": "redis", "RATE_LIMIT_BACKEND": "redis",
           "REDIS_URL": url, "REDIS_KEY_PREFIX": prefix, "RAG_TEST_GATE_DIR": str(tmp_path),
           "RAG_TEST_STORAGE_DIR": str(tmp_path),
           "CHAT_REQUESTS_PER_MINUTE": "4", "CHAT_IP_REQUESTS_PER_MINUTE": "100",
           "DEEPSEEK_API_KEY": "offline-test", "SILICONFLOW_API_KEY": "offline-test"}
    ports = []
    for _ in range(2):
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            ports.append(sock.getsockname()[1])
    processes = []
    urls = [f"http://127.0.0.1:{port}" for port in ports]
    client = httpx.Client(timeout=3, trust_env=False)
    store = redis.Redis.from_url(url, socket_timeout=1, socket_connect_timeout=1)
    try:
        for port in ports:
            processes.append(subprocess.Popen(
                [sys.executable, "-m", "uvicorn", "tests.support.redis_app:app", "--host", "127.0.0.1",
                 "--port", str(port), "--workers", "1", "--log-level", "warning"],
                env=env, cwd=Path(__file__).parents[1], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            ))

        def ready():
            assert all(p.poll() is None for p in processes), "Test API exited at startup"
            try:
                return all(client.get(base + "/ready").status_code == 200 for base in urls)
            except httpx.HTTPError:
                return False

        wait_until(ready)
        yield client, urls, store, prefix, tmp_path
    finally:
        (tmp_path / "release").touch()
        for process in processes:
            if process.poll() is None:
                process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
        try:
            keys = list(store.scan_iter(match=prefix + ":*"))
            if keys:
                store.delete(*keys)
        except redis.RedisError:
            pass
        store.close()
        client.close()


def test_processes_share_history_locks_and_rate_limits(instances):
    client, urls, _store, _prefix, directory = instances
    sid = str(uuid.uuid4())
    payload = {"session_id": sid, "question": "first"}
    assert client.post(urls[0] + "/chat", json=payload).status_code == 200
    second = client.post(urls[1] + "/chat", json={**payload, "question": "second"})
    assert json.loads(second.json()["data"]["answer"]) == ["first", "second"]
    for index in range(2):
        assert client.post(urls[index] + "/chat", json=payload).status_code == 200
    limited = client.post(urls[0] + "/chat", json=payload)
    assert limited.status_code == 429
    assert int(limited.headers["Retry-After"]) >= 1

    payload = {"session_id": str(uuid.uuid4()), "question": "__block__"}
    with ThreadPoolExecutor(max_workers=1) as pool:
        pending = pool.submit(client.post, urls[0] + "/chat", json=payload, timeout=15)
        try:
            wait_until(lambda: (directory / "entered").exists(), timeout=5)
            conflict = client.post(urls[1] + "/chat/stream", json=payload)
            assert conflict.status_code == 409
        finally:
            (directory / "release").touch()
        assert pending.result(timeout=5).status_code == 200


def test_real_redis_stop_start_recovers_without_restarting_apis(instances):
    container = os.getenv("REDIS_TEST_CONTAINER")
    if not container:
        pytest.skip("Set REDIS_TEST_CONTAINER for explicit stop/start fault injection")
    # Only an explicitly named disposable test container may be stopped.
    assert container.startswith(("codex-rag-redis-check-", "rag-redis-test"))
    client, urls, _store, _prefix, _directory = instances
    try:
        subprocess.run(["docker", "stop", container], check=True, capture_output=True, timeout=30)
        for base in urls:
            assert client.get(base + "/health").status_code == 200
            assert client.get(base + "/ready").status_code == 503
            sid = str(uuid.uuid4())
            response = client.post(base + "/chat", json={"session_id": sid, "question": "offline"})
            assert response.status_code == 503
            assert response.json()["error_code"] == "STATE_UNAVAILABLE"
            assert client.delete(base + "/sessions/" + sid).status_code == 503
    finally:
        subprocess.run(["docker", "start", container], check=True, capture_output=True, timeout=30)
    wait_until(lambda: all(client.get(base + "/ready").status_code == 200 for base in urls))
    assert client.post(urls[1] + "/chat", json={
        "session_id": str(uuid.uuid4()), "question": "recovered",
    }).status_code == 200
