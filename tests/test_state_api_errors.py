import json
import uuid

import fakeredis
import pytest
from fastapi.testclient import TestClient

import api.main as api_module
from src.rate_limiter import MemoryRateLimiter, RedisRateLimiter
from src.state_backends import StateBackends
from src.state_runtime import StateSettings
from tests.test_api import FakeService


@pytest.fixture
def client(monkeypatch):
    server = fakeredis.FakeServer()
    redis = fakeredis.FakeRedis(server=server, decode_responses=True)
    backends = StateBackends(StateSettings(session_backend="redis", rate_limit_backend="redis",
                                           redis_url="redis://localhost"), client=redis)
    service = FakeService()
    api_module.app.dependency_overrides[api_module.get_knowledge_service] = lambda: service
    monkeypatch.setattr(api_module, "_backends", backends)
    monkeypatch.setattr(api_module, "_limiter", backends.limiter)
    try:
        yield TestClient(api_module.app, raise_server_exceptions=False), server, service
    finally:
        api_module.app.dependency_overrides.clear()


def test_outage_fails_closed_but_liveness_and_local_documents_work(client):
    api, server, _service = client
    server.connected = False
    assert api.get("/health").status_code == 200
    assert api.get("/documents").status_code == 200
    assert api.get("/ready").status_code == 503
    for endpoint in ("/chat", "/chat/stream"):
        response = api.post(endpoint, json={"session_id": str(uuid.uuid4()), "question": "hello"})
        assert response.status_code == 503
        assert response.json()["error_code"] == "STATE_UNAVAILABLE"
    upload = api.post("/documents/upload", files={"file": ("rag.md", b"content")})
    assert upload.status_code == 503
    assert not _service.documents
    server.connected = True
    assert api.get("/ready").status_code == 200


def test_ip_limit_prevents_session_rotation(client, monkeypatch):
    api, _server, _service = client
    monkeypatch.setattr(api_module, "CHAT_IP_REQUESTS_PER_MINUTE", 2)
    for _ in range(2):
        assert api.post("/chat", json={"session_id": str(uuid.uuid4()), "question": "q"}).status_code == 200
    response = api.post("/chat", json={"session_id": str(uuid.uuid4()), "question": "q"})
    assert response.status_code == 429
    assert int(response.headers["Retry-After"]) >= 1


def test_sse_late_failure_is_an_error_event_without_done(client):
    api, _server, service = client
    service.fail_chat = True
    response = api.post("/chat/stream", json={"session_id": str(uuid.uuid4()), "question": "q"})
    assert response.status_code == 200
    assert '"type": "error"' in response.text
    assert '"type": "done"' not in response.text
    assert "sk-test-secret" not in response.text


def test_redis_mode_missing_url_fails_at_startup(monkeypatch):
    monkeypatch.setenv("SESSION_BACKEND", "redis")
    monkeypatch.delenv("REDIS_URL", raising=False)
    with pytest.raises(ValueError, match="REDIS_URL"):
        with TestClient(api_module.app):
            pass
