from concurrent.futures import ThreadPoolExecutor
import threading
import uuid

import pytest
from fastapi.testclient import TestClient

import api.main as api_module
from src.rate_limiter import MemoryRateLimiter
from src.services.knowledge_service import KnowledgeService
from src.session_store import MemorySessionStore
from src.state_runtime import StateError


class BlockingLLM:
    def __init__(self):
        self.entered = threading.Event()
        self.release = threading.Event()
        self.two_entered = threading.Event()
        self.count = 0
        self.lock = threading.Lock()

    def invoke(self, _messages, **_kwargs):
        with self.lock:
            self.count += 1
            if self.count >= 2:
                self.two_entered.set()
        self.entered.set()
        assert self.release.wait(5)
        return type("Response", (), {"content": "answer", "tool_calls": []})()


def service_for(tmp_path, llm, store=None):
    return KnowledgeService(vector_store=object(), llm=llm,
                            documents_dir=tmp_path / "docs", uploads_dir=tmp_path / "uploads",
                            session_store=store)


def test_same_session_conflict_before_sse_and_other_sessions_can_generate(tmp_path, monkeypatch):
    llm = BlockingLLM()
    service = service_for(tmp_path, llm)
    api_module.app.dependency_overrides[api_module.get_knowledge_service] = lambda: service
    monkeypatch.setattr(api_module, "_limiter", MemoryRateLimiter())
    client = TestClient(api_module.app)
    payload = {"session_id": str(uuid.uuid4()), "question": "q"}
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            first = pool.submit(client.post, "/chat", json=payload)
            try:
                assert llm.entered.wait(3)
                conflict = client.post("/chat/stream", json=payload)
                assert conflict.status_code == 409
                assert conflict.json()["error_code"] == "SESSION_BUSY"
                different = pool.submit(client.post, "/chat", json={**payload, "session_id": str(uuid.uuid4())})
                assert llm.two_entered.wait(3), "Different sessions must not share the LLM lock"
            finally:
                llm.release.set()
            assert first.result(timeout=3).status_code == 200
            assert different.result(timeout=3).status_code == 200
    finally:
        llm.release.set()
        api_module.app.dependency_overrides.clear()


def test_successful_answer_survives_unconfirmed_history_write(tmp_path):
    class FailingSave(MemorySessionStore):
        def save(self, *_args):
            raise StateError()

    llm = BlockingLLM()
    llm.release.set()
    service = service_for(tmp_path, llm, FailingSave())
    result = service.chat(str(uuid.uuid4()), "q")
    assert result["answer"] == "answer"
    assert result["history_status"] == "unconfirmed"
    assert result["warning"]


def test_clear_conflicts_with_active_generation(tmp_path):
    service = service_for(tmp_path, BlockingLLM())
    sid = str(uuid.uuid4())
    lease = service.session_store.acquire(sid)
    try:
        from src.state_runtime import SessionBusy
        with pytest.raises(SessionBusy):
            service.clear_session(sid)
    finally:
        lease.close()


def test_deadline_and_cancelled_requests_do_not_append_history(tmp_path):
    llm = BlockingLLM()
    llm.release.set()
    service = service_for(tmp_path, llm)
    sid = str(uuid.uuid4())
    cancelled = threading.Event()
    cancelled.set()
    with pytest.raises(StateError, match="中断"):
        service.chat(sid, "q", cancelled=cancelled)
    service.chat_deadline = -1
    with pytest.raises(StateError, match="超时"):
        service.chat(sid, "q")
    lease = service.session_store.acquire(sid)
    try:
        assert service.session_store.read(lease) == []
    finally:
        lease.close()
