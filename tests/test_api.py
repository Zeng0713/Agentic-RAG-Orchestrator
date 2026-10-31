import uuid

import pytest
from fastapi.testclient import TestClient

from api.main import RateLimiter, app, get_knowledge_service


class FakeService:
    def __init__(self):
        self.documents = []
        self.chat_result = {
            "answer": "答案 [资料 1]",
            "tool_events": [{"tool": "knowledge_search", "result": [{"source": "rag.md", "content": "依据"}]}],
            "sources": ["rag.md"],
            "retrieved_chunks": [{"source": "rag.md", "content": "依据"}],
        }
        self.fail_chat = False

    def chat(self, session_id, question, mode="agent", on_token=None, on_tool_event=None):
        if self.fail_chat:
            raise RuntimeError("sk-test-secret-must-not-leak")
        if on_token:
            on_token(self.chat_result["answer"])
        if on_tool_event:
            for event in self.chat_result["tool_events"]:
                on_tool_event({
                    "tool": event["tool"],
                    "arguments": {"question": question},
                    "result": event["result"],
                })
        return {**self.chat_result, "mode": mode}

    def upload_document(self, filename, content, max_size_bytes):
        if filename.endswith(".pdf"):
            raise ValueError("仅支持 .md 和 .txt 文件")
        result = {"document_id": "doc-1", "filename": filename, "chunk_count": 1, "updated": False}
        self.documents.append(result)
        return result

    def list_documents(self):
        return list(self.documents)

    def get_document(self, document_id):
        if document_id != "doc-1":
            raise KeyError("未找到该知识库文档")
        return {
            "document_id": document_id,
            "filename": "notes.md",
            "file_type": "md",
            "content": "## 预览内容\n文档正文",
        }

    def clear_session(self, session_id):
        return {"session_id": session_id, "cleared": True}


@pytest.fixture
def client():
    service = FakeService()
    app.dependency_overrides[get_knowledge_service] = lambda: service
    api_client = TestClient(app, raise_server_exceptions=False)
    # Isolate request limits between tests.
    import api.main as api_module
    previous_limiter = api_module._limiter
    api_module._limiter = RateLimiter()
    yield api_client, service
    api_module._limiter = previous_limiter
    app.dependency_overrides.clear()


def test_health_and_chat_response_shape(client):
    api_client, _service = client
    health = api_client.get("/health")
    assert health.json() == {
        "success": True,
        "message": "ok",
        "data": {"status": "ok"},
        "error": None,
    }
    response = api_client.post("/chat", json={
        "session_id": str(uuid.uuid4()),
        "question": "什么是 RAG？",
    })
    payload = response.json()
    assert response.status_code == 200
    assert payload["success"] is True
    assert payload["data"]["sources"] == ["rag.md"]
    assert payload["data"]["tool_events"][0]["tool"] == "knowledge_search"
    assert payload["data"]["mode"] == "agent"


def test_streaming_chat_emits_tokens_tool_trace_and_completion(client):
    api_client, _service = client
    response = api_client.post("/chat/stream", json={
        "session_id": str(uuid.uuid4()),
        "question": "RAG 是什么？",
        "mode": "agent",
    })

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert '"type": "token"' in response.text
    assert '"type": "tool"' in response.text
    assert '"type": "done"' in response.text


def test_upload_list_preview_and_no_document_deletion_route(client):
    api_client, _service = client
    uploaded = api_client.post(
        "/documents/upload",
        files={"file": ("notes.md", b"## RAG\nContent", "text/markdown")},
    )
    assert uploaded.status_code == 200
    assert uploaded.json()["data"]["filename"] == "notes.md"

    listed = api_client.get("/documents")
    assert listed.json()["data"][0]["document_id"] == "doc-1"

    preview = api_client.get("/documents/doc-1")
    assert preview.json()["data"]["content"] == "## 预览内容\n文档正文"

    deletion_attempt = api_client.delete("/documents/doc-1")
    assert deletion_attempt.status_code == 405


def test_api_returns_safe_errors_for_invalid_input_and_internal_failure(client):
    api_client, service = client
    invalid = api_client.post("/documents/upload", files={"file": ("notes.pdf", b"x")})
    assert invalid.status_code == 400
    assert invalid.json()["success"] is False

    service.fail_chat = True
    failed = api_client.post("/chat", json={
        "session_id": str(uuid.uuid4()),
        "question": "hello",
    })
    assert failed.status_code == 500
    assert "sk-test-secret" not in failed.text
    assert "D:\\" not in failed.text


def test_api_validation_and_session_clear(client):
    api_client, _service = client
    invalid = api_client.post("/chat", json={"session_id": "bad", "question": ""})
    assert invalid.status_code == 422
    assert invalid.json()["error"] == "请求参数无效，请检查必填字段和长度限制"
    session_id = str(uuid.uuid4())
    cleared = api_client.delete(f"/sessions/{session_id}")
    assert cleared.json()["data"]["cleared"] is True


def test_upload_request_size_limit(client, monkeypatch):
    import api.main as api_module

    api_client, _service = client
    monkeypatch.setattr(api_module, "MAX_UPLOAD_SIZE_MB", 1)
    response = api_client.post(
        "/documents/upload",
        files={"file": ("large.txt", b"x" * (1024 * 1024 + 1), "text/plain")},
    )
    assert response.status_code == 413
    assert response.json()["success"] is False


def test_chat_rate_limit_is_enforced_per_session(client, monkeypatch):
    import api.main as api_module

    api_client, _service = client
    monkeypatch.setattr(api_module, "CHAT_REQUESTS_PER_MINUTE", 1)
    payload = {"session_id": str(uuid.uuid4()), "question": "hello"}

    assert api_client.post("/chat", json=payload).status_code == 200
    limited = api_client.post("/chat", json=payload)
    assert limited.status_code == 429
    assert limited.json()["success"] is False
