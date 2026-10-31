"""HTTP API for the Agentic RAG web application."""

import logging
import json
import threading
import time
from collections import defaultdict, deque
from queue import Queue
from typing import Literal

from fastapi import Depends, FastAPI, HTTPException, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from src.config import (
    CHAT_REQUESTS_PER_MINUTE,
    MAX_QUESTION_LENGTH,
    MAX_UPLOAD_SIZE_MB,
    UPLOADS_PER_MINUTE,
)
from src.exceptions import EmbeddingServiceError, LLMServiceError
from src.services.knowledge_service import KnowledgeService

logger = logging.getLogger("rag_project")
app = FastAPI(title="Agentic RAG Orchestrator API", version="1.0.0")


class ChatRequest(BaseModel):
    session_id: str = Field(min_length=36, max_length=36)
    question: str = Field(min_length=1, max_length=MAX_QUESTION_LENGTH)
    mode: Literal["agent", "rag"] = "agent"


class RateLimiter:
    def __init__(self):
        self._requests = defaultdict(deque)
        self._lock = threading.Lock()
        self._calls = 0

    def allow(self, key, limit, window=60):
        now = time.monotonic()
        with self._lock:
            self._calls += 1
            if self._calls % 256 == 0 and len(self._requests) > 10000:
                stale = [
                    tracked_key
                    for tracked_key, timestamps in self._requests.items()
                    if not timestamps or now - timestamps[-1] >= window
                ]
                for tracked_key in stale:
                    self._requests.pop(tracked_key, None)
                if len(self._requests) > 10000:
                    oldest = sorted(
                        self._requests,
                        key=lambda tracked_key: self._requests[tracked_key][-1]
                        if self._requests[tracked_key]
                        else 0,
                    )[:len(self._requests) - 10000]
                    for tracked_key in oldest:
                        self._requests.pop(tracked_key, None)
            recent = self._requests[key]
            while recent and now - recent[0] >= window:
                recent.popleft()
            if len(recent) >= limit:
                return False
            recent.append(now)
            return True


_limiter = RateLimiter()
_service = None
_service_lock = threading.Lock()


@app.middleware("http")
async def enforce_content_length_limit(request, call_next):
    raw_length = request.headers.get("content-length")
    if raw_length and raw_length.isdigit():
        maximum = (MAX_UPLOAD_SIZE_MB * 1024 * 1024 + 256 * 1024
                   if request.url.path == "/documents/upload" else 64 * 1024)
        if int(raw_length) > maximum:
            return error_response("请求体超过大小限制", 413)
    return await call_next(request)


def get_knowledge_service():
    global _service
    with _service_lock:
        if _service is None:
            _service = KnowledgeService()
        return _service


def success(data, message="ok"):
    return {"success": True, "message": message, "data": data, "error": None}


def error_response(message, status_code):
    return JSONResponse(
        status_code=status_code,
        content={"success": False, "message": "request failed", "data": None, "error": message},
    )


@app.exception_handler(HTTPException)
async def handle_http_exception(_request, exc):
    message = exc.detail if isinstance(exc.detail, str) else "请求无效"
    return error_response(message, exc.status_code)


@app.exception_handler(RequestValidationError)
async def handle_validation_error(_request, _exc):
    return error_response("请求参数无效，请检查必填字段和长度限制", 422)


@app.exception_handler(Exception)
async def handle_unexpected_error(_request, exc):
    # Never include provider exception text: SDK errors can contain request details.
    logger.error("API 请求失败（%s）", type(exc).__name__)
    return error_response("服务暂时不可用，请稍后重试", 500)


def _rate_key(request, session_id=None):
    host = request.client.host if request.client else "unknown"
    return f"{host}:{session_id or ''}"


@app.get("/health")
def health():
    return success({"status": "ok"})


@app.post("/chat")
def chat(payload: ChatRequest, request: Request, service=Depends(get_knowledge_service)):
    key = _rate_key(request, payload.session_id)
    if not _limiter.allow(key, CHAT_REQUESTS_PER_MINUTE):
        raise HTTPException(status_code=429, detail="提问过于频繁，请稍后再试")
    try:
        result = service.chat(
            payload.session_id,
            payload.question,
            mode=payload.mode,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except (EmbeddingServiceError, LLMServiceError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return success(result)


@app.post("/chat/stream")
def chat_stream(
    payload: ChatRequest,
    request: Request,
    service=Depends(get_knowledge_service),
):
    key = _rate_key(request, payload.session_id)
    if not _limiter.allow(key, CHAT_REQUESTS_PER_MINUTE):
        raise HTTPException(status_code=429, detail="提问过于频繁，请稍后再试")

    events = Queue()

    def emit(event):
        encoded = json.dumps(event, ensure_ascii=False)
        return f"data: {encoded}\n\n"

    def run_chat():
        try:
            result = service.chat(
                payload.session_id,
                payload.question,
                mode=payload.mode,
                on_token=lambda token: events.put({"type": "token", "text": token}),
                on_tool_event=lambda event: events.put({
                    "type": "tool",
                    "tool": event.get("tool", "unknown"),
                    "arguments": event.get("arguments", {}),
                    "result_count": len(event.get("result", []))
                    if isinstance(event.get("result"), list) else 0,
                    "sources": [
                        item.get("source")
                        for item in event.get("result", [])
                        if isinstance(item, dict) and item.get("source")
                    ] if isinstance(event.get("result"), list) else [],
                }),
            )
            events.put({
                "type": "done",
                "sources": result["sources"],
                "chunks": result["retrieved_chunks"],
                "mode": result["mode"],
            })
        except Exception as exc:
            logger.error("流式问答失败（%s）", type(exc).__name__)
            events.put({"type": "error", "message": "服务暂时不可用，请稍后重试"})
        finally:
            events.put(None)

    def event_stream():
        threading.Thread(target=run_chat, daemon=True).start()
        while True:
            event = events.get()
            if event is None:
                break
            yield emit(event)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/documents/upload")
def upload_document(
    request: Request,
    file: UploadFile,
    service=Depends(get_knowledge_service),
):
    if not _limiter.allow(_rate_key(request), UPLOADS_PER_MINUTE):
        raise HTTPException(status_code=429, detail="上传过于频繁，请稍后再试")
    max_bytes = MAX_UPLOAD_SIZE_MB * 1024 * 1024
    content = file.file.read(max_bytes + 1)
    if len(content) > max_bytes:
        raise HTTPException(status_code=413, detail=f"文件不能超过 {MAX_UPLOAD_SIZE_MB} MB")
    try:
        result = service.upload_document(file.filename or "", content, max_bytes)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except EmbeddingServiceError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    finally:
        file.file.close()
    return success(result, "文档已处理并加入知识库")


@app.get("/documents")
def list_documents(service=Depends(get_knowledge_service)):
    return success(service.list_documents())


@app.get("/documents/{document_id}")
def get_document(document_id: str, service=Depends(get_knowledge_service)):
    try:
        result = service.get_document(document_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return success(result)


@app.delete("/sessions/{session_id}")
def clear_session(session_id: str, service=Depends(get_knowledge_service)):
    try:
        result = service.clear_session(session_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return success(result, "会话已清空")
