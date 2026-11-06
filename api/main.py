"""HTTP API for the Agentic RAG web application."""

import logging
import json
import threading
import time
from contextlib import asynccontextmanager
from queue import Empty, Full, Queue
from typing import Literal

from fastapi import Depends, FastAPI, HTTPException, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from src.config import (
    CHAT_REQUESTS_PER_MINUTE,
    CHAT_IP_REQUESTS_PER_MINUTE,
    MAX_QUESTION_LENGTH,
    MAX_UPLOAD_SIZE_MB,
    UPLOADS_PER_MINUTE,
)
from src.exceptions import EmbeddingServiceError, LLMServiceError
from src.services.knowledge_service import KnowledgeService
from src.rate_limiter import MemoryRateLimiter as RateLimiter
from src.state_backends import StateBackends
from src.session_store import normalize_session_id
from src.state_runtime import StateError

logger = logging.getLogger("rag_project")


@asynccontextmanager
async def lifespan(_app):
    global _backends, _limiter, _service
    _backends = StateBackends()
    _limiter = _backends.limiter
    try:
        yield
    finally:
        _backends.close()
        _backends = None
        _service = None
        _limiter = RateLimiter()


app = FastAPI(title="Agentic RAG Orchestrator API", version="1.0.0", lifespan=lifespan)


class ChatRequest(BaseModel):
    session_id: str = Field(min_length=36, max_length=36)
    question: str = Field(min_length=1, max_length=MAX_QUESTION_LENGTH)
    mode: Literal["agent", "rag"] = "agent"


_limiter = RateLimiter()
_service = None
_service_lock = threading.Lock()
_backends = None
_state_lock = threading.Lock()


def get_state_backends():
    global _backends
    with _state_lock:
        if _backends is None:
            _backends = StateBackends()
        return _backends


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
            _service = KnowledgeService(
                session_store=get_state_backends().sessions,
                chat_deadline=get_state_backends().settings.chat_deadline,
            )
        return _service


def success(data, message="ok"):
    return {"success": True, "message": message, "data": data, "error": None}


def error_response(message, status_code, code=None, headers=None):
    return JSONResponse(
        status_code=status_code,
        content={"success": False, "message": "request failed", "data": None, "error": message,
                 "error_code": code or "REQUEST_FAILED"},
        headers=headers,
    )


@app.exception_handler(HTTPException)
async def handle_http_exception(_request, exc):
    message = exc.detail if isinstance(exc.detail, str) else "请求无效"
    return error_response(message, exc.status_code,
                          "RATE_LIMITED" if exc.status_code == 429 else None, exc.headers)


@app.exception_handler(StateError)
async def handle_state_error(_request, exc):
    return error_response(str(exc), exc.status_code, exc.code)


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


def check_limit(limits):
    decision = _limiter.check(limits)
    if not decision.allowed:
        raise HTTPException(status_code=429, detail="请求过于频繁，请稍后再试",
                            headers={"Retry-After": str(decision.retry_after)})


def check_chat_limit(request, session_id):
    try:
        sid = normalize_session_id(session_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from None
    check_limit([
        ("chat-session:" + _rate_key(request, sid), CHAT_REQUESTS_PER_MINUTE),
        ("chat-ip:" + _rate_key(request), CHAT_IP_REQUESTS_PER_MINUTE),
    ])


def chat_admission(payload: ChatRequest, request: Request):
    # Execute before constructing the knowledge service or touching providers.
    check_chat_limit(request, payload.session_id)


def upload_admission(request: Request):
    check_limit([("upload:" + _rate_key(request), UPLOADS_PER_MINUTE)])


@app.get("/ready")
def ready():
    get_state_backends().ready()
    return success({"status": "ready"})


@app.get("/health")
def health():
    return success({"status": "ok"})


@app.post("/chat", dependencies=[Depends(chat_admission)])
def chat(payload: ChatRequest, request: Request, service=Depends(get_knowledge_service)):
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


@app.post("/chat/stream", dependencies=[Depends(chat_admission)])
def chat_stream(payload: ChatRequest, request: Request, service=Depends(get_knowledge_service)):
    events = Queue(maxsize=64)
    cancelled = threading.Event()
    finished = threading.Event()
    deadline = time.monotonic() + getattr(service, "chat_deadline", 300)

    def enqueue(event):
        while not cancelled.is_set():
            if time.monotonic() >= deadline:
                raise StateError("问答处理超时，请稍后重试")
            try:
                events.put(event, timeout=0.25)
                return
            except Full:
                continue
        raise StateError("请求已中断")

    def run_chat():
        try:
            result = service.chat(
                payload.session_id, payload.question, mode=payload.mode,
                on_ready=lambda: enqueue({"type": "ready"}),
                cancelled=cancelled,
                on_token=lambda token: enqueue({"type": "token", "text": token}),
                on_tool_event=lambda event: enqueue({
                    "type": "tool", "tool": event.get("tool", "unknown"),
                    "arguments": event.get("arguments", {}),
                    "result_count": len(event.get("result", []))
                    if isinstance(event.get("result"), list) else 0,
                    "sources": [item.get("source") for item in event.get("result", [])
                                if isinstance(item, dict) and item.get("source")]
                    if isinstance(event.get("result"), list) else [],
                }),
            )
            enqueue({"type": "done", "sources": result["sources"],
                     "chunks": result["retrieved_chunks"], "mode": result["mode"],
                     "history_status": result.get("history_status", "saved"),
                     "warning": result.get("warning"), "request_id": result.get("request_id")})
        except Exception as exc:
            logger.warning("stream_failed error_type=%s", type(exc).__name__)
            if isinstance(exc, StateError):
                status, code, message = exc.status_code, exc.code, str(exc)
            elif isinstance(exc, ValueError):
                status, code, message = 400, "INVALID_REQUEST", str(exc)
            else:
                status, code, message = 503, "GENERATION_FAILED", "问答服务暂时不可用，请稍后重试"
            if not cancelled.is_set():
                try:
                    events.put({"type": "error", "message": message, "error_code": code, "status": status},
                               timeout=0.1)
                except Full:
                    pass
        finally:
            finished.set()

    threading.Thread(target=run_chat, daemon=True).start()
    try:
        first = events.get(timeout=5)
    except Empty:
        cancelled.set()
        raise StateError("会话初始化超时，请稍后重试") from None
    if not first or first.get("type") != "ready":
        cancelled.set()
        return error_response(first.get("message", "服务暂时不可用") if first else "服务暂时不可用",
                              first.get("status", 503) if first else 503,
                              first.get("error_code") if first else None)

    def event_stream():
        try:
            while True:
                try:
                    event = events.get(timeout=10)
                except Empty:
                    if finished.is_set():
                        yield "data: " + json.dumps({"type": "error", "message": "回答未完成，请重试"}, ensure_ascii=False) + "\n\n"
                        break
                    yield ": keepalive\n\n"
                    continue
                if event is None:
                    break
                yield "data: " + json.dumps(event, ensure_ascii=False) + "\n\n"
                if event.get("type") in {"done", "error"}:
                    break
        finally:
            cancelled.set()

    return StreamingResponse(event_stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.post("/documents/upload", dependencies=[Depends(upload_admission)])
def upload_document(
    request: Request,
    file: UploadFile,
    service=Depends(get_knowledge_service),
):
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
