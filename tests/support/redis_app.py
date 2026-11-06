"""Offline two-process API fixture: python -m uvicorn tests.support.redis_app:app.

Uses real Redis but no provider calls, production uploads or production Chroma.
"""

import atexit
import json
import os
from pathlib import Path
import tempfile
import threading
import time

from langchain_core.messages import AIMessage

from api.main import app, get_knowledge_service, get_state_backends
from src.services.knowledge_service import KnowledgeService

_directory = tempfile.TemporaryDirectory(prefix="rag-offline-api-", dir=os.getenv("RAG_TEST_STORAGE_DIR"))
atexit.register(_directory.cleanup)
_service = None
_lock = threading.Lock()


class OfflineLLM:
    def invoke(self, messages, **_kwargs):
        users = [m["content"] for m in messages if isinstance(m, dict) and m.get("role") == "user"]
        if users[-1] == "__block__":
            gate = Path(os.environ["RAG_TEST_GATE_DIR"])
            (gate / "entered").touch()
            until = time.monotonic() + 15
            while not (gate / "release").exists():
                if time.monotonic() > until:
                    raise RuntimeError("test gate timeout")
                time.sleep(0.02)
        return AIMessage(content=json.dumps(users, ensure_ascii=False))


def offline_service():
    global _service
    with _lock:
        if _service is None:
            root = Path(_directory.name)
            _service = KnowledgeService(
                llm=OfflineLLM(), vector_store=object(),
                session_store=get_state_backends().sessions,
                documents_dir=root / "docs", uploads_dir=root / "uploads",
            )
        return _service


app.dependency_overrides[get_knowledge_service] = offline_service
