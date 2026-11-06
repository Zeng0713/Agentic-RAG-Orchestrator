"""Web-facing orchestration around the existing RAG and Agent modules."""

import hashlib
import json
import logging
import os
import re
import time
import uuid
from pathlib import Path

from src.agent import run_agent
from src.conversation_context import BudgetedLLM, rewrite_question, trim_history
from src.knowledge_lock import KnowledgeLock
from src.session_store import MemorySessionStore, normalize_session_id
from src.state_runtime import LeaseLost, StateError
from src.config import MAX_QUESTION_LENGTH, UPLOADS_DIR, validate_config
from src.document_loader import load_documents, parse_document_text
from src.embeddings import create_embeddings
from src.rag_chain import create_llm
from src.rag_chain import generate_answer, get_citations
from src.retriever import retrieve_documents
from src.upload_validation import validate_upload
from src.vector_store import create_vector_store, sync_documents

logger = logging.getLogger("rag_project")


class KnowledgeService:
    """Own a persistent knowledge index and isolated in-memory chat sessions."""

    def __init__(
        self,
        *,
        vector_store=None,
        llm=None,
        embeddings=None,
        documents_dir="data/documents",
        uploads_dir=UPLOADS_DIR,
        manifest_path=None,
        session_store=None,
        chat_deadline=300,
    ):
        self.documents_dir = Path(documents_dir)
        self.uploads_dir = Path(uploads_dir)
        self.uploads_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_path = Path(manifest_path or self.uploads_dir / "manifest.json")
        self._knowledge_lock = KnowledgeLock()
        self.session_store = session_store or MemorySessionStore()
        self.chat_deadline = chat_deadline

        self.documents = load_documents(self.documents_dir)
        if vector_store is None or llm is None:
            validate_config()
            embeddings = embeddings or create_embeddings()
            if vector_store is None:
                vector_store = create_vector_store(embeddings)
            sync_documents(vector_store, self.documents)
            llm = llm or create_llm()
        self.vector_store = vector_store
        self.llm = llm
        self._retriever = KnowledgeRetriever(self)

        if not self.manifest_path.exists():
            self._write_manifest({})
        self._load_uploaded_documents()

    def _read_manifest(self):
        if not self.manifest_path.exists():
            return {}
        try:
            data = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            logger.error("知识库清单读取失败（%s）", type(error).__name__)
            raise RuntimeError("知识库文档清单无法读取，请先备份并检查清单文件") from error
        if not isinstance(data, dict):
            raise RuntimeError("知识库文档清单格式无效")
        return data

    def _write_manifest(self, manifest):
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = self.manifest_path.with_suffix(".tmp")
        temp_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        os.replace(temp_path, self.manifest_path)

    def _load_uploaded_documents(self):
        for doc_id, entry in self._read_manifest().items():
            stored_name = entry.get("stored_name")
            if not stored_name or Path(stored_name).name != stored_name:
                continue
            stored_path = self.uploads_dir / stored_name
            if not stored_path.is_file():
                continue
            try:
                text = stored_path.read_text(encoding="utf-8")
                self.documents.extend(parse_document_text(
                    entry["filename"],
                    text,
                    metadata={"source": entry["filename"], "document_id": doc_id, "origin": "upload"},
                ))
            except (OSError, ValueError, KeyError) as error:
                logger.error("上传文档重载失败（%s）", type(error).__name__)

    @staticmethod
    def _document_id(filename):
        digest = hashlib.sha256(filename.casefold().encode("utf-8")).hexdigest()
        return str(uuid.UUID(digest[:32]))

    def list_documents(self):
        with self._knowledge_lock.read():
            manifest = self._read_manifest()
            items = []
            for path in sorted(self.documents_dir.glob("*.md")):
                items.append({
                    "document_id": f"static-{self._document_id(path.name)}",
                    "filename": path.name,
                    "file_type": "md",
                    "chunk_count": sum(
                        1 for doc in self.documents
                        if Path(doc.metadata.get("source", "")).name == path.name
                    ),
                    "origin": "内置",
                })
            for doc_id, entry in manifest.items():
                items.append({
                    "document_id": doc_id,
                    "filename": entry["filename"],
                    "file_type": entry["file_type"],
                    "chunk_count": len(entry.get("chunk_ids", [])),
                    "origin": "上传",
                })
            return items

    def get_document(self, document_id):
        with self._knowledge_lock.read():
            manifest = self._read_manifest()
            entry = manifest.get(document_id)
            if entry:
                stored_name = entry.get("stored_name", "")
                if not stored_name or Path(stored_name).name != stored_name:
                    raise RuntimeError("文档存储记录无效")
                path = self.uploads_dir / stored_name
                try:
                    content = path.read_text(encoding="utf-8")
                except OSError as error:
                    raise KeyError("未找到该知识库文档") from error
                return {
                    "document_id": document_id,
                    "filename": entry["filename"],
                    "file_type": entry["file_type"],
                    "content": content,
                }

            for path in self.documents_dir.glob("*.md"):
                static_id = f"static-{self._document_id(path.name)}"
                if static_id == document_id:
                    try:
                        content = path.read_text(encoding="utf-8")
                    except OSError as error:
                        raise KeyError("未找到该知识库文档") from error
                    return {
                        "document_id": document_id,
                        "filename": path.name,
                        "file_type": "md",
                        "content": content,
                    }
            raise KeyError("未找到该知识库文档")

    def upload_document(self, filename, content, max_size_bytes):
        safe_name, text = validate_upload(filename, content, max_size_bytes)
        doc_id = self._document_id(safe_name)
        extension = Path(safe_name).suffix.lower().lstrip(".")
        docs = parse_document_text(
            safe_name,
            text,
            metadata={
                "source": safe_name,
                "document_id": doc_id,
                "origin": "upload",
            },
        )
        if not docs:
            raise ValueError("文件中没有可索引的文本内容")

        prepared = (self.vector_store.prepare_documents(docs)
                    if hasattr(self.vector_store, "prepare_documents") else None)
        with self._knowledge_lock.write():
            manifest = self._read_manifest()
            old_entry = manifest.get(doc_id)
            version = uuid.uuid4().hex
            chunk_ids = [f"upload_{doc_id}_{version}_{index}" for index in range(len(docs))]
            stored_name = f"{doc_id}_{version}.{extension}"
            stored_path = self.uploads_dir / stored_name
            stored_path.write_text(text, encoding="utf-8")
            try:
                if prepared is not None:
                    self.vector_store.add_prepared_documents(docs, chunk_ids, prepared)
                else:
                    self.vector_store.add_documents(documents=docs, ids=chunk_ids)
                updated_manifest = dict(manifest)
                updated_manifest[doc_id] = {
                    "filename": safe_name,
                    "file_type": extension,
                    "chunk_ids": chunk_ids,
                    "stored_name": stored_name,
                }
                self._write_manifest(updated_manifest)
            except Exception:
                try:
                    self.vector_store.delete(ids=chunk_ids)
                except Exception:
                    logger.error("上传失败后的向量回滚失败")
                stored_path.unlink(missing_ok=True)
                raise

            if old_entry:
                try:
                    if old_entry.get("chunk_ids"):
                        self.vector_store.delete(ids=old_entry["chunk_ids"])
                except Exception as error:
                    try:
                        self._write_manifest(manifest)
                        self.vector_store.delete(ids=chunk_ids)
                        stored_path.unlink(missing_ok=True)
                    except Exception as rollback_error:
                        logger.error(
                            "文档更新回滚失败（%s）",
                            type(rollback_error).__name__,
                        )
                    raise RuntimeError("文档更新失败，未能清理旧版本") from error
                old_file = self.uploads_dir / old_entry.get("stored_name", "")
                if old_file.parent == self.uploads_dir:
                    try:
                        old_file.unlink(missing_ok=True)
                    except OSError as error:
                        logger.error("旧上传文件清理失败（%s）", type(error).__name__)
            self.documents = [
                doc for doc in self.documents
                if doc.metadata.get("document_id") != doc_id
            ] + docs
            logger.info("知识文档已入库：%s（%d 个片段）", safe_name, len(docs))
            return {
                "document_id": doc_id,
                "filename": safe_name,
                "file_type": extension,
                "chunk_count": len(docs),
                "updated": old_entry is not None,
            }

    def chat(
        self, session_id, question, mode="agent", on_token=None,
        on_tool_event=None, on_ready=None, cancelled=None,
    ):
        if not question or not question.strip():
            raise ValueError("问题不能为空")
        if len(question) > MAX_QUESTION_LENGTH:
            raise ValueError(f"问题长度不能超过 {MAX_QUESTION_LENGTH} 个字符")
        if mode not in {"agent", "rag"}:
            raise ValueError("mode 必须是 agent 或 rag")
        lease = self.session_store.acquire(session_id)
        deadline = time.monotonic() + self.chat_deadline
        request_id = uuid.uuid4().hex
        logger.info("chat_started request_id=%s mode=%s", request_id, mode)

        def check():
            lease.ensure_owned()
            if cancelled is not None and cancelled.is_set():
                raise StateError("请求已中断")
            if time.monotonic() >= deadline:
                raise StateError("问答处理超时，请缩短问题后重试")

        def token(text):
            check()
            if on_token:
                on_token(text)

        events = []

        def record_tool_event(event):
            check()
            events.append(event)
            if on_tool_event:
                on_tool_event(event)

        try:
            history = trim_history(self.session_store.read(lease))
            if on_ready:
                on_ready()
            check()
            llm = BudgetedLLM(self.llm, check, deadline)
            with self._knowledge_lock.read():
                documents = list(self.documents)
            if mode == "agent":
                answer = run_agent(
                    question=question.strip(), llm=llm, documents=documents,
                    vector_store=self._retriever, history=history,
                    on_tool_event=record_tool_event,
                    on_token=token if on_token else None,
                )
                retrieved_chunks = []
                seen = set()
                for event in events:
                    for item in event.get("result", []):
                        if not isinstance(item, dict) or not item.get("content"):
                            continue
                        source = Path(str(item.get("source", "未知来源")).replace("\\", "/")).name
                        key = (source, item["content"])
                        if key not in seen:
                            seen.add(key)
                            retrieved_chunks.append({"source": source, "content": item["content"]})
                sources = list(dict.fromkeys(item["source"] for item in retrieved_chunks))
            else:
                query = rewrite_question(question.strip(), history, llm)
                retrieved_docs = retrieve_documents(query, documents, self._retriever)
                check()
                answer = generate_answer(
                    question.strip(), retrieved_docs, llm,
                    on_token=token if on_token else None, history=history,
                )
                retrieved_chunks = [
                    {"source": Path(str(doc.metadata.get("source", "未知来源")).replace("\\", "/")).name,
                     "content": doc.page_content}
                    for doc in retrieved_docs
                ]
                sources = []
                for citation in get_citations(answer, retrieved_docs):
                    match = re.match(r"- \[资料 \d+\] (.+)$", citation)
                    if match:
                        source = Path(match.group(1).replace("\\", "/")).name
                        if source not in sources:
                            sources.append(source)
            check()
            persistence, warning = "saved", None
            try:
                self.session_store.save(lease, request_id, question.strip(), answer)
            except LeaseLost:
                raise
            except StateError:
                persistence = "unconfirmed"
                warning = "回答已生成，但无法确认本轮对话是否保存；后续追问可能缺少这轮上下文。"
            logger.info("chat_finished request_id=%s history_status=%s", request_id, persistence)
            return {
                "answer": answer, "tool_events": events, "sources": sources,
                "retrieved_chunks": retrieved_chunks, "mode": mode,
                "history_status": persistence, "warning": warning,
                "request_id": request_id,
            }
        finally:
            lease.close()

    def clear_session(self, session_id):
        lease = self.session_store.acquire(session_id)
        try:
            self.session_store.clear(lease)
            return {"session_id": normalize_session_id(session_id), "cleared": True}
        finally:
            lease.close()


class KnowledgeRetriever:
    """Query embedding outside the read lock; title/vector reads share one view."""

    def __init__(self, service):
        self.service = service

    def retrieve(self, question, k=4, max_results=2):
        service = self.service
        store = service.vector_store
        embeddings = getattr(store, "embeddings", None)
        if embeddings is not None:
            vector = embeddings.embed_query(question)

            class VectorQuery:
                def similarity_search_with_score(self, _question, k):
                    return store.similarity_search_by_vector_with_relevance_scores(vector, k=k)

            search_store = VectorQuery()
        else:
            search_store = store
        with service._knowledge_lock.read():
            return retrieve_documents(question, list(service.documents), search_store,
                                      k=k, max_results=max_results)
