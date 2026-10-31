import uuid

from langchain_core.documents import Document

import src.services.knowledge_service as service_module
from src.services.knowledge_service import KnowledgeService
from src.vector_store import create_vector_store


class FakeVectorStore:
    def __init__(self):
        self.records = {}
        self.fail_next_add = False

    def add_documents(self, documents, ids):
        if self.fail_next_add:
            self.fail_next_add = False
            raise RuntimeError("simulated indexing failure")
        for doc, record_id in zip(documents, ids):
            self.records[record_id] = doc

    def delete(self, ids):
        for record_id in ids:
            self.records.pop(record_id, None)

    def get(self, include=None):
        return {
            "ids": list(self.records),
            "metadatas": [doc.metadata for doc in self.records.values()],
        }


class FakeLLM:
    def __init__(self):
        self.messages = []

    def invoke(self, messages, tools=None):
        self.messages.append(list(messages))
        return type("Response", (), {"content": "基于知识库的回答", "tool_calls": []})()


def make_service(tmp_path, vector_store=None, llm=None):
    return KnowledgeService(
        vector_store=vector_store or FakeVectorStore(),
        llm=llm or FakeLLM(),
        documents_dir=tmp_path / "documents",
        uploads_dir=tmp_path / "uploads",
    )


def test_upload_index_replace_and_restart(tmp_path):
    vector_store = FakeVectorStore()
    service = make_service(tmp_path, vector_store=vector_store)
    max_bytes = 1024 * 1024

    first = service.upload_document("rag.txt", b"RAG first version", max_bytes)
    first_ids = set(vector_store.records)
    assert first["chunk_count"] == 1
    assert service.list_documents()[0]["filename"] == "rag.txt"
    assert service.get_document(first["document_id"])["content"] == "RAG first version"

    updated = service.upload_document(
        "rag.txt",
        b"RAG updated version\n\nSecond paragraph",
        max_bytes,
    )
    assert updated["updated"] is True
    assert first_ids.isdisjoint(vector_store.records)
    assert len(vector_store.records) == updated["chunk_count"]

    restarted = make_service(tmp_path, vector_store=vector_store)
    assert restarted.list_documents()[0]["document_id"] == first["document_id"]
    assert len(restarted.documents) == updated["chunk_count"]
    assert set(vector_store.records) != set()
    assert restarted.get_document(first["document_id"])["content"].startswith("RAG updated")


def test_failed_update_keeps_previous_document_version(tmp_path):
    vector_store = FakeVectorStore()
    service = make_service(tmp_path, vector_store=vector_store)
    first = service.upload_document("stable.md", b"## Original\nold text", 1024)
    previous_ids = set(vector_store.records)
    vector_store.fail_next_add = True

    import pytest
    with pytest.raises(RuntimeError, match="simulated indexing failure"):
        service.upload_document("stable.md", b"## Replacement\nnew text", 1024)

    assert set(vector_store.records) == previous_ids
    assert service.list_documents()[0]["document_id"] == first["document_id"]
    assert len(list((tmp_path / "uploads").glob("*.md"))) == 1


def test_session_histories_are_isolated(tmp_path):
    llm = FakeLLM()
    service = make_service(tmp_path, llm=llm)
    session_a, session_b = str(uuid.uuid4()), str(uuid.uuid4())

    service.chat(session_a, "A 的第一轮")
    service.chat(session_b, "B 的独立对话")
    service.chat(session_a, "A 的追问")

    third_request = llm.messages[2]
    assert any(message.get("content") == "A 的第一轮" for message in third_request)
    assert not any(message.get("content") == "B 的独立对话" for message in third_request)

    service.clear_session(session_a)
    service.chat(session_a, "清空后的新问题")
    fourth_request = llm.messages[3]
    assert not any(message.get("content") == "A 的第一轮" for message in fourth_request)


def test_rag_mode_streams_answer_and_returns_only_cited_sources(tmp_path, monkeypatch):
    source_doc = Document(
        page_content="## RAG\nRAG 内容",
        metadata={"source": r"C:\\knowledge\\rag_intro.md"},
    )
    monkeypatch.setattr(service_module, "retrieve_documents", lambda *_args: [source_doc])

    def fake_generate_answer(_question, _docs, _llm, on_token=None):
        if on_token:
            on_token("RAG 是检索增强生成 [资料 1]。")
        return "RAG 是检索增强生成 [资料 1]。"

    monkeypatch.setattr(service_module, "generate_answer", fake_generate_answer)
    service = make_service(tmp_path)
    streamed = []

    result = service.chat(
        str(uuid.uuid4()),
        "什么是 RAG？",
        mode="rag",
        on_token=streamed.append,
    )

    assert result["mode"] == "rag"
    assert result["sources"] == ["rag_intro.md"]
    assert result["retrieved_chunks"] == [{
        "source": "rag_intro.md",
        "content": "## RAG\nRAG 内容",
    }]
    assert result["tool_events"] == []
    assert streamed == ["RAG 是检索增强生成 [资料 1]。"]


def test_chroma_data_persists_after_reopening(tmp_path):
    class FakeEmbeddings:
        def embed_documents(self, texts):
            return [[float(len(text)), 1.0, 0.0] for text in texts]

        def embed_query(self, text):
            return [float(len(text)), 1.0, 0.0]

    path = tmp_path / "persistent-chroma"
    store = create_vector_store(FakeEmbeddings(), db_path=path)
    store.add_documents(
        [Document(page_content="persistent content", metadata={"source": "memory.md"})],
        ids=["persistent_0"],
    )

    reopened = create_vector_store(FakeEmbeddings(), db_path=path)

    assert reopened.get()["ids"] == ["persistent_0"]
