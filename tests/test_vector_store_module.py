
import pytest

from langchain_core.documents import Document

from src import vector_store as vector_store_module
from src.document_loader import load_documents
from src.embeddings import create_embeddings
from src.vector_store import (
    create_vector_store,
    sync_documents,
)


class FakeVectorStore:
    """模拟 Chroma，不访问真实数据库。"""

    def __init__(
        self,
        existing_ids=None,
        existing_documents=None,
    ):
        self.existing_ids = existing_ids or []
        self.existing_documents = existing_documents or []

        self.add_calls = []
        self.update_calls = []

    def get(self):
        return {
            "ids": self.existing_ids,
            "documents": self.existing_documents,
        }

    def add_documents(self, documents, ids):
        self.add_calls.append({
            "documents": documents,
            "ids": ids,
        })

    def update_documents(self, ids, documents):
        self.update_calls.append({
            "ids": ids,
            "documents": documents,
        })


def make_document(content, source):
    return Document(
        page_content=content,
        metadata={"source": source},
    )


def test_create_vector_store(monkeypatch, tmp_path):
    """测试 Chroma 数据库创建。"""

    captured = {}

    class FakeChroma:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(
        vector_store_module,
        "Chroma",
        FakeChroma,
    )

    fake_embedding = object()
    db_path = tmp_path / "test_db"

    result = create_vector_store(
        embeddings=fake_embedding,
        db_path=db_path,
    )

    assert isinstance(result, FakeChroma)
    assert captured["collection_name"] == (
        vector_store_module.CHROMA_COLLECTION_NAME
    )
    assert captured["embedding_function"] is fake_embedding
    assert captured["persist_directory"] == str(db_path)


def test_sync_adds_new_documents():
    """测试新增知识片段。"""

    documents = [
        make_document(
            "Agent 内容 1",
            "data/documents/agent_intro.md",
        ),
        make_document(
            "Agent 内容 2",
            "data/documents/agent_intro.md",
        ),
        make_document(
            "RAG 内容",
            "data/documents/rag_intro.md",
        ),
    ]

    store = FakeVectorStore()

    sync_documents(store, documents)

    assert len(store.add_calls) == 1

    added = store.add_calls[0]

    assert added["ids"] == [
        "agent_intro_0",
        "agent_intro_1",
        "rag_intro_0",
    ]

    assert len(added["documents"]) == 3
    assert store.update_calls == []


def test_sync_updates_changed_documents():
    """测试更新已经存在但内容发生变化的文档。"""

    documents = [
        make_document(
            "更新后的 Agent 内容",
            "data/documents/agent_intro.md",
        )
    ]

    store = FakeVectorStore(
        existing_ids=["agent_intro_0"],
        existing_documents=["旧的 Agent 内容"],
    )

    sync_documents(store, documents)

    assert len(store.update_calls) == 1

    updated = store.update_calls[0]

    assert updated["ids"] == ["agent_intro_0"]

    assert (
        updated["documents"][0].page_content
        == "更新后的 Agent 内容"
    )

    assert store.add_calls == []


def test_sync_skips_unchanged_documents():
    """测试内容没有变化时不重复写入。"""

    documents = [
        make_document(
            "Agent 内容",
            "data/documents/agent_intro.md",
        )
    ]

    store = FakeVectorStore(
        existing_ids=["agent_intro_0"],
        existing_documents=["Agent 内容"],
    )

    sync_documents(store, documents)

    assert store.add_calls == []
    assert store.update_calls == []


def test_sync_handles_new_and_updated_documents():
    """测试同时存在新增和更新的情况。"""

    documents = [
        make_document(
            "更新后的 Agent 内容",
            "data/documents/agent_intro.md",
        ),
        make_document(
            "新增的 RAG 内容",
            "data/documents/rag_intro.md",
        ),
    ]

    store = FakeVectorStore(
        existing_ids=["agent_intro_0"],
        existing_documents=["旧的 Agent 内容"],
    )

    sync_documents(store, documents)

    assert len(store.update_calls) == 1
    assert len(store.add_calls) == 1

    assert store.update_calls[0]["ids"] == [
        "agent_intro_0"
    ]

    assert store.add_calls[0]["ids"] == [
        "rag_intro_0"
    ]


# 真实集成测试：需要 SiliconFlow API
@pytest.mark.integration
def test_real_vector_store(tmp_path):

    documents = load_documents()

    embeddings = create_embeddings()

    test_db_path = tmp_path / "chroma_test"

    vector_store = create_vector_store(
        embeddings,
        db_path=test_db_path,
    )

    sync_documents(
        vector_store,
        documents,
    )

    data = vector_store.get()

    assert len(data["ids"]) == len(documents)

    assert len(data["documents"]) == len(documents)
