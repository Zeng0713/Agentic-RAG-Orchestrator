from langchain_core.documents import Document

import src.tools as tools


def test_knowledge_search(monkeypatch):
    fake_docs = [
        Document(
            page_content=(
                "## 什么是 RAG\n\n"
                "RAG 是检索增强生成。"
            ),
            metadata={
                "source": "rag_intro.md"
            },
        )
    ]

    def fake_retrieve_documents(
        question,
        documents,
        vector_store,
    ):
        return fake_docs

    monkeypatch.setattr(
        tools,
        "retrieve_documents",
        fake_retrieve_documents,
    )

    results = tools.knowledge_search(
        question="什么是 RAG？",
        documents=[],
        vector_store=None,
    )

    assert len(results) == 1

    assert (
        results[0]["content"]
        == fake_docs[0].page_content
    )

    assert (
        results[0]["source"]
        == "rag_intro.md"
    )