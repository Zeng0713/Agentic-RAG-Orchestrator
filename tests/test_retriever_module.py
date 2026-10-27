import pytest
from langchain_core.documents import Document
from src.retriever import retrieve_documents


class FakeVectorStore:
    """
    模拟向量数据库，
    避免测试时调用真实 Embedding API。
    """

    def __init__(self, results):
        self.results = results

    def similarity_search_with_score(
        self,
        question,
        k=4,
    ):
        return self.results[:k]


@pytest.mark.parametrize(
    "question, expected_title",
    [
        (
            "什么是 Agent？",
            "## 什么是 Agent",
        ),
        (
            "什么是 RAG？",
            "## 什么是 RAG",
        ),
        (
            "向量数据库有什么作用？",
            "## 什么是向量数据库",
        ),
        (
            "Agent 的执行循环有什么作用？",
            "## Agent 的执行循环",
        ),
    ],
)
def test_title_match_retrieval(
    question,
    expected_title,
):
    # Arrange

    documents = [
        Document(
            page_content=(
                "## 什么是 Agent\n\n"
                "Agent 是能够执行任务的系统。"
            )
        ),
        Document(
            page_content=(
                "## 什么是 RAG\n\n"
                "RAG 是检索增强生成。"
            )
        ),
        Document(
            page_content=(
                "## 什么是向量数据库\n\n"
                "向量数据库用于存储和检索向量。"
            )
        ),
        Document(
            page_content=(
                "## Agent 的执行循环\n\n"
                "执行循环负责协调模型和工具。"
            )
        ),
    ]

    # 故意让向量检索结果不可靠
    vector_results = [
        (
            documents[0],
            0.8,
        )
    ]

    fake_vector_store = FakeVectorStore(
        vector_results
    )

    # Act

    results = retrieve_documents(
        question=question,
        documents=documents,
        vector_store=fake_vector_store,
    )

    # Assert

    first_title = (
        results[0]
        .page_content
        .splitlines()[0]
    )

    assert first_title == expected_title