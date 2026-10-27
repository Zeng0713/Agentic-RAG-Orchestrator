from langchain_core.documents import Document

from src.rag_chain import generate_answer


class FakeResponse:
    def __init__(self, content):
        self.content = content


class FakeLLM:
    def invoke(self, prompt):
        return FakeResponse(
            "RAG 是检索增强生成。[资料 1]"
        )


def test_generate_answer():
    # Arrange
    documents = [
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

    fake_llm = FakeLLM()

    # Act
    answer = generate_answer(
        question="什么是 RAG？",
        retrieved_docs=documents,
        llm=fake_llm,
    )

    # Assert
    assert (
        answer
        == "RAG 是检索增强生成。[资料 1]"
    )