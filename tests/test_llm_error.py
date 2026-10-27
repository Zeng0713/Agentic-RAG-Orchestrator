import pytest

from langchain_core.documents import Document

from src.exceptions import LLMServiceError
from src.rag_chain import generate_answer


class FakeBrokenLLM:
    """
    模拟调用失败的 LLM。
    """

    def invoke(self, prompt):
        raise ConnectionError(
            "模拟网络连接失败"
        )


def test_generate_answer_llm_error():
    fake_llm = FakeBrokenLLM()

    test_docs = [
        Document(
            page_content=(
                "RAG 是检索增强生成。"
            ),
            metadata={
                "source": "test.md"
            },
        )
    ]

    with pytest.raises(
        LLMServiceError,
        match="大语言模型服务暂时不可用",
    ):
        generate_answer(
            "什么是 RAG？",
            test_docs,
            fake_llm,
        )