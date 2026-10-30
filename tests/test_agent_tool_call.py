import pytest

from src.agent import select_tool
from src.rag_chain import create_llm


@pytest.mark.integration
@pytest.mark.parametrize(
    "question, expected_tool",
    [
        (
            "什么是 RAG？",
            "knowledge_search",
        ),
        (
            "知识库里有哪些资料？",
            "list_knowledge_sources",
        ),
    ],
)
def test_real_tool_selection(
    question,
    expected_tool,
):
    llm = create_llm()

    response = select_tool(
        question=question,
        llm=llm,
    )

    assert response.tool_calls

    tool_call = response.tool_calls[0]

    assert tool_call["name"] == expected_tool