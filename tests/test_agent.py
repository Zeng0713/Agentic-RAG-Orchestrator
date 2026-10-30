from langchain_core.documents import Document

import src.agent as agent


class FakeResponse:
    def __init__(
        self,
        content="",
        tool_calls=None,
    ):
        self.content = content
        self.tool_calls = tool_calls or []


class FakeLLM:
    def __init__(self, responses):
        self.responses = responses
        self.index = 0

    def invoke(self, messages, tools=None):
        response = self.responses[self.index]
        self.index += 1
        return response


def test_run_agent_without_tool():
    llm = FakeLLM(
        [
            FakeResponse(
                content="你好！",
            )
        ]
    )

    result = agent.run_agent(
        question="你好",
        llm=llm,
        documents=[],
        vector_store=None,
    )

    assert result == "你好！"


def test_run_agent_with_tool(monkeypatch):
    first_response = FakeResponse(
        tool_calls=[
            {
                "id": "call_1",
                "name": "knowledge_search",
                "args": {
                    "question": "什么是 RAG？"
                },
            }
        ]
    )

    final_response = FakeResponse(
        content="RAG 是检索增强生成。"
    )

    llm = FakeLLM(
        [
            first_response,
            final_response,
        ]
    )

    def fake_execute_tool(
        tool_call,
        documents,
        vector_store,
    ):
        return [
            {
                "content": "RAG 是检索增强生成。",
                "source": "rag_intro.md",
            }
        ]

    monkeypatch.setattr(
        agent,
        "execute_tool",
        fake_execute_tool,
    )

    result = agent.run_agent(
        question="什么是 RAG？",
        llm=llm,
        documents=[],
        vector_store=None,
    )

    assert result == "RAG 是检索增强生成。"


def test_execute_list_knowledge_sources():
    documents = [
        Document(
            page_content="Agent",
            metadata={
                "source": "agent_intro.md"
            },
        ),
        Document(
            page_content="RAG",
            metadata={
                "source": "rag_intro.md"
            },
        ),
    ]

    tool_call = {
        "name": "list_knowledge_sources",
        "args": {},
    }

    result = agent.execute_tool(
        tool_call=tool_call,
        documents=documents,
        vector_store=None,
    )

    assert result == [
        "agent_intro.md",
        "rag_intro.md",
    ]