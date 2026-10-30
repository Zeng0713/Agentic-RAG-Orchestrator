import pytest
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
        self.messages = []

    def invoke(self, messages, tools=None):
        self.messages.append(list(messages))
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


def test_run_agent_includes_previous_turns():
    llm = FakeLLM([FakeResponse(content="RAG 是检索增强生成。")])
    history = [
        {"role": "user", "content": "什么是 RAG？"},
        {"role": "assistant", "content": "RAG 是检索增强生成。"},
    ]

    agent.run_agent(
        question="它有哪些搭建步骤？",
        llm=llm,
        documents=[],
        vector_store=None,
        history=history,
    )

    sent_messages = llm.messages[0]
    assert sent_messages[-3:] == history + [
        {"role": "user", "content": "它有哪些搭建步骤？"}
    ]


def test_add_conversation_turn_keeps_only_six_turns():
    history = []
    for index in range(7):
        agent.add_conversation_turn(history, f"问题{index}", f"回答{index}")

    assert len(history) == 12
    assert history[0] == {"role": "user", "content": "问题1"}
    assert history[-1] == {"role": "assistant", "content": "回答6"}


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


def test_run_agent_handles_multiple_tool_rounds(monkeypatch):
    responses = [
        FakeResponse(tool_calls=[{
            "id": f"call_{index}",
            "name": "knowledge_search",
            "args": {"question": f"查询 {index}"},
        }])
        for index in range(1, 4)
    ] + [FakeResponse(content="汇总答案")]
    llm = FakeLLM(responses)
    executed = []

    def fake_execute_tool(tool_call, documents, vector_store):
        executed.append(tool_call["args"]["question"])
        return [{"content": "检索结果"}]

    monkeypatch.setattr(agent, "execute_tool", fake_execute_tool)

    result = agent.run_agent("问题", llm, [], None)

    assert result == "汇总答案"
    assert executed == ["查询 1", "查询 2", "查询 3"]


def test_run_agent_stops_after_six_tool_rounds(monkeypatch):
    responses = [
        FakeResponse(tool_calls=[{
            "id": f"call_{index}",
            "name": "knowledge_search",
            "args": {"question": f"查询 {index}"},
        }])
        for index in range(1, 7)
    ] + [FakeResponse(content="达到上限后的回答")]
    llm = FakeLLM(responses)
    executed = []

    def fake_execute_tool(tool_call, documents, vector_store):
        executed.append(tool_call["args"]["question"])
        return []

    monkeypatch.setattr(agent, "execute_tool", fake_execute_tool)

    result = agent.run_agent("问题", llm, [], None)

    assert result == "达到上限后的回答"
    assert len(executed) == agent.MAX_TOOL_ROUNDS == 6
    assert llm.index == 7


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


def test_execute_tool_rejects_missing_required_argument():
    tool_call = {
        "name": "knowledge_search",
        "args": {},
    }

    with pytest.raises(ValueError, match="缺少必填参数：question"):
        agent.execute_tool(tool_call, documents=[], vector_store=None)


def test_execute_tool_rejects_wrong_argument_type():
    tool_call = {
        "name": "knowledge_search",
        "args": {"question": 42},
    }

    with pytest.raises(ValueError, match="参数 question 类型错误"):
        agent.execute_tool(tool_call, documents=[], vector_store=None)


def test_execute_tool_rejects_unknown_argument():
    tool_call = {
        "name": "list_knowledge_sources",
        "args": {"path": "private"},
    }

    with pytest.raises(ValueError, match="包含未知参数：path"):
        agent.execute_tool(tool_call, documents=[], vector_store=None)
