import time

import pytest
from langchain_core.documents import Document

from src.conversation_context import BudgetedLLM, INPUT_BUDGET, size, trim_history
from src.exceptions import LLMServiceError
from src.rag_chain import generate_answer
from src.services.knowledge_service import KnowledgeService


def test_rag_rewrites_followup_before_retrieval_and_uses_history(tmp_path, monkeypatch):
    queries = []
    prompts = []

    class LLM:
        def invoke(self, prompt, **kwargs):
            prompts.append(prompt)
            text = "混合检索有哪些缺点？" if isinstance(prompt, list) else "关键词与语义结合 [资料 1]"
            return type("Response", (), {"content": text})()

    def retrieve(question, *_args, **_kwargs):
        queries.append(question)
        return [Document(page_content="## 混合检索\n说明", metadata={"source": "rag.md"})]

    monkeypatch.setattr("src.services.knowledge_service.retrieve_documents", retrieve)
    service = KnowledgeService(vector_store=object(), llm=LLM(),
                               documents_dir=tmp_path / "docs", uploads_dir=tmp_path / "uploads")
    import uuid
    sid = str(uuid.uuid4())
    service.chat(sid, "混合检索是什么？", mode="rag")
    result = service.chat(sid, "它有什么缺点？", mode="rag")
    assert queries == ["混合检索是什么？", "混合检索有哪些缺点？"]
    assert "混合检索是什么？" in prompts[-1]
    assert "它有什么缺点？" in prompts[-1]
    assert result["sources"] == ["rag.md"]


def test_prompt_budget_clips_large_chunks_without_mutating_documents():
    captured = []

    class LLM:
        def invoke(self, prompt):
            captured.append(prompt)
            return type("Response", (), {"content": "answer"})()

    doc = Document(page_content="知识" * 20000, metadata={"source": "rag.md"})
    generate_answer("问题", [doc], LLM(), history=[
        {"role": "user", "content": "old"}, {"role": "assistant", "content": "answer" * 3000},
    ])
    assert size(captured[0]) <= INPUT_BUDGET
    assert len(doc.page_content) == 40000


def test_long_latest_turn_keeps_context_and_request_timeout_is_bounded():
    history = [{"role": "user", "content": "混合检索是什么"},
               {"role": "assistant", "content": "解释" * 5000}]
    trimmed = trim_history(history)
    assert trimmed[0]["content"] == "混合检索是什么"
    assert size(trimmed) <= 2000
    assert history[1]["content"] == "解释" * 5000
    captured = {}

    class LLM:
        def invoke(self, _messages, **kwargs):
            captured.update(kwargs)
            return object()

    BudgetedLLM(LLM(), lambda: None, time.monotonic() + 100).invoke("q", timeout=10)
    assert captured["timeout"] == 10
    assert captured["max_tokens"] == 2048


def test_rewrite_failure_is_explicit_not_a_pronoun_search():
    from src.conversation_context import rewrite_question

    class Broken:
        def invoke(self, *_args, **_kwargs):
            raise RuntimeError("secret-provider-detail")

    with pytest.raises(LLMServiceError, match="追问理解服务") as error:
        rewrite_question("它呢", [{"role": "user", "content": "RAG"},
                                 {"role": "assistant", "content": "解释"}], Broken())
    assert "secret-provider-detail" not in str(error.value)


def test_agent_prunes_complete_old_tool_groups_and_keeps_latest_evidence():
    from langchain_core.messages import AIMessage
    from src.agent import _invoke

    def call(call_id):
        return AIMessage(content="", tool_calls=[{
            "id": call_id, "name": "knowledge_search", "args": {"question": "q"},
        }])

    messages = [{"role": "system", "content": "rules"}, {"role": "user", "content": "q"},
                call("old"), {"role": "tool", "tool_call_id": "old", "content": "x" * 6000},
                call("latest"), {"role": "tool", "tool_call_id": "latest", "content": "evidence" * 300}]

    class LLM:
        def invoke(self, sent):
            assert [m["tool_call_id"] for m in sent if isinstance(m, dict) and m.get("role") == "tool"] == ["latest"]
            return AIMessage(content="answer")

    assert _invoke(LLM(), messages).content == "answer"


@pytest.mark.parametrize("streaming", [False, True])
def test_provider_failures_are_sanitized_for_both_modes(streaming, caplog):
    class Broken:
        def invoke(self, *_args, **_kwargs):
            raise ValueError("redis://user:private-password@server secret-api-key")

        def stream(self, *_args, **_kwargs):
            raise ConnectionError("private-password secret-api-key")

    wrapper = BudgetedLLM(Broken(), lambda: None, time.monotonic() + 20)
    with pytest.raises(LLMServiceError) as error:
        list(wrapper.stream("q")) if streaming else wrapper.invoke("q")
    assert "private-password" not in str(error.value) + caplog.text
    assert "secret-api-key" not in str(error.value) + caplog.text
