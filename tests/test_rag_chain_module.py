import pytest
from langchain_core.documents import Document
import src.rag_chain as rag_chain_module
from src.exceptions import LLMServiceError

class FakeResponse:
    def __init__(self, content):
        self.content = content


class FakeLLM:
    def __init__(self, content=None, error=None):
        self.content = content
        self.error = error
        self.last_prompt = None

    def invoke(self, prompt):
        self.last_prompt = prompt

        if self.error:
            raise self.error

        return FakeResponse(self.content)


def make_document(content, source=None):
    metadata = {}

    if source is not None:
        metadata["source"] = source

    return Document(
        page_content=content,
        metadata=metadata,
    )

# 1. 测试 LLM 初始化
def test_create_llm(monkeypatch):
    captured = {}

    class FakeChatOpenAI:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(
        rag_chain_module,
        "ChatOpenAI",
        FakeChatOpenAI,
    )

    result = rag_chain_module.create_llm()

    assert isinstance(result, FakeChatOpenAI)

    assert captured["model"] == (
        rag_chain_module.LLM_MODEL
    )

    assert captured["base_url"] == (
        rag_chain_module.LLM_BASE_URL
    )

    assert captured["api_key"] == (
        rag_chain_module.DEEPSEEK_API_KEY
    )

# 2. 测试正常生成回答
def test_generate_answer():
    documents = [
        make_document(
            "## 什么是 RAG\nRAG 是检索增强生成。",
            "rag_intro.md",
        )
    ]

    fake_llm = FakeLLM(
        content="RAG 是检索增强生成。[资料 1]"
    )

    answer = rag_chain_module.generate_answer(
        question="什么是 RAG？",
        retrieved_docs=documents,
        llm=fake_llm,
    )

    assert answer == (
        "RAG 是检索增强生成。[资料 1]"
    )

    assert "什么是 RAG？" in fake_llm.last_prompt

    assert "rag_intro.md" in fake_llm.last_prompt

    assert "RAG 是检索增强生成" in fake_llm.last_prompt


# 3. 测试多个文档的上下文构建
def test_generate_answer_multiple_documents():
    documents = [
        make_document(
            "Agent 是能够自主执行任务的系统。",
            "agent_intro.md",
        ),
        make_document(
            "RAG 是检索增强生成。",
            "rag_intro.md",
        ),
    ]

    fake_llm = FakeLLM(
        content="这是综合回答。"
    )

    answer = rag_chain_module.generate_answer(
        question="什么是 Agent 和 RAG？",
        retrieved_docs=documents,
        llm=fake_llm,
    )

    assert answer == "这是综合回答。"

    assert "[资料 1]" in fake_llm.last_prompt
    assert "[资料 2]" in fake_llm.last_prompt

    assert "agent_intro.md" in fake_llm.last_prompt
    assert "rag_intro.md" in fake_llm.last_prompt


# 4. 测试空文档
def test_generate_answer_empty_documents():
    fake_llm = FakeLLM(
        content="抱歉，当前知识库中的资料不足以回答这个问题。"
    )

    answer = rag_chain_module.generate_answer(
        question="什么是未知技术？",
        retrieved_docs=[],
        llm=fake_llm,
    )

    assert answer == (
        "抱歉，当前知识库中的资料不足以回答这个问题。"
    )

    assert "什么是未知技术？" in fake_llm.last_prompt


# 5. 测试缺少 source 的文档
def test_generate_answer_missing_source():
    documents = [
        make_document(
            "这是一段测试内容。"
        )
    ]

    fake_llm = FakeLLM(
        content="测试回答。"
    )

    rag_chain_module.generate_answer(
        question="测试问题",
        retrieved_docs=documents,
        llm=fake_llm,
    )

    assert "未知来源" in fake_llm.last_prompt


# 6. 测试 LLM API 异常处理
def test_generate_answer_api_error():
    fake_llm = FakeLLM(
        error=ConnectionError("模拟 DeepSeek 网络错误")
    )

    with pytest.raises(
        LLMServiceError,
        match="大语言模型服务暂时不可用",
    ):
        rag_chain_module.generate_answer(
            question="什么是 RAG？",
            retrieved_docs=[],
            llm=fake_llm,
        )


# 7. 测试引用提取
def test_get_citations():
    documents = [
        make_document(
            "Agent 内容",
            "agent_intro.md",
        ),
        make_document(
            "RAG 内容",
            "rag_intro.md",
        ),
        make_document(
            "Embedding 内容",
            "embedding_intro.md",
        ),
    ]

    answer = (
        "RAG 是检索增强生成。[资料 2]"
    )

    citations = rag_chain_module.get_citations(
        answer,
        documents,
    )

    assert citations == [
        "- [资料 2] rag_intro.md"
    ]

# 8. 测试引用缺少来源
def test_get_citations_missing_source():
    documents = [
        make_document("测试内容")
    ]

    answer = "这是测试回答。[资料 1]"

    citations = rag_chain_module.get_citations(
        answer,
        documents,
    )

    assert citations == [
        "- [资料 1] 未知来源"
    ]

# 9. 测试回答没有引用
def test_get_citations_without_references():
    documents = [
        make_document(
            "RAG 内容",
            "rag_intro.md",
        )
    ]

    citations = rag_chain_module.get_citations(
        "这是一个没有引用的回答。",
        documents,
    )

    assert citations == []


# 10. 测试多个引用
def test_get_citations_multiple_references():
    documents = [
        make_document("Agent 内容", "agent_intro.md"),
        make_document("RAG 内容", "rag_intro.md"),
    ]

    answer = (
        "Agent 是智能体。[资料 1]"
        "RAG 是检索增强生成。[资料 2]"
    )

    citations = rag_chain_module.get_citations(
        answer,
        documents,
    )

    assert citations == [
        "- [资料 1] agent_intro.md",
        "- [资料 2] rag_intro.md",
    ]