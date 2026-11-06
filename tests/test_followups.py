from pathlib import Path

from src.document_loader import load_documents
from src.followups import suggest_followups
from src.tools import knowledge_search


def test_followups_require_available_sources_and_exclude_asked_questions():
    docs = [{"filename": "rag_build_intro.md"}]
    asked = ["如何验证文档上传已经入库？"]
    result = suggest_followups("RAG 文档上传流程", "文档切分后入库", [], docs, asked)
    assert result and len(result) <= 3
    assert asked[0] not in result
    assert all("Agent" not in item for item in result)
    assert suggest_followups("RAG", "RAG", [], []) == []


def test_custom_document_headings_can_supply_followups():
    chunks = [{"source": "team.md", "content": "## 如何配置开发环境\n配置说明"}]
    assert suggest_followups("团队开发规范", "说明", chunks, [{"filename": "team.md"}]) == ["如何配置开发环境？"]


def test_agent_followups_are_related_and_do_not_repeat_current_question():
    question = "Agentic RAG 与传统 RAG 有什么区别？"
    result = suggest_followups(question, "Agent 使用工具调用", [], [{"filename": "agentic_rag_build_intro.md"}])
    assert question not in result
    assert "什么情况下应该选择传统 RAG 而不是 Agent？" in result


def test_homepage_questions_three_and_four_have_direct_searchable_evidence():
    documents = load_documents(Path(__file__).parents[1] / "data/documents")

    class EmptyVectorStore:
        def similarity_search_with_score(self, _question, k):
            return []

    for question, filename, expected in [
        ("从文档上传到向量入库，RAG 搭建要经过哪些步骤？", "rag_build_intro.md", "BGE-M3"),
        ("Agentic RAG 与传统 RAG 有什么区别？", "agentic_rag_build_intro.md", "流程控制"),
    ]:
        result = knowledge_search(question, documents, EmptyVectorStore())
        assert result[0]["source"] == filename
        assert expected in result[0]["content"]
        # The overview must survive the Agent's current 1,800-byte snippet budget.
        assert len(result[0]["content"].encode("utf-8")) <= 1800


def test_expanded_library_exposes_direct_answers_without_claiming_missing_features():
    documents = load_documents(Path(__file__).parents[1] / "data/documents")

    class EmptyVectorStore:
        def similarity_search_with_score(self, _question, k):
            return []

    for question, filename, expected in [
        ("RRF 排名融合是什么？", "retrieval_quality_guide.md", "没有实现 RRF"),
        ("测试覆盖率高是否意味着回答准确率高？", "retrieval_quality_guide.md", "不是"),
        ("当前项目可以直接多机器部署吗？", "project_operations_guide.md", "本地知识库单点"),
        ("已经显示答案但提示历史未确认保存怎么办？", "project_operations_guide.md", "unconfirmed"),
    ]:
        result = knowledge_search(question, documents, EmptyVectorStore())
        assert result[0]["source"] == filename
        assert expected in result[0]["content"]
        assert len(result[0]["content"].encode("utf-8")) <= 1800
