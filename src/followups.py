"""Local, knowledge-backed follow-up suggestions; no extra model requests."""

from pathlib import Path
import re


TOPICS = [
    ("为什么知识库有内容却检索不到？", "retrieval_quality_guide.md", ("检索", "找不到", "命中", "知识库")),
    ("RRF 排名融合是什么？", "retrieval_quality_guide.md", ("混合", "融合", "排名", "检索")),
    ("Reranker 重排序是什么？", "retrieval_quality_guide.md", ("重排序", "质量", "排序", "检索")),
    ("测试覆盖率高是否意味着回答准确率高？", "retrieval_quality_guide.md", ("测试", "覆盖率", "准确率", "评估")),
    ("服务重启后哪些数据会保留？", "project_operations_guide.md", ("持久化", "重启", "redis", "历史")),
    ("为什么健康检查成功却无法回答？", "project_operations_guide.md", ("故障", "健康", "启动", "失败")),
    ("已经显示答案但提示历史未确认保存怎么办？", "project_operations_guide.md", ("历史", "保存", "redis", "故障")),
    ("如何验证文档上传已经入库？", "rag_build_intro.md", ("上传", "入库", "文档", "流程")),
    ("文档切分过大或过小会有什么影响？", "rag_build_intro.md", ("切分", "片段", "chunk", "搭建")),
    ("文档更新后为什么仍然检索到旧内容？", "rag_build_intro.md", ("文档", "更新", "索引", "入库")),
    ("混合检索为什么要结合标题匹配和向量检索？", "rag_build_intro.md", ("检索", "混合", "向量", "embedding")),
    ("如何评估 RAG 的检索结果和回答质量？", "rag_build_intro.md", ("rag", "评估", "检索", "回答")),
    ("Agentic RAG 与传统 RAG 有什么区别？", "agentic_rag_build_intro.md", ("agent", "传统", "区别", "工具")),
    ("Tool Schema、Registry 和 Executor 分别负责什么？", "agentic_rag_build_intro.md", ("工具", "schema", "registry", "executor")),
    ("六轮工具调用与六轮会话有什么区别？", "agentic_rag_build_intro.md", ("六轮", "多轮", "循环", "工具", "会话")),
    ("什么情况下应该选择传统 RAG 而不是 Agent？", "agentic_rag_build_intro.md", ("agent", "传统", "选择", "区别")),
    ("如何从传统 RAG 升级到 Agentic RAG？", "agentic_rag_build_intro.md", ("agent", "搭建", "流程", "升级")),
]


def normalize(question):
    return re.sub(r"[\W_]+", "", question.casefold())


def suggest_followups(question, answer, chunks, documents, asked=(), limit=3):
    """Rank maintained topics, then headings from the actual retrieved chunks.

    These are navigation hints, not claims that a model verified the answers.
    Only advertise maintained topics whose source exists in the current library.
    """
    if limit <= 0:
        return []
    available = {item["filename"] for item in documents if item.get("filename")}
    used_sources = {Path(str(item.get("source", "")).replace("\\", "/")).name for item in chunks}
    available |= used_sources
    excluded = {normalize(item) for item in [question, *asked]}
    query, context = question.casefold(), answer.casefold()
    candidates = []
    for text, source, keywords in TOPICS:
        if source not in available or normalize(text) in excluded:
            continue
        score = sum(4 for keyword in keywords if keyword in query)
        score += sum(1 for keyword in keywords if keyword in context)
        if score:
            candidates.append((score + (2 if source in used_sources else 0), text))
    for chunk in chunks:
        for heading in re.findall(r"(?m)^##\s+(.+)$", chunk.get("content", "")):
            heading = heading.strip()
            if not heading or len(heading) > 80 or normalize(heading) in normalize(question):
                continue
            text = heading.rstrip("？?") + "？" if heading.startswith(("如何", "什么", "为什么")) else f"请解释「{heading}」。"
            if normalize(text) not in excluded:
                candidates.append((3, text))
    result = []
    for _score, text in sorted(candidates, key=lambda item: -item[0]):
        if normalize(text) not in {normalize(item) for item in result}:
            result.append(text)
        if len(result) >= limit:
            break
    return result
