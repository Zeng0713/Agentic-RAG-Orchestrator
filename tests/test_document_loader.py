from src.document_loader import load_documents


def test_load_documents(tmp_path):
    # Arrange：准备测试数据

    test_file = tmp_path / "test.md"

    test_file.write_text(
        """
# 测试知识库

## 什么是 Agent

Agent 是能够根据任务采取行动的系统。

## 什么是 RAG

RAG 是检索增强生成。
""".strip(),
        encoding="utf-8",
    )

    # Act：执行被测试函数

    documents = load_documents(
        tmp_path
    )

    # Assert：验证结果

    assert len(documents) == 3

    assert (
        documents[0].page_content
        == "# 测试知识库"
    )

    assert (
        "## 什么是 Agent"
        in documents[1].page_content
    )

    assert (
        "Agent 是能够根据任务采取行动的系统。"
        in documents[1].page_content
    )

    assert (
        "## 什么是 RAG"
        in documents[2].page_content
    )

    assert (
        documents[1].metadata[
            "document_type"
        ]
        == "markdown"
    )

    assert (
        documents[1].metadata["source"]
        == str(test_file)
    )


def test_loads_project_knowledge_documents():
    documents = load_documents("data/documents")

    sources = {
        document.metadata["source"]
        for document in documents
    }

    assert any(
        source.endswith("rag_bulid_intro.md")
        for source in sources
    )
    assert any(
        source.endswith("agentic_rag_build_intro.md")
        for source in sources
    )
    assert any(
        "混合检索" in document.page_content
        for document in documents
    )
    assert any(
        "Agentic RAG 搭建流程总览" in document.page_content
        for document in documents
    )
