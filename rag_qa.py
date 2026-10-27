from dotenv import load_dotenv
from src.config import validate_config
from src.document_loader import load_documents
from src.embeddings import create_embeddings
from src.vector_store import (
    create_vector_store,
    sync_documents,
)
from src.retriever import retrieve_documents
from src.rag_chain import (
    create_llm,
    generate_answer,
    get_citations,
)
from src.exceptions import (
    LLMServiceError,
    EmbeddingServiceError,
)
# 0. 检查项目配置
validate_config()


# 1. 读取环境变量
# ==========================================
load_dotenv()

# ==========================================
# 2. 加载知识库
# ==========================================

chunks = load_documents()

print(
    f"知识库加载完成，共 {len(chunks)} 个 Chunk"
)


# ==========================================
# 3. 创建 Embedding 模型
# ==========================================

embeddings = create_embeddings()


# ==========================================
# 4. 创建 Chroma 向量数据库
# ==========================================

vector_store = create_vector_store(
    embeddings
)


# ==========================================
# 5. 同步知识库
# ==========================================

sync_documents(
    vector_store,
    chunks,
)


# ==========================================
# 6. 创建 DeepSeek 模型
# ==========================================

llm = create_llm()


# ==========================================
# 7. RAG 问答循环
# ==========================================

while True:

    question = input(
        "\n请输入你的问题（输入 exit 退出）："
    ).strip()

    # ---------- 退出程序 ----------

    if question.lower() == "exit":
        print("程序已退出，再见！")
        break

    # ---------- 防止空问题 ----------

    if not question:
        print("问题不能为空，请重新输入。")
        continue

    # ======================================
    # 8. 检索相关资料
    # ======================================
    try:
        retrieved_docs = retrieve_documents(
            question,
            chunks,
            vector_store,
        )

    except EmbeddingServiceError:
        print(
            "\n知识检索失败，"
            "Embedding 服务暂时不可用，"
            "请稍后重试。"
        )

        continue
    # ---------- 显示检索结果 ----------

    print("\n检索到的参考资料：")

    for index, doc in enumerate(
        retrieved_docs,
        start=1,
    ):
        print(
            f"\n参考资料 {index}："
        )

        print(doc.page_content)

        print(
            "来源：",
            doc.metadata.get(
                "source",
                "未知来源",
            ),
        )

    # ======================================
    # 9. 使用 DeepSeek 生成答案
    # ======================================

    try:
        answer = generate_answer(
            question,
            retrieved_docs,
            llm,
        )

    except LLMServiceError:
        print(
            "\n生成回答失败，"
            "大语言模型服务暂时不可用，"
            "请稍后重试。"
        )

        continue

    citations = get_citations(
        answer,
        retrieved_docs,
    )
    # ======================================
    # 11. 输出最终结果
    # ======================================

    print("\n" + "=" * 40)
    print("RAG 问答结果")
    print("=" * 40)

    print(answer)

    print("\n参考来源：")

    if citations:
        for citation in citations:
            print(citation)
    else:
        print("无")