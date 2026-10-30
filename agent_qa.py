from src.config import validate_config
from src.document_loader import load_documents
from src.embeddings import create_embeddings
from src.vector_store import (
    create_vector_store,
    sync_documents,
)
from src.rag_chain import create_llm
from src.agent import run_agent
from src.exceptions import EmbeddingServiceError, LLMServiceError

validate_config()
documents = load_documents()
embeddings = create_embeddings()
vector_store = create_vector_store(
    embeddings
)
sync_documents(
    vector_store,
    documents,
)
llm = create_llm()

print(
    "Knowledge Base Agent 已启动，"
    "输入 exit 或 退出结束。"
)


while True:
    question = input(
        "\n请输入你的问题："
    ).strip()

    if question.lower() == "exit" or question == "退出":
        print("程序已退出，再见！")
        break

    if not question:
        print("问题不能为空。")
        continue

    if not documents:
        print("知识库为空，请先在 data/documents/ 中添加 Markdown 文档。")
        continue

    try:
        answer = run_agent(
            question=question,
            llm=llm,
            documents=documents,
            vector_store=vector_store,
        )
    except EmbeddingServiceError:
        print("知识检索失败，Embedding 服务暂时不可用，请稍后重试。")
        continue
    except LLMServiceError:
        print("回答生成失败，DeepSeek 服务暂时不可用，请稍后重试。")
        continue
    except Exception as error:
        print(f"本次问答失败：{error}")
        continue

    print("\nAgent 回答：")
    print(answer)
