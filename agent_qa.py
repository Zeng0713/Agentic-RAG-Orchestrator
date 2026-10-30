from src.config import validate_config
from src.document_loader import load_documents
from src.embeddings import create_embeddings
from src.vector_store import (
    create_vector_store,
    sync_documents,
)
from src.rag_chain import create_llm
from src.agent import run_agent

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
    "输入 exit 退出。"
)


while True:
    question = input(
        "\n请输入你的问题："
    ).strip()

    if question.lower() == "exit" or "退出":
        print("程序已退出，再见！")
        break

    if not question:
        print("问题不能为空。")
        continue

    answer = run_agent(
        question=question,
        llm=llm,
        documents=documents,
        vector_store=vector_store,
    )

    print("\nAgent 回答：")
    print(answer)