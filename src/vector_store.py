from pathlib import Path
from src.logger import setup_logger
from langchain_chroma import Chroma

from src.config import (
    CHROMA_DB_PATH,
    CHROMA_COLLECTION_NAME,
)

logger = setup_logger()



def create_vector_store(
    embeddings,
    db_path=CHROMA_DB_PATH,
):
    """
    创建并返回 Chroma 向量数据库。
    """

    db_dir = Path(db_path)

    vector_store = Chroma(
        collection_name=CHROMA_COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(db_dir),
    )

    return vector_store
def sync_documents(vector_store, documents):
    """
    将当前文档同步到向量数据库。

    新文档会被添加，
    已修改的文档会被更新。
    """

    document_ids = []
    source_counters = {}

    for doc in documents:
        source_name = Path(
            doc.metadata["source"]
        ).stem

        index = source_counters.get(
            source_name,
            0,
        )

        document_ids.append(
            f"{source_name}_{index}"
        )

        source_counters[source_name] = index + 1

    existing_data = vector_store.get()

    existing_ids = existing_data["ids"]
    existing_documents = existing_data["documents"]

    existing_document_map = dict(
        zip(
            existing_ids,
            existing_documents,
        )
    )

    logger.info(
        "数据库当前记录数：%d",
        len(existing_ids),
    )

    new_documents = []
    new_ids = []

    updated_documents = []
    updated_ids = []

    for doc, doc_id in zip(
        documents,
        document_ids,
    ):
        if doc_id not in existing_document_map:
            new_documents.append(doc)
            new_ids.append(doc_id)

        elif (
            existing_document_map[doc_id]
            != doc.page_content
        ):
            updated_documents.append(doc)
            updated_ids.append(doc_id)

    if new_documents:
        vector_store.add_documents(
            documents=new_documents,
            ids=new_ids,
        )

        logger.info(
            "新增 %d 个知识片段",
            len(new_documents),
        )
    else:
        logger.info(
            "没有新增知识片段"
        )

    if updated_documents:
        vector_store.update_documents(
            ids=updated_ids,
            documents=updated_documents,
        )

        logger.info(
            "更新 %d 个知识片段",
            len(updated_documents),
        )
    else:
        logger.info(
        "没有需要更新的知识片段"
    )

    logger.info(
        "知识库向量已准备完成"
    )