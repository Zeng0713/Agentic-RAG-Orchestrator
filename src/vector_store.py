from pathlib import Path
import re
from src.logger import setup_logger
from langchain_chroma import Chroma as LangChainChroma

from src.config import (
    CHROMA_DB_PATH,
    CHROMA_COLLECTION_NAME,
)

logger = setup_logger()


class Chroma(LangChainChroma):
    """Keep provider calls outside the knowledge publication lock.

    The pinned LangChain adapter exposes no pre-embedded add method. Its Chroma
    collection access is confined here and covered by a real local-store test.
    """

    def prepare_documents(self, documents):
        return self.embeddings.embed_documents([doc.page_content for doc in documents])

    def add_prepared_documents(self, documents, ids, embeddings):
        self._collection.upsert(
            ids=ids, embeddings=embeddings,
            documents=[doc.page_content for doc in documents],
            metadatas=[doc.metadata for doc in documents],
        )


def _is_builtin_chunk(record_id, metadata):
    """Recognize owned records, including IDs written before origin was added."""
    metadata = metadata or {}
    if metadata.get("origin") == "upload" or metadata.get("document_id"):
        return False
    if metadata.get("origin") == "builtin":
        return True
    if metadata.get("origin"):
        return False
    source = Path(str(metadata.get("source", "")).replace("\\", "/"))
    return source.suffix.lower() == ".md" and re.fullmatch(
        re.escape(source.stem) + r"_\d+", record_id
    ) is not None



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
    已修改的文档会被更新；成功写入后清理不再存在的内置片段。
    documents 必须是完整的内置文档快照，不得传入局部文档集合。
    上传文档及无法确认归属的记录不参与清理。
    """

    documents = [
        doc.model_copy(update={"metadata": {**doc.metadata, "origin": "builtin"}})
        for doc in documents
    ]
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
    existing_metadatas = existing_data.get("metadatas") or [None] * len(existing_ids)
    current_ids = set(document_ids)
    stale_ids = [
        record_id
        for record_id, metadata in zip(existing_ids, existing_metadatas)
        if record_id not in current_ids and _is_builtin_chunk(record_id, metadata)
    ]

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

    # Delete only after additions and updates succeed, keeping old data on failure.
    if stale_ids:
        vector_store.delete(ids=stale_ids)
        logger.info("清理 %d 个过期内置知识片段", len(stale_ids))

    logger.info("知识库向量已准备完成")
