from langchain_openai import OpenAIEmbeddings

from src.config import (
    SILICONFLOW_API_KEY,
    EMBEDDING_MODEL,
    EMBEDDING_BASE_URL,
)
from src.exceptions import EmbeddingServiceError
from src.logger import setup_logger

logger = setup_logger()
class SafeOpenAIEmbeddings(OpenAIEmbeddings):
    """
    对 Embedding API 调用增加统一异常处理。
    """

    def embed_query(self, text):
        try:
            return super().embed_query(text)

        except Exception as error:
            logger.error(
                "Embedding API 查询调用失败（%s）",
                type(error).__name__,
            )

            raise EmbeddingServiceError(
                "Embedding 服务暂时不可用"
            ) from error

    def embed_documents(self, texts, chunk_size=None):
        try:
            return super().embed_documents(
                texts,
                chunk_size=chunk_size,
            )

        except Exception as error:
            logger.error(
                "Embedding API 文档调用失败（%s）",
                type(error).__name__,
            )

            raise EmbeddingServiceError(
                "Embedding 服务暂时不可用"
            ) from error


def create_embeddings():
    """
    创建并返回带异常处理的 Embedding 模型。
    """

    embeddings = SafeOpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        base_url=EMBEDDING_BASE_URL,
        api_key=SILICONFLOW_API_KEY,
    )

    return embeddings
