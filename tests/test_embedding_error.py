import pytest

from langchain_openai import OpenAIEmbeddings

from src.embeddings import create_embeddings
from src.exceptions import EmbeddingServiceError


def test_embedding_query_error(
    monkeypatch,
):
    embeddings = create_embeddings()

    def fake_embed_query(
        self,
        text,
    ):
        raise ConnectionError(
            "模拟 SiliconFlow 网络连接失败"
        )

    monkeypatch.setattr(
        OpenAIEmbeddings,
        "embed_query",
        fake_embed_query,
    )

    with pytest.raises(
        EmbeddingServiceError,
        match="Embedding 服务暂时不可用",
    ):
        embeddings.embed_query(
            "什么是 RAG？"
        )
