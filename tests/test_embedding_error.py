import pytest

from langchain_openai import OpenAIEmbeddings

import src.embeddings as embeddings_module
from src.exceptions import EmbeddingServiceError


def test_embedding_query_error(monkeypatch):

    # 使用假的 API Key，不需要真实密钥
    monkeypatch.setattr(
        embeddings_module,
        "SILICONFLOW_API_KEY",
        "test-key",
    )

    # 模拟 Embedding API 网络连接失败
    def fake_embed_query(self, text):
        raise ConnectionError(
            "模拟 SiliconFlow 网络连接失败"
        )

    # 先模拟，再创建对象
    monkeypatch.setattr(
        OpenAIEmbeddings,
        "embed_query",
        fake_embed_query,
    )

    # 创建 Embedding 对象
    embeddings = embeddings_module.create_embeddings()

    # 验证异常是否被正确包装
    with pytest.raises(
        EmbeddingServiceError,
        match="Embedding 服务暂时不可用",
    ):
        embeddings.embed_query(
            "什么是 RAG？"
        )
