import pytest
from src.embeddings import create_embeddings


@pytest.mark.integration
def test_real_embedding_api():
    embeddings = create_embeddings()

    vector = embeddings.embed_query(
        "什么是 RAG？"
    )

    assert isinstance(vector, list)

    assert len(vector) == 1024

    assert all(
        isinstance(value, float)
        for value in vector
    )