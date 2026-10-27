import pytest

from src.document_loader import load_documents
from src.embeddings import create_embeddings
from src.vector_store import (
    create_vector_store,
    sync_documents,
)


@pytest.mark.integration
def test_real_vector_store(tmp_path):
    # Arrange
    documents = load_documents()

    embeddings = create_embeddings()

    test_db_path = (
        tmp_path / "chroma_test"
    )

    vector_store = create_vector_store(
        embeddings,
        db_path=test_db_path,
    )

    # Act
    sync_documents(
        vector_store,
        documents,
    )

    data = vector_store.get()

    # Assert
    assert len(data["ids"]) == len(documents)

    assert len(data["documents"]) == len(
        documents
    )