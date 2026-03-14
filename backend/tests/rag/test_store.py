"""Tests for vector store functionality.

This module contains unit tests for the ChromaDB vector store service,
including tests for document storage, retrieval, and persistence.
"""

import pytest
from unittest.mock import patch, Mock


def test_store_add_documents(store):
    """Test that store can add documents with embeddings and metadata."""
    chunks = ["Chunk 1 content", "Chunk 2 content"]
    embeddings = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
    metadata = [{"source": "doc1"}, {"source": "doc2"}]

    store.add_documents(chunks, embeddings, metadata)

    store.collection.add.assert_called_once()
    call_kwargs = store.collection.add.call_args[1]
    assert call_kwargs["documents"] == chunks
    assert call_kwargs["embeddings"] == embeddings
    assert call_kwargs["metadatas"] == metadata


def test_store_add_documents_without_metadata(store):
    """Test that store fills in None values when no metadata is provided."""
    chunks = ["Chunk 1", "Chunk 2"]
    embeddings = [[0.1, 0.2], [0.3, 0.4]]

    store.add_documents(chunks, embeddings)

    call_kwargs = store.collection.add.call_args[1]
    assert call_kwargs["metadatas"] == [None, None]


def test_store_add_empty_documents(store):
    """Test that adding an empty list is a no-op and never calls collection.add."""
    store.add_documents([], [], [])

    store.collection.add.assert_not_called()


def test_store_search_returns_formatted_results(store):
    """Test that search returns correctly shaped result dicts."""
    query_embedding = [0.1, 0.2, 0.3]

    results = store.search(query_embedding, top_k=3)

    assert isinstance(results, list)
    assert len(results) == 3
    for result in results:
        assert "chunk" in result
        assert "metadata" in result
        assert "score" in result
        assert "id" in result

    store.collection.query.assert_called_once_with(
        query_embeddings=[query_embedding],
        n_results=3
    )


def test_store_search_score_range(store):
    """Test that scores are in [0, 1] (cosine similarity from cosine distance)."""
    results = store.search([0.1, 0.2, 0.3], top_k=3)

    for result in results:
        assert 0.0 <= result["score"] <= 1.0


def test_store_search_top_k_zero_returns_empty(store):
    """Test that top_k=0 short-circuits and returns empty list."""
    results = store.search([0.1, 0.2, 0.3], top_k=0)

    assert results == []
    store.collection.query.assert_not_called()


def test_store_search_respects_top_k(store, mock_chroma_collection):
    """Test that search passes top_k correctly to ChromaDB."""
    store.search([0.1, 0.2, 0.3], top_k=1)
    store.search([0.1, 0.2, 0.3], top_k=5)

    calls = mock_chroma_collection.query.call_args_list
    assert calls[0][1]["n_results"] == 1
    assert calls[1][1]["n_results"] == 5


def test_store_count_documents(store):
    """Test that count_documents returns the value from the collection."""
    count = store.count_documents()

    assert count == 2
    store.collection.count.assert_called_once()


def test_store_delete_collection(store):
    """Test that delete_collection calls client and nulls out the collection ref."""
    store.delete_collection()

    store._client.delete_collection.assert_called_once_with(name=store.collection_name)
    assert store._collection is None


def test_store_error_handling_mismatched_inputs(store):
    """Test that mismatched chunks/embeddings raises ValueError."""
    with pytest.raises(ValueError):
        store.add_documents(
            chunks=["Chunk 1", "Chunk 2"],
            embeddings=[[0.1, 0.2]],  # One embedding for two chunks
        )


def test_store_error_handling_mismatched_metadata(store):
    """Test that mismatched chunks/metadata raises ValueError."""
    with pytest.raises(ValueError):
        store.add_documents(
            chunks=["Chunk 1", "Chunk 2"],
            embeddings=[[0.1], [0.2]],
            metadata=[{"source": "doc1"}],  # One metadata for two chunks
        )


def test_store_error_handling_invalid_query_embedding(store):
    """Test that an empty query embedding raises ValueError."""
    with pytest.raises(ValueError):
        store.search([], top_k=3)


def test_store_error_handling_none_inputs(store):
    """Test that None chunks or embeddings raise ValueError."""
    with pytest.raises(ValueError):
        store.add_documents(None, [[0.1]])

    with pytest.raises(ValueError):
        store.add_documents(["chunk"], None)


@pytest.mark.integration
def test_store_with_temp_directory(temp_app_data_dir):
    """Integration test: store initializes correctly with a real temp directory."""
    from app.services.rag.store import get_store

    store = get_store(persist_directory=str(temp_app_data_dir / "chroma"))

    assert store is not None
    assert store.count_documents() == 0