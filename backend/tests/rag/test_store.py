"""Tests for vector store functionality.

This module contains unit tests for the ChromaDB vector store service,
including tests for document storage, retrieval, and persistence.
"""

import pytest
from unittest.mock import Mock, patch
from app.services.rag.store import get_store


def test_store_add_documents(mock_chroma_collection):
    """Test that store can add documents with embeddings.
    
    Verifies that the vector store can store documents along with their
    embeddings and metadata.
    
    Args:
        mock_chroma_collection: Fixture providing mock ChromaDB collection
    """
    # Arrange
    store = get_store()
    store.collection = mock_chroma_collection  # Inject mock
    chunks = ["Chunk 1 content", "Chunk 2 content"]
    embeddings = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
    metadata = [{"source": "doc1"}, {"source": "doc2"}]
    
    # Act
    store.add_documents(chunks, embeddings, metadata)
    
    # Assert
    mock_chroma_collection.add.assert_called_once()
    call_args = mock_chroma_collection.add.call_args
    assert call_args[1]["documents"] == chunks
    assert call_args[1]["embeddings"] == embeddings
    assert call_args[1]["metadatas"] == metadata


def test_store_search_documents(mock_chroma_collection):
    """Test that store can search for similar documents.
    
    Verifies that the vector store can retrieve documents based on
    query embeddings.
    
    Args:
        mock_chroma_collection: Fixture providing mock ChromaDB collection
    """
    # Arrange
    store = get_store()
    store.collection = mock_chroma_collection  # Inject mock
    query_embedding = [0.1, 0.2, 0.3]
    top_k = 3
    
    # Act
    results = store.search(query_embedding, top_k)
    
    # Assert
    assert isinstance(results, list)
    assert len(results) == 3  # Should return top_k results
    
    # Verify the mock was called correctly
    mock_chroma_collection.query.assert_called_once_with(
        query_embeddings=[query_embedding],
        n_results=top_k
    )


def test_store_search_with_different_top_k(mock_chroma_collection):
    """Test that store respects different top_k parameters.
    
    Verifies that the search function returns the correct number of results.
    
    Args:
        mock_chroma_collection: Fixture providing mock ChromaDB collection
    """
    # Arrange
    store = get_store()
    store.collection = mock_chroma_collection  # Inject mock
    query_embedding = [0.1, 0.2, 0.3]
    
    # Act
    results_1 = store.search(query_embedding, top_k=1)
    results_5 = store.search(query_embedding, top_k=5)
    
    # Assert
    assert len(results_1) == 1
    assert len(results_5) == 5
    
    # Verify mock was called with correct parameters
    assert mock_chroma_collection.query.call_count == 2


def test_store_delete_collection(mock_chroma_collection):
    """Test that store can delete a collection.
    
    Verifies that the vector store can clean up collections.
    
    Args:
        mock_chroma_collection: Fixture providing mock ChromaDB collection
    """
    # Arrange
    store = get_store()
    store.collection = mock_chroma_collection  # Inject mock
    
    # Act
    store.delete_collection()
    
    # Assert
    mock_chroma_collection.delete.assert_called_once()


def test_store_get_collection_count(mock_chroma_collection):
    """Test that store can count documents in collection.
    
    Verifies that the vector store can report the number of stored documents.
    
    Args:
        mock_chroma_collection: Fixture providing mock ChromaDB collection
    """
    # Arrange
    store = get_store()
    store.collection = mock_chroma_collection  # Inject mock
    
    # Act
    count = store.count_documents()
    
    # Assert
    assert count == 2  # Based on mock setup
    mock_chroma_collection.count.assert_called_once()


def test_store_persistence(mock_chroma_collection):
    """Test that store handles persistence operations.
    
    Verifies that the vector store can save and load collections.
    
    Args:
        mock_chroma_collection: Fixture providing mock ChromaDB collection
    """
    # Arrange
    store = get_store()
    store.collection = mock_chroma_collection  # Inject mock
    
    # Act
    store.persist()
    store.load()
    
    # Assert
    # ChromaDB handles persistence automatically, but we can verify
    # the methods exist and don't raise errors
    assert True  # If we get here, no exceptions were raised


def test_store_with_empty_inputs(mock_chroma_collection):
    """Test that store handles empty inputs gracefully.
    
    Verifies that the vector store can handle edge cases like empty
    chunks or embeddings.
    
    Args:
        mock_chroma_collection: Fixture providing mock ChromaDB collection
    """
    # Arrange
    store = get_store()
    store.collection = mock_chroma_collection  # Inject mock
    
    # Act & Assert - Empty documents
    store.add_documents([], [], [])
    mock_chroma_collection.add.assert_called_with(
        documents=[],
        embeddings=[],
        metadatas=[]
    )
    
    # Act & Assert - Empty search
    results = store.search([0.1, 0.2, 0.3], top_k=0)
    assert results == []


def test_store_error_handling():
    """Test that store handles errors gracefully.
    
    Verifies that the vector store provides meaningful error messages
    when encountering invalid inputs or connection issues.
    """
    # Arrange
    store = get_store()
    
    # Test with mismatched inputs
    with pytest.raises((ValueError, IndexError)):
        store.add_documents(
            chunks=["Chunk 1", "Chunk 2"],
            embeddings=[[0.1, 0.2]],  # Only one embedding for two chunks
            metadata=[{"source": "doc1"}]  # Only one metadata for two chunks
        )
    
    # Test with invalid query embedding
    with pytest.raises((ValueError, TypeError)):
        store.search([], top_k=3)  # Empty embedding


@patch('chromadb.Client')
def test_store_initialization(mock_chroma_client):
    """Test that store initializes correctly.
    
    Verifies that the vector store can be initialized with the correct
    configuration and creates collections as needed.
    
    Args:
        mock_chroma_client: Mock ChromaDB client
    """
    # Arrange
    mock_collection = Mock()
    mock_client_instance = Mock()
    mock_client_instance.get_or_create_collection.return_value = mock_collection
    mock_chroma_client.return_value = mock_client_instance
    
    # Act
    store = get_store()
    
    # Assert
    assert store.collection is not None
    mock_chroma_client.assert_called_once()


def test_store_with_temp_directory(temp_app_data_dir):
    """Test that store works with temporary directory.
    
    Verifies that the vector store can operate with a temporary
    app_data directory for testing.
    
    Args:
        temp_app_data_dir: Fixture providing temporary app_data directory
    """
    # Arrange
    store = get_store(persist_directory=str(temp_app_data_dir / "chroma"))
    
    # Assert
    assert store is not None
    # Store should be able to initialize without errors
