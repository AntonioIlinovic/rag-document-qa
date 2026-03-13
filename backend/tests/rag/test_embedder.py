"""Tests for embedding functionality.

This module contains unit tests for the embedding service, including
tests for both mock and real embedding models.
"""

import pytest
from app.services.rag.embedder import get_embedder


def test_mock_embedder_generates_embeddings(mock_embedding_model):
    """Test that mock embedder generates consistent embeddings.
    
    Verifies that the mock embedding model returns embeddings in the
    expected format and that embeddings are consistent for the same input.
    
    Args:
        mock_embedding_model: Fixture providing mock embedding model
    """
    # Arrange
    embedder = get_embedder()
    embedder.model = mock_embedding_model  # Inject mock for testing
    test_text = "This is a test text for embedding."
    
    # Act
    embeddings = embedder.embed_text(test_text)
    
    # Assert
    assert isinstance(embeddings, list)
    assert len(embeddings) > 0
    assert all(isinstance(emb, list) for emb in embeddings)
    assert all(isinstance(val, (int, float)) for emb in embeddings for val in emb)


def test_embedder_handles_batch_texts(mock_embedding_model):
    """Test that embedder handles multiple texts efficiently.
    
    Verifies that the embedder can process multiple texts in a single call.
    
    Args:
        mock_embedding_model: Fixture providing mock embedding model
    """
    # Arrange
    embedder = get_embedder()
    embedder.model = mock_embedding_model  # Inject mock for testing
    texts = [
        "First test text",
        "Second test text", 
        "Third test text"
    ]
    
    # Act
    embeddings = embedder.embed_texts(texts)
    
    # Assert
    assert isinstance(embeddings, list)
    assert len(embeddings) == len(texts)
    assert all(isinstance(emb, list) for emb in embeddings)


def test_embedder_handles_empty_text(mock_embedding_model):
    """Test that embedder handles empty text gracefully.
    
    Verifies that the embedder returns appropriate results for empty input.
    
    Args:
        mock_embedding_model: Fixture providing mock embedding model
    """
    # Arrange
    embedder = get_embedder()
    embedder.model = mock_embedding_model  # Inject mock for testing
    
    # Act
    embeddings = embedder.embed_text("")
    
    # Assert
    assert isinstance(embeddings, list)
    # Mock should still return something even for empty text


def test_embedder_handles_empty_list(mock_embedding_model):
    """Test that embedder handles empty list of texts gracefully.
    
    Verifies that the embedder returns appropriate results for empty list input.
    
    Args:
        mock_embedding_model: Fixture providing mock embedding model
    """
    # Arrange
    embedder = get_embedder()
    embedder.model = mock_embedding_model  # Inject mock for testing
    
    # Act
    embeddings = embedder.embed_texts([])
    
    # Assert
    assert isinstance(embeddings, list)
    assert len(embeddings) == 0


def test_embedder_embedding_consistency(mock_embedding_model):
    """Test that embedder produces consistent embeddings for same input.
    
    Verifies that the same text always produces the same embedding.
    
    Args:
        mock_embedding_model: Fixture providing mock embedding model
    """
    # Arrange
    embedder = get_embedder()
    embedder.model = mock_embedding_model  # Inject mock for testing
    test_text = "Consistency test text"
    
    # Act
    embeddings1 = embedder.embed_text(test_text)
    embeddings2 = embedder.embed_text(test_text)
    
    # Assert
    assert embeddings1 == embeddings2


def test_embedder_embedding_dimensions(mock_embedding_model):
    """Test that embeddings have consistent dimensions.
    
    Verifies that all embeddings have the same dimensionality.
    
    Args:
        mock_embedding_model: Fixture providing mock embedding model
    """
    # Arrange
    embedder = get_embedder()
    embedder.model = mock_embedding_model  # Inject mock for testing
    texts = ["Text 1", "Text 2", "Text 3"]
    
    # Act
    embeddings = embedder.embed_texts(texts)
    
    # Assert
    if len(embeddings) > 1:
        # All embeddings should have the same dimension
        first_dim = len(embeddings[0])
        assert all(len(emb) == first_dim for emb in embeddings)


@pytest.mark.integration  # Mark as integration test (uses real model)
def test_real_embedder_integration():
    """Integration test using real embedding model.
    
    This test uses the actual sentence-transformers model to verify
    that the real embedder works correctly. This test is slower and
    should be run sparingly.
    
    Note:
        This test requires the actual embedding model to be downloaded.
        It may take several seconds to run.
    """
    # Arrange
    embedder = get_embedder()  # Use real embedder
    test_text = "This is a test for the real embedding model."
    
    # Act
    embeddings = embedder.embed_text(test_text)
    
    # Assert
    assert isinstance(embeddings, list)
    assert len(embeddings) > 0
    assert len(embeddings[0]) > 0  # Should have actual embedding dimensions
    assert all(isinstance(val, float) for val in embeddings[0])
    
    # Typical sentence transformer embeddings have 384 dimensions for MiniLM
    assert len(embeddings[0]) == 384


@pytest.mark.integration
def test_real_embedder_batch_processing():
    """Integration test for batch processing with real model.
    
    Verifies that the real embedder can handle multiple texts efficiently.
    """
    # Arrange
    embedder = get_embedder()  # Use real embedder
    texts = [
        "First sentence for embedding test.",
        "Second sentence for embedding test.",
        "Third sentence for embedding test."
    ]
    
    # Act
    embeddings = embedder.embed_texts(texts)
    
    # Assert
    assert isinstance(embeddings, list)
    assert len(embeddings) == len(texts)
    assert all(len(emb) == 384 for emb in embeddings)  # MiniLM dimensions
    assert all(isinstance(val, float) for emb in embeddings for val in emb)


def test_embedder_error_handling():
    """Test that embedder handles errors gracefully.
    
    Verifies that the embedder provides meaningful error messages
    when encountering invalid inputs or model failures.
    """
    # Arrange
    embedder = get_embedder()
    
    # Test with None input - should handle gracefully
    with pytest.raises((ValueError, TypeError, AttributeError)):
        embedder.embed_text(None)
    
    # Test with invalid type
    with pytest.raises((ValueError, TypeError, AttributeError)):
        embedder.embed_text(123)  # Should not accept numbers
