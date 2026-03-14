"""Tests for embedding functionality.

This module contains unit tests for the embedding service, including
tests for both mock and real embedding models.
"""

import pytest
from app.services.rag.embedder import get_embedder


def test_mock_embedder_generates_embeddings(embedder):
    """Test that mock embedder generates consistent embeddings."""
    embeddings = embedder.embed_texts(["This is a test text for embedding."])

    assert isinstance(embeddings, list)
    assert len(embeddings) == 1
    assert isinstance(embeddings[0], list)
    assert all(isinstance(val, (int, float)) for val in embeddings[0])


def test_embedder_handles_batch_texts(embedder):
    """Test that embedder handles multiple texts efficiently."""
    texts = ["First test text", "Second test text", "Third test text"]

    embeddings = embedder.embed_texts(texts)

    assert isinstance(embeddings, list)
    assert len(embeddings) == len(texts)
    assert all(isinstance(emb, list) for emb in embeddings)


def test_embedder_handles_empty_text(embedder):
    """Test that embedder handles empty text gracefully."""
    embeddings = embedder.embed_texts([""])

    assert isinstance(embeddings, list)
    assert len(embeddings) == 1


def test_embedder_handles_empty_list(embedder):
    """Test that embedder returns empty list for empty input."""
    embeddings = embedder.embed_texts([])

    assert isinstance(embeddings, list)
    assert len(embeddings) == 0


def test_embedder_embedding_consistency(embedder):
    """Test that embedder produces consistent embeddings for same input."""
    test_text = "Consistency test text"

    embeddings1 = embedder.embed_texts([test_text])
    embeddings2 = embedder.embed_texts([test_text])

    assert embeddings1 == embeddings2


def test_embedder_embedding_dimensions(embedder):
    """Test that all embeddings have the same dimensionality."""
    texts = ["Text 1", "Text 2", "Text 3"]

    embeddings = embedder.embed_texts(texts)

    first_dim = len(embeddings[0])
    assert all(len(emb) == first_dim for emb in embeddings)


def test_embedder_error_handling(embedder):
    """Test that embedder raises ValueError for invalid inputs."""
    with pytest.raises(ValueError):
        embedder.embed_texts(None)

    with pytest.raises(ValueError):
        embedder.embed_texts([None])

    with pytest.raises(ValueError):
        embedder.embed_texts([123])

    with pytest.raises(ValueError):
        embedder.embed_texts("not a list")


@pytest.mark.integration
def test_real_embedder_integration():
    """Integration test using real embedding model.

    Requires the actual sentence-transformers model to be downloaded.
    Run sparingly — this test is slow.
    """
    embedder = get_embedder()

    embeddings = embedder.embed_texts(["This is a test for the real embedding model."])

    assert isinstance(embeddings, list)
    assert len(embeddings) == 1
    assert len(embeddings[0]) == 384
    assert all(isinstance(val, float) for val in embeddings[0])


@pytest.mark.integration
def test_real_embedder_batch_processing():
    """Integration test for batch processing with real model."""
    embedder = get_embedder()
    texts = [
        "First sentence for embedding test.",
        "Second sentence for embedding test.",
        "This is a unrelated sentence. It should have least semantic similarity with the other sentences.",
        "The quick brown fox jumps over the lazy dog.",
        "A fast brown fox leaps above the sleeping dog."
    ]

    embeddings = embedder.embed_texts(texts)

    assert len(embeddings) == len(texts)
    assert all(len(emb) == 384 for emb in embeddings)
    assert all(isinstance(val, float) for emb in embeddings for val in emb)
    
    # Check similarity across different text relationships
    import math
    
    def cosine_similarity(emb1, emb2):
        return sum(a*b for a,b in zip(emb1, emb2)) / (math.sqrt(sum(a*a for a in emb1)) * math.sqrt(sum(b*b for b in emb2)))
    
    # Test similar sentences (indices 0-1)
    sim_0_1 = cosine_similarity(embeddings[0], embeddings[1])
    
    # Test fox sentences (indices 3-4) - should be very similar
    sim_3_4 = cosine_similarity(embeddings[3], embeddings[4])
    
    # Test cross-domain similarities (should be lower)
    sim_0_3 = cosine_similarity(embeddings[0], embeddings[3])  # test vs fox
    
    print(f"Similar pairs:")
    print(f"  Test sentences (0-1): {sim_0_1:.3f}")
    print(f"  Fox sentences (3-4): {sim_3_4:.3f}")
    print(f"Cross-domain pairs:")
    print(f"  Test vs Fox (0-3): {sim_0_3:.3f}")
    
    # Assertions for semantic relationships
    assert sim_0_1 > 0.7, "Similar test sentences should have high similarity"
    assert sim_3_4 > 0.7, "Similar fox sentences should have high similarity"
    
    # Cross-domain should be lower than within-domain
    assert sim_0_1 > sim_0_3, "Similar test sentences should be more similar than cross-domain"