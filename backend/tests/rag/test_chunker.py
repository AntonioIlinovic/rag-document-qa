"""Tests for text chunking functionality.

This module contains unit tests for the text chunking service, including
tests for different chunk sizes, overlap settings, and edge cases.
"""

import pytest
from app.services.rag.chunker import get_chunker


def test_chunker_deterministic_behavior():
    """Test chunker with deterministic input for predictable results.
    
    Uses controlled text without natural breaks to force character-level
    splitting, ensuring deterministic chunking and overlap behavior.
    """
    chunker = get_chunker()
    
    # Use predictable text with 10 numbers, repeated pattern
    numbers = "1234567890"  # 10 characters
    text = numbers * 5  # 50 characters total (10 * 5)
    chunks = chunker.chunk_text(text, chunk_size=15, chunk_overlap=5)
    
    # Predictable results: 50 chars with 15 + (15-5) + (15-5) + (15-5) + (10-5)
    assert len(chunks) == 5
    assert len(chunks[0]) == 15
    assert len(chunks[1]) == 15
    assert len(chunks[2]) == 15  
    assert len(chunks[3]) == 15
    assert len(chunks[4]) == 10
    
    # Verify deterministic overlap behavior
    assert chunks[1].startswith(chunks[0][-5:])  # 5 char overlap
    assert chunks[2].startswith(chunks[1][-5:])  # 5 char overlap
    assert chunks[3].startswith(chunks[2][-5:])  # 5 char overlap


def test_chunker_handles_empty_text():
    """Test that chunker handles empty text gracefully.
    
    Verifies that the chunker returns an empty list when given empty input.
    """
    # Arrange
    chunker = get_chunker()
    
    # Act
    chunks = chunker.chunk_text("", chunk_size=1000, chunk_overlap=200)
    
    # Assert
    assert chunks == []


def test_chunker_handles_short_text():
    """Test that chunker handles text shorter than chunk size.
    
    Verifies that the chunker returns a single chunk when text is shorter
    than the configured chunk size.
    """
    # Arrange
    chunker = get_chunker()
    short_text = "This is a short text."
    
    # Act
    chunks = chunker.chunk_text(short_text, chunk_size=1000, chunk_overlap=200)
    
    # Assert
    assert len(chunks) == 1
    assert chunks[0] == short_text


def test_chunker_handles_very_long_text(sample_text_content):
    """Test that chunker handles very long text efficiently.
    
    Verifies that the chunker can process long documents without issues.
    """
    # Arrange
    chunker = get_chunker()
    # Create a long text by repeating the sample content
    long_text = sample_text_content * 10
    
    # Act
    chunks = chunker.chunk_text(long_text, chunk_size=1000, chunk_overlap=200)
    
    # Assert
    assert len(chunks) > 5  # Should create multiple chunks
    assert all(len(chunk) > 0 for chunk in chunks)


def test_chunker_with_different_parameters(sample_text_content):
    """Test chunker with different chunk size and overlap parameters.
    
    Verifies that the chunker respects different configuration parameters.
    
    Args:
        sample_text_content: Fixture providing sample text for testing
    """
    # Arrange
    chunker = get_chunker()
    
    # Act - Test with small chunks
    small_chunks = chunker.chunk_text(sample_text_content, chunk_size=200, chunk_overlap=50)
    
    # Act - Test with large chunks
    large_chunks = chunker.chunk_text(sample_text_content, chunk_size=2000, chunk_overlap=400)
    
    # Assert
    assert len(small_chunks) > len(large_chunks)  # Smaller chunks should create more pieces
    
    # Verify chunk sizes are respected
    for chunk in small_chunks[:-1]:
        assert len(chunk) <= 250  # Allow some flexibility
    
    for chunk in large_chunks[:-1]:
        assert len(chunk) <= 2400  # Allow some flexibility


def test_chunker_preserves_content_integrity():
    """Test that chunker preserves content integrity.
    
    Uses deterministic input to verify that chunking preserves
    the original content structure and overlap behavior.
    """
    chunker = get_chunker()
    
    # Use the same deterministic pattern as the other test
    numbers = "1234567890"  # 10 characters
    original_text = numbers * 8  # 80 characters total
    
    chunks = chunker.chunk_text(original_text, chunk_size=25, chunk_overlap=5)
    
    # Verify the first chunk starts with original text
    assert original_text.startswith(chunks[0])
    
    # Verify the last chunk ends with original text
    assert original_text.endswith(chunks[-1])
    
    # Verify all chunks combined preserve the pattern
    combined_text = "".join(chunks)  # No spaces for this test
    # Check that the pattern appears throughout
    assert "1234567890" in combined_text
    
    # Verify specific overlap behavior
    for i in range(len(chunks) - 1):
        current_chunk = chunks[i]
        next_chunk = chunks[i+1]
        if len(current_chunk) >= 5 and len(next_chunk) >= 5:
            overlap_text = current_chunk[-5:]
            assert next_chunk.startswith(overlap_text)
