"""Tests for text chunking functionality.

This module contains unit tests for the text chunking service, including
tests for different chunk sizes, overlap settings, and edge cases.
"""

import pytest
from app.services.rag.chunker import get_chunker


def test_chunker_creates_reasonable_chunks(sample_text_content):
    """Test that chunker creates chunks with reasonable size and overlap.
    
    Verifies that the chunker splits text into chunks that respect the
    configured chunk size and overlap parameters.
    
    Args:
        sample_text_content: Fixture providing sample text for testing
        
    Asserts:
        - Result is a list of chunks
        - Each chunk is a string
        - Chunk sizes are within expected range
        - Overlap is working correctly
    """
    # Arrange
    chunker = get_chunker()
    
    # Act
    chunks = chunker.chunk_text(sample_text_content, chunk_size=500, chunk_overlap=100)
    
    # Assert
    assert isinstance(chunks, list)
    assert len(chunks) > 1  # Should create multiple chunks
    
    # Check that most chunks are within reasonable size range
    for chunk in chunks[:-1]:  # Skip last chunk which might be smaller
        assert len(chunk) <= 600  # Allow some flexibility for word boundaries
    
    # Verify overlap between adjacent chunks (except for very small chunks)
    for i in range(len(chunks) - 1):
        current_chunk = chunks[i]
        next_chunk = chunks[i+1]
        
        # Skip overlap check if either chunk is too small
        if len(current_chunk) < 50 or len(next_chunk) < 50:
            continue
            
        # Check for overlap using the smaller of: expected overlap or available text
        overlap_size = min(50, len(current_chunk), len(next_chunk))
        overlap_text = current_chunk[-overlap_size:]
        assert overlap_text in next_chunk, f"No overlap found between chunk {i} and {i+1}"


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


def test_chunker_handles_very_long_text():
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


def test_chunker_preserves_content_integrity(sample_text_from_file):
    """Test that chunker preserves content integrity.
    
    Verifies that when chunks are concatenated back together (with overlap handled),
    the original content is preserved.
    
    Args:
        sample_text_from_file: Fixture providing text from actual test file
    """
    # Arrange
    chunker = get_chunker()
    original_text = sample_text_from_file
    
    # Act
    chunks = chunker.chunk_text(original_text, chunk_size=500, chunk_overlap=100)
    
    # Assert - Simple integrity check
    # The first chunk should start with the original text
    assert original_text.startswith(chunks[0])
    
    # The last chunk should end with the original text
    assert original_text.endswith(chunks[-1])
    
    # All chunks combined should contain all major sections
    combined_text = " ".join(chunks)
    # Check that key phrases from original are present
    key_phrases = ["SolarSolutions", "SC-100", "waterproof", "warranty"]
    for phrase in key_phrases:
        assert phrase in combined_text, f"Key phrase '{phrase}' not found in combined chunks"
