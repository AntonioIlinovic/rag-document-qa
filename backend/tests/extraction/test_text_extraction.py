"""Tests for text document extraction functionality.

This module contains unit tests for the text extraction service, including
tests for the TextExtractor implementation and the factory function.
Tests use pytest fixtures to provide test data and follow AAA pattern
(Arrange, Act, Assert).
"""

import pytest
from app.services.extraction import get_extractor
from app.services.extraction.text import TextExtractor
from app.services.extraction.base import ExtractionError


def test_text_extractor_returns_non_empty_text(sample_txt_bytes):
    """Test that text extraction produces non-empty text with expected content.
    
    Verifies that the TextExtractor can successfully extract text from a TXT
    file and that the extracted text contains known content from the test document.
    
    Args:
        sample_txt_bytes: Fixture providing the TXT file bytes
        
    Asserts:
        - Result is a string
        - Result is not empty after stripping whitespace
        - Result contains expected text content from the TXT file
    """
    # Arrange
    extractor = get_extractor("solar_solutions.txt")
    
    # Act
    result = extractor.extract(sample_txt_bytes, "solar_solutions.txt")
    
    # Assert
    assert isinstance(result, str)
    assert len(result.strip()) > 0
    # Test for known content from the test document
    assert "SolarSolutions: Residential Photovoltaic Operations" in result
    assert "Congratulations on your transition to clean, renewable energy" in result


def test_text_extractor_handles_markdown(sample_md_bytes):
    """Test that text extraction works with markdown files.
    
    Verifies that the TextExtractor can successfully extract text from a MD
    file and preserves the markdown content as plain text.
    
    Args:
        sample_md_bytes: Fixture providing the MD file bytes
        
    Asserts:
        - Result is a string
        - Result is not empty after stripping whitespace
        - Result contains expected markdown content
    """
    # Arrange
    extractor = get_extractor("example.md")
    
    # Act
    result = extractor.extract(sample_md_bytes, "example.md")
    
    # Assert
    assert isinstance(result, str)
    assert len(result.strip()) > 0
    # Test for markdown content
    assert "# Sample Markdown" in result
    assert "**sample markdown**" in result


def test_text_extractor_handles_various_encodings():
    """Test that text extraction handles different text encodings.
    
    Verifies that the TextExtractor can handle UTF-8 encoded text and
    falls back to latin-1 when needed.
    
    Asserts:
        - UTF-8 text is decoded correctly
        - Latin-1 text fallback works when UTF-8 fails
    """
    # Arrange
    extractor = TextExtractor()
    
    # Test UTF-8
    utf8_text = "Hello world! 🌍 café naïve"
    utf8_bytes = utf8_text.encode('utf-8')
    
    # Act & Assert for UTF-8
    result_utf8 = extractor.extract(utf8_bytes, "test_utf8.txt")
    assert result_utf8 == utf8_text
    assert "🌍" in result_utf8
    
    # Test latin-1 fallback
    latin1_text = "Text with latin-1 chars: àâäåæçè"
    latin1_bytes = latin1_text.encode('latin-1')
    
    # Act & Assert for latin-1
    result_latin1 = extractor.extract(latin1_bytes, "test_latin1.txt")
    assert result_latin1 == latin1_text


def test_factory_function_txt():
    """Test that factory function returns correct extractor for TXT files.
    
    Verifies that the get_extractor factory function correctly identifies TXT
    files and returns a TextExtractor instance.
    
    Asserts:
        - Returned extractor is an instance of TextExtractor
    """
    # Arrange & Act
    extractor = get_extractor("test.txt")
    
    # Assert
    assert isinstance(extractor, TextExtractor)


def test_factory_function_md():
    """Test that factory function returns correct extractor for MD files.
    
    Verifies that the get_extractor factory function correctly identifies MD
    files and returns a TextExtractor instance.
    
    Asserts:
        - Returned extractor is an instance of TextExtractor
    """
    # Arrange & Act
    extractor = get_extractor("test.md")
    
    # Assert
    assert isinstance(extractor, TextExtractor)


def test_factory_function_case_insensitive():
    """Test that factory function handles case-insensitive extensions.
    
    Verifies that the get_extractor factory function correctly identifies
    text files with uppercase extensions.
    
    Asserts:
        - Returned extractor is an instance of TextExtractor for both cases
    """
    # Arrange & Act
    extractor_upper = get_extractor("test.TXT")
    extractor_mixed = get_extractor("test.Md")
    
    # Assert
    assert isinstance(extractor_upper, TextExtractor)
    assert isinstance(extractor_mixed, TextExtractor)


def test_text_extractor_empty_file():
    """Test that text extraction handles empty files correctly.
    
    Verifies that the TextExtractor can handle empty files and returns
    an empty string.
    
    Asserts:
        - Result is an empty string
    """
    # Arrange
    empty_bytes = b""
    extractor = TextExtractor()
    
    # Act
    result = extractor.extract(empty_bytes, "empty.txt")
    
    # Assert
    assert isinstance(result, str)
    assert result == ""
