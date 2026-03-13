"""Tests for PDF document extraction functionality.

This module contains unit tests for the PDF extraction service, including
tests for the PyMuPDF extractor implementation and the factory function.
Tests use pytest fixtures to provide test data and follow AAA pattern
(Arrange, Act, Assert).
"""

import pytest
from app.services.extraction import get_extractor


def test_pdf_extractor_returns_non_empty_text(sample_pdf_bytes):
    """Test that PDF extraction produces non-empty text with expected content.
    
    Verifies that the PyMuPDFExtractor can successfully extract text from a PDF
    file and that the extracted text contains known content from the test document.
    
    Args:
        sample_pdf_bytes: Fixture providing the PDF file bytes
        
    Asserts:
        - Result is a string
        - Result is not empty after stripping whitespace
        - Result contains expected text content from the PDF
    """
    # Arrange
    extractor = get_extractor("Zagreb_ice_skating_info.pdf")
    
    # Act
    result = extractor.extract(sample_pdf_bytes, "Zagreb_ice_skating_info.pdf")
    
    # Assert
    assert isinstance(result, str)
    assert len(result.strip()) > 0
    # Test for known content from your PDF
    assert """Zagreb is the capital of Croatia, one of Europe's youngest countries""" in result
    assert """This high achieving army general abolished serfdom and conducted a number of successful military""" in result


def test_factory_function_pdf():
    """Test that factory function returns correct extractor for PDF files.
    
    Verifies that the get_extractor factory function correctly identifies PDF
    files and returns a PyMuPDFExtractor instance.
    
    Asserts:
        - Returned extractor is an instance of PyMuPDFExtractor
    """
    # Arrange & Act
    extractor = get_extractor("Zagreb_ice_skating_info.pdf")
    
    # Assert
    from app.services.extraction.pdf import PyMuPDFExtractor
    assert isinstance(extractor, PyMuPDFExtractor)


def test_factory_function_unsupported_type():
    """Test that factory function raises error for unsupported file types.
    
    Verifies that the get_extractor function properly validates file extensions
    and raises a ValueError for unsupported file types.
    
    Asserts:
        - ValueError is raised with appropriate error message
    """
    # Arrange & Act & Assert
    with pytest.raises(ValueError, match="Unsupported file type"):
        get_extractor("test.txt")