"""Tests for OCR document extraction functionality.

This module contains unit tests for the OCR extraction service, including
tests for the EasyOCR extractor implementation and the factory function.
Tests use pytest fixtures to provide test data and follow AAA pattern
(Arrange, Act, Assert).
"""

import pytest
from app.services.extraction import get_extractor


def test_ocr_extractor_returns_non_empty_text(sample_image_bytes):
    """Test that OCR extraction produces non-empty text with expected content.
    
    Verifies that the EasyOCRExtractor can successfully extract text from an image
    file and that the extracted text contains known content from the test document.
    
    Args:
        sample_image_bytes: Fixture providing the image file bytes
        
    Asserts:
        - Result is a string
        - Result is not empty after stripping whitespace
        - Result contains expected text content from the image
    """
    # Arrange
    extractor = get_extractor("ice_skating_1.jpg")
    
    # Act
    result = extractor.extract(sample_image_bytes, "ice_skating_1.jpg")
    
    # Assert
    assert isinstance(result, str)
    assert len(result.strip()) > 0
    # Test for known content from the image
    assert "Ice skating is the self-propulsion" in result
    assert "Synchronized skating is a unique artistic team sport" in result


def test_factory_function_image():
    """Test that factory function returns correct extractor for image files.
    
    Verifies that the get_extractor factory function correctly identifies image
    files and returns an EasyOCRExtractor instance.
    
    Asserts:
        - Returned extractor is an instance of EasyOCRExtractor
    """
    # Arrange & Act
    extractor = get_extractor("ice_skating_1.jpg")
    
    # Assert
    from app.services.extraction.ocr import EasyOCRExtractor
    assert isinstance(extractor, EasyOCRExtractor)


def test_factory_function_multiple_image_formats():
    """Test that factory function handles different image file extensions.
    
    Verifies that the get_extractor function correctly identifies various image
    file formats and returns EasyOCRExtractor instances for each.
    
    Asserts:
        - All supported image formats return EasyOCRExtractor instances
    """
    from app.services.extraction.ocr import EasyOCRExtractor
    
    # Test PNG
    extractor_png = get_extractor("test.png")
    assert isinstance(extractor_png, EasyOCRExtractor)
    
    # Test JPG
    extractor_jpg = get_extractor("test.jpg")
    assert isinstance(extractor_jpg, EasyOCRExtractor)
    
    # Test JPEG
    extractor_jpeg = get_extractor("test.jpeg")
    assert isinstance(extractor_jpeg, EasyOCRExtractor)
    
    # Test TIFF
    extractor_tiff = get_extractor("test.tiff")
    assert isinstance(extractor_tiff, EasyOCRExtractor)
    
    # Test TIF
    extractor_tif = get_extractor("test.tif")
    assert isinstance(extractor_tif, EasyOCRExtractor)


def test_factory_function_case_insensitive():
    """Test that factory function handles case-insensitive file extensions.
    
    Verifies that the get_extractor function correctly identifies image files
    regardless of the case of the file extension.
    
    Asserts:
        - Uppercase and mixed case extensions are handled correctly
    """
    from app.services.extraction.ocr import EasyOCRExtractor
    
    # Test uppercase
    extractor_upper = get_extractor("test.PNG")
    assert isinstance(extractor_upper, EasyOCRExtractor)
    
    # Test mixed case
    extractor_mixed = get_extractor("test.JpG")
    assert isinstance(extractor_mixed, EasyOCRExtractor)


def test_factory_function_unsupported_image_type():
    """Test that factory function raises error for unsupported image file types.
    
    Verifies that the get_extractor function properly validates file extensions
    and raises a ValueError for unsupported image file types.
    
    Asserts:
        - ValueError is raised with appropriate error message
    """
    # Arrange & Act & Assert
    with pytest.raises(ValueError, match="Unsupported file type"):
        get_extractor("test.bmp")