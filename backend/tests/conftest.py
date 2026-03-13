"""Pytest configuration and shared fixtures.

This module provides common test fixtures that can be used across multiple test files.
Fixtures are automatically discovered by pytest and provide test setup/teardown
functionality following dependency injection principles.
"""

import pytest
from pathlib import Path


@pytest.fixture
def sample_pdf_path():
    """Path to a sample PDF file for testing.
    
    Returns:
        Path: The absolute path to the test PDF file located in dummy_docs
        
    Note:
        This fixture provides the file path, not the file contents. Use
        sample_pdf_bytes fixture if you need the actual file bytes.
    """
    return Path(__file__).parent.parent.parent / "dummy_docs" / "Zagreb_ice_skating_info.pdf"


@pytest.fixture
def sample_pdf_bytes(sample_pdf_path):
    """PDF file bytes for testing.
    
    This fixture depends on sample_pdf_path and reads the file into memory,
    providing the raw bytes needed for testing document extraction.
    
    Args:
        sample_pdf_path: The path fixture providing the PDF file location
        
    Returns:
        bytes: The complete contents of the PDF file as bytes
        
    Raises:
        FileNotFoundError: If the PDF file doesn't exist at the expected path
        IOError: If there are issues reading the file
    """
    with open(sample_pdf_path, "rb") as f:
        return f.read()


@pytest.fixture
def sample_image_path():
    """Path to a sample image file for testing.
    
    Returns:
        Path: The absolute path to the test image file located in dummy_docs
        
    Note:
        This fixture provides the file path, not the file contents. Use
        sample_image_bytes fixture if you need the actual file bytes.
    """
    return Path(__file__).parent.parent.parent / "dummy_docs" / "ice_skating_1.jpg"


@pytest.fixture
def sample_image_bytes(sample_image_path):
    """Image file bytes for testing.
    
    This fixture depends on sample_image_path and reads the file into memory,
    providing the raw bytes needed for testing OCR document extraction.
    
    Args:
        sample_image_path: The path fixture providing the image file location
        
    Returns:
        bytes: The complete contents of the image file as bytes
        
    Raises:
        FileNotFoundError: If the image file doesn't exist at the expected path
        IOError: If there are issues reading the file
    """
    with open(sample_image_path, "rb") as f:
        return f.read()