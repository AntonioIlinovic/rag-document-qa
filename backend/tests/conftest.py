"""Pytest configuration and shared fixtures.

This module provides common test fixtures that can be used across multiple test files.
Fixtures are automatically discovered by pytest and provide test setup/teardown
functionality following dependency injection principles.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock


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


@pytest.fixture
def sample_text_content():
    """Sample text content for RAG testing.
    
    Returns:
        str: Sample text content for chunking and embedding tests
    """
    return """The SolarSolutions SC-100 is a high-performance, ultra-portable monocrystalline solar charger designed for outdoor enthusiasts, hikers, and emergency preparedness. Utilizing high-efficiency SunPower cells, the SC-100 provides reliable power for smartphones, tablets, and other USB-powered devices even in challenging lighting conditions.

Technical Specifications:
- Peak Power Output: 15 Watts
- Transformation Efficiency: 22% - 25%
- Output Ports: Dual USB-A (5V / 2.1A Max per port)
- Weight: 12.8 oz (360g)

Frequently Asked Questions:
How long does it take to charge a standard phone? Under ideal direct sunlight, the SC-100 can charge a typical 3000mAh smartphone from 0% to 100% in approximately 2 to 2.5 hours.

Is the SC-100 waterproof? The solar panels themselves and the outer fabric are IP65 water-resistant, meaning they can handle light rain and splashes. However, the USB output controller box is NOT waterproof.
"""


@pytest.fixture  
def rag_test_file():
    """Path to the single RAG test document.
    
    Returns:
        Path: Path to the example document in rag_test folder
    """
    return Path(__file__).parent.parent.parent / "dummy_docs" / "rag_test" / "example_document.txt"


@pytest.fixture
def sample_text_from_file(rag_test_file):
    """Text content from the test document.
    
    Args:
        rag_test_file: Fixture providing path to test document
        
    Returns:
        str: Content of the test document
    """
    with open(rag_test_file, "r", encoding="utf-8") as f:
        return f.read()


@pytest.fixture
def mock_embedding_model():
    """Mock embedding model for testing.
    
    Returns:
        Mock: Mock object that mimics embedding model behavior
    """
    mock = Mock()
    # Mock encode method to return realistic embeddings
    mock.encode.return_value = [[0.1, 0.2, 0.3, 0.4, 0.5] for _ in range(10)]
    return mock


@pytest.fixture
def temp_app_data_dir():
    """Temporary app_data directory for testing.
    
    Creates a temporary directory that mimics the app_data structure
    and cleans it up after the test completes.
    
    Yields:
        Path: Path to temporary app_data directory
    """
    temp_dir = Path(tempfile.mkdtemp(prefix="test_app_data_"))
    
    # Create subdirectories that the app expects
    (temp_dir / "chroma").mkdir(exist_ok=True)
    (temp_dir / "sessions").mkdir(exist_ok=True)
    
    yield temp_dir
    
    # Cleanup
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def mock_chroma_collection():
    """Mock ChromaDB collection for testing.
    
    Returns:
        Mock: Mock object that mimics ChromaDB collection behavior
    """
    mock = Mock()
    # Mock common ChromaDB methods
    mock.add.return_value = None
    mock.query.return_value = {
        "ids": [["doc1", "doc2", "doc3"]],
        "documents": [["Chunk 1 content", "Chunk 2 content", "Chunk 3 content"]],
        "metadatas": [[{"source": "doc1"}, {"source": "doc2"}, {"source": "doc3"}]],
        "distances": [[0.1, 0.2, 0.3]]
    }
    mock.get.return_value = {
        "ids": ["doc1", "doc2"],
        "documents": ["Content 1", "Content 2"],
        "metadatas": [{"source": "doc1"}, {"source": "doc2"}]
    }
    mock.count.return_value = 2
    return mock