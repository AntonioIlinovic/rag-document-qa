"""Shared fixtures for session tests."""

from unittest.mock import patch
import pytest
from fastapi.testclient import TestClient
from app.main import create_app

@pytest.fixture
def client(tmp_path):
    """Create a test client with isolated app_data directory."""
    with patch("app.config.settings.app_data_dir", str(tmp_path / "app_data")):
        app = create_app()
        return TestClient(app)

@pytest.fixture
def mock_pdf_content():
    """Mock PDF file content."""
    return b"%PDF-1.4\n1 0 obj\n<<\n/Type /Catalog\n/Pages 2 0 R\n>>\nendobj\n"

@pytest.fixture
def mock_extracted_text():
    """Mock extracted text content."""
    return "This is a sample document text. It contains multiple sentences. The termination clause states that either party may terminate the agreement with 30 days notice."
