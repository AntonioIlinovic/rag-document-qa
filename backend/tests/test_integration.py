"""Simple integration tests for the working RAG Document QA API.

These tests verify the end-to-end functionality without complex mocking.
"""

import pytest
from fastapi.testclient import TestClient
from pathlib import Path
from unittest.mock import patch, Mock

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
    return "Zagreb is the capital of Croatia. It has many tourist attractions including the Stone Gate and Mirogoj cemetery."


class TestBasicIntegration:
    """Test basic integration scenarios."""

    @patch("app.api.upload.get_extractor")
    def test_upload_and_ask_workflow(self, mock_get_extractor, client, mock_pdf_content, mock_extracted_text):
        """Test the complete upload and ask workflow."""
        # Mock extractor
        mock_extractor = Mock()
        mock_extractor.extract.return_value = mock_extracted_text
        mock_get_extractor.return_value = mock_extractor
        
        # Step 1: Upload a document
        files = {"files": ("test.pdf", mock_pdf_content, "application/pdf")}
        upload_response = client.post("/upload/", files=files)
        
        assert upload_response.status_code == 200
        upload_data = upload_response.json()
        assert "session_id" in upload_data
        assert len(upload_data["documents"]) == 1
        assert upload_data["documents"][0]["filename"] == "test.pdf"
        assert upload_data["documents"][0]["status"] == "processed"
        assert upload_data["documents"][0]["chunks"] > 0
        
        session_id = upload_data["session_id"]
        
        # Step 2: Ask a question about the uploaded document
        ask_request = {"session_id": session_id, "question": "What is Zagreb?"}
        ask_response = client.post("/ask/", json=ask_request)
        
        assert ask_response.status_code == 200
        ask_data = ask_response.json()
        assert "answer" in ask_data
        assert "sources" in ask_data
        assert len(ask_data["sources"]) > 0
        
        # Verify sources have the expected structure
        for source in ask_data["sources"]:
            assert "chunk" in source
            assert "score" in source
            assert isinstance(source["score"], float)
            assert 0 <= source["score"] <= 1

    def test_health_endpoint(self, client):
        """Test the health check endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}

    @patch("app.api.upload.get_extractor")
    def test_upload_unsupported_file_type(self, mock_get_extractor, client):
        """Test upload with unsupported file type."""
        files = {"files": ("test.txt", b"some text", "text/plain")}
        response = client.post("/upload/", files=files)
        
        assert response.status_code == 422
        assert "Unsupported file type" in response.json()["detail"]

    @patch("app.api.upload.get_extractor")
    def test_ask_nonexistent_session(self, mock_get_extractor, client):
        """Test ask with non-existent session."""
        ask_request = {"session_id": "non-existent-session", "question": "What is this?"}
        response = client.post("/ask/", json=ask_request)
        
        assert response.status_code == 404

    @patch("app.api.upload.get_extractor")
    def test_ask_empty_session(self, mock_get_extractor, client):
        """Test ask with empty session (no documents)."""
        # Create a session but don't upload any documents
        from app.services.session.service import create_session
        session_id = create_session()
        
        ask_request = {"session_id": session_id, "question": "What is this?"}
        response = client.post("/ask/", json=ask_request)
        
        assert response.status_code == 400
        assert "No documents found" in response.json()["detail"]
