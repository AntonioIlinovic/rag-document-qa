"""Tests for the upload endpoint."""

from unittest.mock import Mock, patch
import pytest

class TestUploadEndpoint:
    """Test cases for the upload endpoint."""

    @patch("app.api.upload.get_extractor")
    def test_upload_new_session_success(self, mock_get_extractor, client, mock_pdf_content, mock_extracted_text):
        """Test successful upload to new session."""
        # Mock extractor
        mock_extractor = Mock()
        mock_extractor.extract.return_value = mock_extracted_text
        mock_get_extractor.return_value = mock_extractor
        
        # Test file upload
        files = {"files": ("test.pdf", mock_pdf_content, "application/pdf")}
        response = client.post("/upload/", files=files)
        
        assert response.status_code == 200
        data = response.json()
        assert "session_id" in data
        assert len(data["documents"]) == 1
        assert data["documents"][0]["filename"] == "test.pdf"
        assert data["documents"][0]["status"] == "processed"
        # The actual chunk count will vary based on the text, so just check it's positive
        assert data["documents"][0]["chunks"] > 0

    @patch("app.api.upload.get_extractor")
    @patch("app.api.upload.session_get_pipeline_for_session")
    @patch("app.api.upload.get_session")
    @patch("app.api.upload.add_document_to_session")
    def test_upload_existing_session_success(self, mock_add_doc, mock_get_session, mock_get_pipeline_for_session, mock_get_extractor, client, mock_pdf_content, mock_extracted_text):
        """Test successful upload to existing session."""
        session_id = "12345678-1234-5678-9abc-123456789abc"
        
        # Mock existing session
        mock_session = Mock()
        mock_session.session_id = session_id
        mock_get_session.return_value = mock_session
        
        # Mock extractor
        mock_extractor = Mock()
        mock_extractor.extract.return_value = mock_extracted_text
        mock_get_extractor.return_value = mock_extractor
        
        # Mock pipeline
        mock_pipeline = Mock()
        mock_pipeline.ingest.return_value = 2
        mock_get_pipeline_for_session.return_value = mock_pipeline
        
        # Test file upload to existing session
        files = {"files": ("test2.pdf", mock_pdf_content, "application/pdf")}
        data = {"session_id": session_id}
        response = client.post("/upload/", files=files, data=data)
        
        assert response.status_code == 200
        response_data = response.json()
        assert response_data["session_id"] == session_id
        assert len(response_data["documents"]) == 1

    def test_upload_no_files(self, client):
        """Test upload with no files provided."""
        response = client.post("/upload/")
        assert response.status_code == 422
        # Pydantic v2 returns a list of error objects in 'detail'
        detail = response.json()["detail"]
        assert any(err["msg"] == "Field required" and "files" in err["loc"] for err in detail)

    def test_upload_unsupported_file_type(self, client):
        """Test upload with unsupported file type."""
        files = {"files": ("test.txt", b"some text", "text/plain")}
        response = client.post("/upload/", files=files)
        assert response.status_code == 422
        assert "Unsupported file type" in response.json()["detail"]

    def test_upload_session_not_found(self, client, mock_pdf_content):
        """Test upload to non-existent session."""
        files = {"files": ("test.pdf", mock_pdf_content, "application/pdf")}
        data = {"session_id": "non-existent-session"}
        response = client.post("/upload/", files=files, data=data)
        assert response.status_code == 404

    @patch("app.api.upload.get_extractor")
    @patch("app.services.session.service.get_pipeline_for_session")
    @patch("app.services.session.service.create_session")
    def test_upload_extraction_failure(self, mock_create_session, mock_get_pipeline_for_session, mock_get_extractor, client, mock_pdf_content):
        """Test upload when text extraction fails."""
        session_id = "test-session-id"
        mock_create_session.return_value = session_id
        
        # Mock extractor to raise ExtractionError
        from app.services.extraction.base import ExtractionError
        mock_extractor = Mock()
        mock_extractor.extract.side_effect = ExtractionError("Failed to extract text")
        mock_get_extractor.return_value = mock_extractor
        
        # Mock pipeline
        mock_pipeline = Mock()
        mock_get_pipeline_for_session.return_value = mock_pipeline
        
        files = {"files": ("test.pdf", mock_pdf_content, "application/pdf")}
        response = client.post("/upload/", files=files)
        
        assert response.status_code == 422
        assert "Text extraction failed" in response.json()["detail"]
