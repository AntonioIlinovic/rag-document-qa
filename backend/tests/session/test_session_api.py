"""Tests for session data persistence via API."""

from unittest.mock import Mock, patch
import pytest

class TestSessionPersistence:
    """Test cases for session data persistence."""

    @patch("app.api.upload.get_extractor")
    @patch("app.api.upload.session_get_pipeline_for_session")
    @patch("app.api.upload.session_create_session")
    @patch("app.api.upload.add_document_to_session")
    def test_session_data_persistence(self, mock_add_doc, mock_create_session, mock_get_pipeline_for_session, mock_get_extractor, client, mock_pdf_content, mock_extracted_text, tmp_path):
        """Test that session data is persisted correctly."""
        session_id = "test-session-id"
        mock_create_session.return_value = session_id
        
        # Mock extractor
        mock_extractor = Mock()
        mock_extractor.extract.return_value = mock_extracted_text
        mock_get_extractor.return_value = mock_extractor
        
        # Mock pipeline
        mock_pipeline = Mock()
        mock_pipeline.ingest.return_value = 3
        mock_get_pipeline_for_session.return_value = mock_pipeline
        
        # Upload file
        files = {"files": ("test.pdf", mock_pdf_content, "application/pdf")}
        upload_response = client.post("/upload/", files=files)
        assert upload_response.status_code == 200
        
        session_id = upload_response.json()["session_id"]
        
        # Verify add_document_to_session was called
        assert mock_add_doc.called
        call_args = mock_add_doc.call_args
        assert call_args[1]['filename'] == 'test.pdf'
        assert call_args[1]['chunks_created'] == 3
