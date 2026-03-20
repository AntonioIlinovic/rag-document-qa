"""Tests for the ask endpoint."""

from unittest.mock import AsyncMock, Mock
import pytest
from fastapi import HTTPException
from app.api.deps import get_pipeline_from_request, get_session_data_from_request

class TestAskEndpoint:
    """Test cases for the ask endpoint."""

    def test_ask_success(self, client, mock_qa_answer):
        """Test successful question answering."""
        session_id = "12345678-1234-5678-9abc-123456789abc"
        
        # Mock the pipeline dependency
        mock_pipeline = Mock()
        mock_pipeline.count_documents.return_value = 2
        mock_pipeline.query.return_value = [
            {"chunk": "Sample document text with relevant information.", "score": 0.87},
            {"chunk": "Another relevant chunk from the documents.", "score": 0.76}
        ]
        
        # Mock QA engine
        mock_qa_engine = AsyncMock()
        mock_qa_engine.answer.return_value = mock_qa_answer
        
        # Setup dependency overrides
        client.app.dependency_overrides[get_pipeline_from_request] = lambda: mock_pipeline
        
        # Test ask request
        request_data = {"session_id": session_id, "question": "What is the termination clause?"}
        response = client.post("/ask/", json=request_data)
        
        # Clear overrides for other tests
        client.app.dependency_overrides.clear()
        
        assert response.status_code == 200
        data = response.json()
        assert data["answer"] == mock_qa_answer
        assert len(data["sources"]) == 2
        assert data["sources"][0]["score"] == 0.87

    def test_ask_session_not_found(self, client):
        """Test ask with non-existent session."""
        def override_get_session_data():
            raise HTTPException(status_code=404, detail="Session not found")
            
        client.app.dependency_overrides[get_session_data_from_request] = override_get_session_data
        
        request_data = {"session_id": "non-existent-session", "question": "What is the termination clause?"}
        response = client.post("/ask/", json=request_data)
        
        client.app.dependency_overrides.clear()
        assert response.status_code == 404

    def test_ask_empty_session(self, client):
        """Test ask with session that has no documents."""
        session_id = "12345678-1234-5678-9abc-123456789abc"
        
        # Mock pipeline to return 0 documents
        mock_pipeline = Mock()
        mock_pipeline.count_documents.return_value = 0
        
        client.app.dependency_overrides[get_pipeline_from_request] = lambda: mock_pipeline
        
        request_data = {"session_id": session_id, "question": "What is the termination clause?"}
        response = client.post("/ask/", json=request_data)
        
        client.app.dependency_overrides.clear()
        assert response.status_code == 400
        assert "No documents found" in response.json()["detail"]

    def test_ask_empty_question(self, client):
        """Test ask with empty question."""
        session_id = "12345678-1234-5678-9abc-123456789abc"
        
        # Mock dependencies just in case
        client.app.dependency_overrides[get_pipeline_from_request] = lambda: Mock()
        client.app.dependency_overrides[get_qa_engine] = lambda: Mock()
        
        request_data = {"session_id": session_id, "question": ""}
        response = client.post("/ask/", json=request_data)
        
        client.app.dependency_overrides.clear()
        assert response.status_code == 400

    def test_ask_no_relevant_chunks(self, client):
        """Test ask when no relevant chunks are found."""
        session_id = "12345678-1234-5678-9abc-123456789abc"
        
        # Mock pipeline with documents but no search results
        mock_pipeline = Mock()
        mock_pipeline.count_documents.return_value = 5
        mock_pipeline.query.return_value = []
        
        client.app.dependency_overrides[get_pipeline_from_request] = lambda: mock_pipeline
        
        request_data = {"session_id": session_id, "question": "What is the termination clause?"}
        response = client.post("/ask/", json=request_data)
        
        client.app.dependency_overrides.clear()
        assert response.status_code == 200
        data = response.json()
        assert "couldn't find relevant information" in data["answer"]
        assert len(data["sources"]) == 0
