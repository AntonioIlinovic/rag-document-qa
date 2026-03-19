"""Session schemas for RAG Document QA API.

Defines the request/response models for session management endpoints.
"""

from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel, Field


class DocumentSummary(BaseModel):
    """Summary of a document in a session."""
    filename: str = Field(..., description="Document filename")
    chunks: int = Field(..., description="Number of chunks created from document")
    status: str = Field(..., description="Processing status")


class SessionSummary(BaseModel):
    """Summary of a session for listing."""
    session_id: str = Field(..., description="Session ID")
    created_at: datetime = Field(..., description="Session creation timestamp")
    document_count: int = Field(..., description="Number of documents in session")
    filenames: List[str] = Field(..., description="List of document filenames")
    total_chunks: int = Field(..., description="Total number of chunks across all documents")


class SessionDetails(BaseModel):
    """Detailed session information."""
    session_id: str = Field(..., description="Session ID")
    created_at: datetime = Field(..., description="Session creation timestamp")
    documents: List[DocumentSummary] = Field(..., description="List of documents in session")
    document_count: int = Field(..., description="Number of documents in session")
    total_chunks: int = Field(..., description="Total number of chunks across all documents")


class SessionsListResponse(BaseModel):
    """Response for sessions list endpoint."""
    sessions: List[SessionSummary] = Field(..., description="List of session summaries")
