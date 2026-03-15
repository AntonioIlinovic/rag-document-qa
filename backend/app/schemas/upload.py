"""Upload API schemas for RAG Document QA.

Defines Pydantic models for upload request/response validation
and API documentation generation.
"""

from typing import List
from pydantic import BaseModel, Field


class DocumentInfo(BaseModel):
    """API response model for document processing information.
    
    This represents the shape of document information returned
    to clients via the upload endpoint. It's separate from the
    internal DocumentMeta model to maintain clean separation
    between service layer and HTTP layer.
    """
    filename: str = Field(..., description="Original filename of the uploaded document")
    chunks: int = Field(..., description="Number of text chunks created from the document")
    status: str = Field(..., description="Processing status (e.g., 'processed', 'failed')")


class UploadResponse(BaseModel):
    """API response model for document upload endpoint.
    
    Returns session ID and processing information for all uploaded documents.
    """
    session_id: str = Field(..., description="Session ID for the uploaded documents")
    documents: List[DocumentInfo] = Field(..., description="List of processed documents with metadata")
