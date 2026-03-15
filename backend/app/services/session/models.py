"""Session data models for RAG Document QA.

This module defines the internal data structures used for session management.
These are separate from API schemas to maintain clean separation between
service layer and HTTP layer.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import List
from uuid import UUID


class SessionNotFoundError(Exception):
    """Raised when a session ID cannot be found or accessed.
    
    This exception is used by the session service and caught by API
    dependencies to return appropriate HTTP 404 responses.
    """
    pass


@dataclass
class DocumentMeta:
    """Internal metadata for a processed document.
    
    This represents the stored document information within a session,
    separate from the API response shape (DocumentInfo).
    """
    filename: str
    chunks: int
    status: str


@dataclass
class SessionData:
    """Internal session data structure.
    
    Contains all session metadata and document information.
    Stored as JSON in app_data/sessions/{session_id}/session.json.
    """
    session_id: str
    created_at: datetime
    documents: List[DocumentMeta]
    
    def __post_init__(self) -> None:
        """Validate session ID format."""
        try:
            UUID(self.session_id)
        except ValueError as exc:
            raise ValueError(f"Invalid session_id format: {self.session_id}") from exc
    
    def add_document(self, filename: str, chunks: int, status: str = "processed") -> None:
        """Add a document to the session.
        
        Args:
            filename: Name of the processed document
            chunks: Number of chunks created from the document
            status: Processing status (processed, failed, etc.)
        """
        doc_meta = DocumentMeta(filename=filename, chunks=chunks, status=status)
        self.documents.append(doc_meta)
    
    def get_document_count(self) -> int:
        """Get the total number of documents in the session."""
        return len(self.documents)
    
    def get_total_chunks(self) -> int:
        """Get the total number of chunks across all documents."""
        return sum(doc.chunks for doc in self.documents)
