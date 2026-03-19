"""Session data models for RAG Document QA.

This module defines the internal data structures used for session management.
These are separate from API schemas to maintain clean separation between
service layer and HTTP layer.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional
from uuid import UUID


class SessionNotFoundError(Exception):
    """Raised when a session ID cannot be found or accessed.
    
    This exception is used by the session service and caught by API
    dependencies to return appropriate HTTP 404 responses.
    """
    pass


@dataclass
class ChatMessage:
    """Chat message data structure.
    
    Represents a single chat message in the conversation history.
    Matches the frontend chat message format exactly.
    """
    role: str  # "user" or "assistant"
    content: str
    timestamp: datetime
    details: Optional[dict] = None  # For assistant messages with source info


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
    
    Contains all session metadata, document information, and chat history.
    Stored as JSON in app_data/sessions/{session_id}/session.json.
    Chat history stored separately in app_data/sessions/{session_id}/chat.json.
    """
    session_id: str
    created_at: datetime
    documents: List[DocumentMeta]
    chat_messages: List[ChatMessage] = None
    
    def __post_init__(self) -> None:
        """Validate session ID format and initialize chat messages."""
        if self.chat_messages is None:
            self.chat_messages = []
        
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
    
    def add_chat_message(self, role: str, content: str, details: Optional[dict] = None) -> None:
        """Add a chat message to the session.
        
        Args:
            role: Message role ("user" or "assistant")
            content: Message content
            details: Optional details for assistant messages (sources, models, etc.)
        """
        message = ChatMessage(
            role=role,
            content=content,
            timestamp=datetime.now(),
            details=details
        )
        self.chat_messages.append(message)
    
    def get_chat_message_count(self) -> int:
        """Get the total number of chat messages in the session."""
        return len(self.chat_messages)
