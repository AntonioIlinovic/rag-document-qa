"""Chat API schemas for RAG Document QA.

Pydantic models for chat history API requests and responses.
"""

from datetime import datetime
from typing import List, Optional, Dict, Any

from pydantic import BaseModel, Field


class ChatMessageRequest(BaseModel):
    """Request model for adding a chat message."""
    session_id: str = Field(..., description="Session ID")
    role: str = Field(..., description="Message role: 'user' or 'assistant'")
    content: str = Field(..., description="Message content")
    details: Optional[Dict[str, Any]] = Field(None, description="Optional details for assistant messages")


class ChatMessageResponse(BaseModel):
    """Response model for chat message operations."""
    success: bool = Field(True, description="Operation success status")
    message: str = Field(..., description="Response message")


class ChatMessage(BaseModel):
    """Chat message model for API responses."""
    role: str = Field(..., description="Message role: 'user' or 'assistant'")
    content: str = Field(..., description="Message content")
    timestamp: datetime = Field(..., description="Message timestamp")
    details: Optional[Dict[str, Any]] = Field(None, description="Optional details for assistant messages")


class ChatHistoryResponse(BaseModel):
    """Response model for chat history retrieval."""
    session_id: str = Field(..., description="Session ID")
    messages: List[ChatMessage] = Field(..., description="List of chat messages")
    message_count: int = Field(..., description="Total number of messages")
