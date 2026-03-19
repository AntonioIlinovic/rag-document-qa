"""Chat endpoints for RAG Document QA.

Handles chat history storage and retrieval operations.
"""

import logging
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status

from app.schemas.chat import (
    ChatMessageRequest, 
    ChatMessageResponse, 
    ChatHistoryResponse,
    ChatMessage
)
from app.services.session.service import add_chat_message, get_chat_history
from .deps import get_session_from_chat_request, get_session_from_path

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("/message", response_model=ChatMessageResponse, status_code=status.HTTP_200_OK)
async def save_chat_message(
    request: ChatMessageRequest,
    session: Annotated[dict, Depends(get_session_from_chat_request)]
) -> ChatMessageResponse:
    """Save a chat message to session history.
    
    Args:
        request: ChatMessageRequest with session_id, role, content, and optional details
        session: Session data (injected from dependency)
        
    Returns:
        ChatMessageResponse indicating success
        
    Raises:
        HTTPException: For various processing errors
    """
    if not request.role.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Role cannot be empty"
        )
    
    if request.role not in ["user", "assistant"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Role must be 'user' or 'assistant'"
        )
    
    if not request.content.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Content cannot be empty"
        )
    
    try:
        logger.info(f"Saving {request.role} message for session {request.session_id}")
        
        # Add message to chat history
        add_chat_message(
            session_id=request.session_id,
            role=request.role,
            content=request.content,
            details=request.details
        )
        
        logger.info(f"Successfully saved {request.role} message for session {request.session_id}")
        
        return ChatMessageResponse(
            success=True,
            message=f"{request.role.capitalize()} message saved successfully"
        )
        
    except Exception as exc:
        logger.error(f"Error saving chat message: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error saving chat message: {str(exc)}"
        ) from exc


@router.get("/history/{session_id}", response_model=ChatHistoryResponse, status_code=status.HTTP_200_OK)
async def get_session_chat_history(
    session_id: str,
    session: Annotated[dict, Depends(get_session_from_path)]
) -> ChatHistoryResponse:
    """Get chat history for a session.
    
    Args:
        session_id: UUID string identifying the session
        session: Session data (injected from dependency)
        
    Returns:
        ChatHistoryResponse with list of messages
        
    Raises:
        HTTPException: For various processing errors
    """
    try:
        logger.info(f"Retrieving chat history for session {session_id}")
        
        # Get chat history
        messages_data = get_chat_history(session_id)
        
        # Convert to API response format
        messages = [
            ChatMessage(
                role=msg["role"],
                content=msg["content"],
                timestamp=msg["timestamp"],
                details=msg.get("details")
            )
            for msg in messages_data
        ]
        
        logger.info(f"Retrieved {len(messages)} messages for session {session_id}")
        
        return ChatHistoryResponse(
            session_id=session_id,
            messages=messages,
            message_count=len(messages)
        )
        
    except Exception as exc:
        logger.error(f"Error retrieving chat history: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving chat history: {str(exc)}"
        ) from exc
