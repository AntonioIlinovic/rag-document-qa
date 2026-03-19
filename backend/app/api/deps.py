"""FastAPI dependencies for RAG Document QA.

Provides dependency injection functions for session management,
QA engine selection, and other common API dependencies.
"""

from typing import Annotated, Optional

from fastapi import Depends, HTTPException, status
from app.config import Settings, settings
from app.schemas.ask import AskRequest
from app.schemas.chat import ChatMessageRequest
from app.services.qa.base import BaseQAEngine
from app.services.rag.pipeline import BaseRAGPipeline
from app.services.ner.base import BaseNERExtractor
from app.services.session.models import SessionData, SessionNotFoundError
from app.services.session.service import get_session, get_pipeline_for_session


async def get_settings() -> Settings:
    """Dependency to get application settings.
    
    Returns:
        Application settings instance
    """
    return settings


async def get_session_data_from_request(request: AskRequest) -> SessionData:
    """Dependency to validate and retrieve session data from request body.
    
    Args:
        request: AskRequest containing session_id
        
    Returns:
        SessionData object with session metadata
        
    Raises:
        HTTPException: 404 if session not found
    """
    try:
        return get_session(request.session_id)
    except SessionNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session {request.session_id} not found"
        ) from exc


async def get_pipeline_from_request(
    request: AskRequest,
    session_data: Annotated[SessionData, Depends(get_session_data_from_request)]
) -> BaseRAGPipeline:
    """Dependency to get RAG pipeline for a session from request.
    
    Args:
        request: AskRequest containing session_id
        session_data: Validated session data
        
    Returns:
        RAGPipeline instance configured for the session
    """
    return get_pipeline_for_session(session_data.session_id)


async def get_qa_engine(
    app_settings: Annotated[Settings, Depends(get_settings)]
) -> BaseQAEngine:
    """Dependency to get QA engine based on configuration."""
    from app.services.qa import get_qa_engine
    return get_qa_engine(app_settings)


async def get_ner_extractor(
    app_settings: Annotated[Settings, Depends(get_settings)]
) -> Optional[BaseNERExtractor]:
    """Dependency to get NER extractor based on configuration.
    
    Returns:
        NER extractor instance or None if NER is disabled
    """
    from app.services.ner import get_ner_extractor
    return get_ner_extractor(app_settings)


async def get_session_from_path(session_id: str) -> SessionData:
    """Dependency to validate and retrieve session data from path parameter.
    
    Args:
        session_id: Session ID from URL path
        
    Returns:
        SessionData object with session metadata
        
    Raises:
        HTTPException: 404 if session not found
    """
    try:
        return get_session(session_id)
    except SessionNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session {session_id} not found"
        ) from exc


async def get_session_from_chat_request(request: ChatMessageRequest) -> SessionData:
    """Dependency to validate and retrieve session data from chat request.
    
    Args:
        request: ChatMessageRequest containing session_id
        
    Returns:
        SessionData object with session metadata
        
    Raises:
        HTTPException: 404 if session not found
    """
    try:
        return get_session(request.session_id)
    except SessionNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session {request.session_id} not found"
        ) from exc
