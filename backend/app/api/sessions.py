"""Sessions endpoint for RAG Document QA.

Provides session listing and retrieval functionality.
"""

import logging
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status

from app.config import settings
from app.schemas.session import (
    SessionDetails,
    SessionSummary,
    SessionsListResponse
)
from app.services.session import (
    get_session,
    get_session_document_count,
    get_session_total_chunks
)
from app.services.session.store import session_exists

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/sessions", tags=["sessions"])


def list_all_sessions() -> list[SessionSummary]:
    """List all available sessions.
    
    Returns:
        List of session summaries
        
    Raises:
        OSError: If session directory cannot be accessed
    """
    sessions_path = Path(settings.app_data_dir) / "sessions"
    
    if not sessions_path.exists():
        logger.info("Sessions directory does not exist")
        return []
    
    sessions = []
    
    # Iterate through session directories
    for session_dir in sessions_path.iterdir():
        if not session_dir.is_dir():
            continue
            
        session_id = session_dir.name
        session_file = session_dir / "session.json"
        
        if not session_file.exists():
            logger.warning(f"Session file not found for {session_id}")
            continue
        
        try:
            # Load session data
            session_data = get_session(session_id)
            
            # Create summary
            filenames = [doc.filename for doc in session_data.documents]
            
            session_summary = SessionSummary(
                session_id=session_id,
                created_at=session_data.created_at,
                document_count=len(session_data.documents),
                filenames=filenames,
                total_chunks=session_data.get_total_chunks()
            )
            
            sessions.append(session_summary)
            
        except Exception as exc:
            logger.error(f"Error loading session {session_id}: {exc}")
            continue
    
    # Sort by creation time (newest first)
    sessions.sort(key=lambda s: s.created_at, reverse=True)
    
    return sessions


@router.get("/", response_model=SessionsListResponse, status_code=status.HTTP_200_OK)
async def list_sessions() -> SessionsListResponse:
    """List all available sessions.
    
    Returns a list of all sessions with their metadata including
    session ID, creation time, document count, and filenames.
    
    Returns:
        SessionsListResponse with list of session summaries
    """
    try:
        sessions = list_all_sessions()
        logger.info(f"Listed {len(sessions)} sessions")
        
        return SessionsListResponse(sessions=sessions)
        
    except Exception as exc:
        logger.error(f"Error listing sessions: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error listing sessions"
        ) from exc


@router.get("/{session_id}", response_model=SessionDetails, status_code=status.HTTP_200_OK)
async def get_session_details(
    session_id: str
) -> SessionDetails:
    """Get detailed information about a specific session.
    
    Returns complete session metadata including all documents
    and their processing details.
    
    Args:
        session_id: UUID string identifying the session
        
    Returns:
        SessionDetails with complete session information
        
    Raises:
        HTTPException: If session not found
    """
    try:
        if not session_exists(session_id):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Session {session_id} not found"
            )
        
        session_data = get_session(session_id)
        
        # Convert documents to summary format
        from app.schemas.session import DocumentSummary
        documents = [
            DocumentSummary(
                filename=doc.filename,
                chunks=doc.chunks,
                status=doc.status
            )
            for doc in session_data.documents
        ]
        
        session_details = SessionDetails(
            session_id=session_id,
            created_at=session_data.created_at,
            documents=documents,
            document_count=len(session_data.documents),
            total_chunks=session_data.get_total_chunks()
        )
        
        logger.info(f"Retrieved details for session {session_id}")
        return session_details
        
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"Error getting session details for {session_id}: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error retrieving session details"
        ) from exc
