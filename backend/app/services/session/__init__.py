"""Session management service for RAG Document QA.

This package provides session lifecycle management with per-session
document storage and ChromaDB collections.

Public API:
- create_session() -> str
- get_session(session_id) -> SessionData
- delete_session(session_id) -> None
- get_pipeline_for_session(session_id) -> RAGPipeline
"""

from .service import (
    create_session, 
    get_session, 
    delete_session, 
    get_pipeline_for_session,
    add_document_to_session
)

__all__ = [
    "create_session",
    "get_session", 
    "delete_session",
    "get_pipeline_for_session",
    "add_document_to_session",
]
