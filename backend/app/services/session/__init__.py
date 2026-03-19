"""Session management service for RAG Document QA.

This package provides session lifecycle management with per-session
document storage and ChromaDB collections.

Public API:
- create_session() -> str
- get_session(session_id) -> SessionData
- get_pipeline_for_session(session_id) -> RAGPipeline
- add_document_to_session(session_id, filename, file_bytes, extracted_text, chunks_created) -> None
- get_session_document_count(session_id) -> int
- get_session_total_chunks(session_id) -> int
"""

from .service import (
    create_session, 
    get_session, 
    get_pipeline_for_session,
    add_document_to_session,
    get_session_document_count,
    get_session_total_chunks
)

__all__ = [
    "create_session",
    "get_session", 
    "get_pipeline_for_session",
    "add_document_to_session",
    "get_session_document_count",
    "get_session_total_chunks",
]
