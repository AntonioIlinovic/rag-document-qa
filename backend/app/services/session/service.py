"""Session service business logic for RAG Document QA.

Provides high-level session management operations including
session creation, retrieval, deletion, and pipeline management.
"""

from datetime import datetime
from pathlib import Path
from typing import List, Optional
from uuid import uuid4

from app.services.rag.pipeline import get_pipeline, BaseRAGPipeline
from .models import SessionData, SessionNotFoundError
from .store import (
    save_session, 
    load_session, 
    delete_session_files,
    chroma_dir,
    original_files_dir,
    extracted_files_dir,
    load_chat_history,
    save_chat_history
)


def create_session() -> str:
    """Create a new session with unique ID.
    
    Returns:
        New session ID (UUID4 string)
    """
    session_id = str(uuid4())
    
    # Create initial session data
    session_data = SessionData(
        session_id=session_id,
        created_at=datetime.now(),
        documents=[]
    )
    
    # Save session metadata
    save_session(session_data)
    
    return session_id


def get_session(session_id: str) -> SessionData:
    """Retrieve session data by ID.
    
    Args:
        session_id: UUID string identifying the session
        
    Returns:
        SessionData object with session metadata
        
    Raises:
        SessionNotFoundError: If session doesn't exist
    """
    return load_session(session_id)


def delete_session(session_id: str) -> None:
    """Delete a session and all its associated data.
    
    Args:
        session_id: UUID string identifying the session
        
    Raises:
        SessionNotFoundError: If session doesn't exist
    """
    # Verify session exists before deletion
    load_session(session_id)  # Will raise SessionNotFoundError if not found
    
    # Delete all session files
    delete_session_files(session_id)


def get_pipeline_for_session(session_id: str) -> BaseRAGPipeline:
    """Get a RAG pipeline configured for a specific session.
    
    Args:
        session_id: UUID string identifying the session
        
    Returns:
        RAGPipeline instance with session-scoped ChromaDB storage
        
    Raises:
        SessionNotFoundError: If session doesn't exist
    """
    # Verify session exists
    load_session(session_id)  # Will raise SessionNotFoundError if not found
    
    # Get ChromaDB directory for this session
    persist_directory = str(chroma_dir(session_id))
    
    # Use session-specific collection name
    collection_name = f"session_{session_id}"
    
    # Create and return pipeline with session-scoped storage
    from app.services.rag.pipeline import get_pipeline
    return get_pipeline(persist_directory=persist_directory, collection_name=collection_name)


def add_document_to_session(
    session_id: str, 
    filename: str, 
    file_bytes: bytes, 
    extracted_text: str,
    chunks_created: int
) -> None:
    """Add a processed document to a session.
    
    Args:
        session_id: UUID string identifying the session
        filename: Original filename
        file_bytes: Original file bytes
        extracted_text: Extracted text content
        chunks_created: Number of chunks created from the text
        
    Raises:
        SessionNotFoundError: If session doesn't exist
    """
    # Load current session data
    session_data = load_session(session_id)
    
    # Save original file
    original_path = original_files_dir(session_id) / filename
    with open(original_path, 'wb') as f:
        f.write(file_bytes)
    
    # Save extracted text (for debugging/verification)
    extracted_filename = Path(filename).stem + '.txt'
    extracted_path = extracted_files_dir(session_id) / extracted_filename
    with open(extracted_path, 'w', encoding='utf-8') as f:
        f.write(extracted_text)
    
    # Add document metadata to session
    session_data.add_document(
        filename=filename,
        chunks=chunks_created,
        status="processed"
    )
    
    # Save updated session data
    save_session(session_data)


def get_session_document_count(session_id: str) -> int:
    """Get the number of documents in a session.
    
    Args:
        session_id: UUID string identifying the session
        
    Returns:
        Number of documents in the session
        
    Raises:
        SessionNotFoundError: If session doesn't exist
    """
    session_data = load_session(session_id)
    return session_data.get_document_count()


def get_session_total_chunks(session_id: str) -> int:
    """Get the total number of chunks in a session.
    
    Args:
        session_id: UUID string identifying the session
        
    Returns:
        Total number of chunks across all documents
        
    Raises:
        SessionNotFoundError: If session doesn't exist
    """
    session_data = load_session(session_id)
    return session_data.get_total_chunks()


def add_chat_message(session_id: str, role: str, content: str, details: dict = None) -> None:
    """Add a chat message to a session.
    
    Args:
        session_id: UUID string identifying the session
        role: Message role ("user" or "assistant")
        content: Message content
        details: Optional details for assistant messages (sources, models, etc.)
        
    Raises:
        SessionNotFoundError: If session doesn't exist
    """
    # Verify session exists
    load_session(session_id)  # Will raise SessionNotFoundError if not found
    
    # Load existing chat history
    messages = load_chat_history(session_id)
    
    # Add new message
    new_message = {
        'role': role,
        'content': content,
        'timestamp': datetime.now(),
        'details': details
    }
    messages.append(new_message)
    
    # Save updated chat history
    save_chat_history(session_id, messages)


def get_chat_history(session_id: str) -> List:
    """Get chat history for a session.
    
    Args:
        session_id: UUID string identifying the session
        
    Returns:
        List of chat message dictionaries
        
    Raises:
        SessionNotFoundError: If session doesn't exist
    """
    # Verify session exists
    load_session(session_id)  # Will raise SessionNotFoundError if not found
    
    return load_chat_history(session_id)
