"""Session persistence layer for RAG Document QA.

Handles JSON-based storage of session metadata and provides
path utilities for session directory structure.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from app.config import settings
from .models import SessionData, SessionNotFoundError


def session_dir(session_id: str) -> Path:
    """Get the directory path for a session without creating it.
    
    Args:
        session_id: UUID string identifying the session
        
    Returns:
        Path to the session directory
    """
    base_dir = Path(settings.app_data_dir) / "sessions" / session_id
    return base_dir


def ensure_session_dir(session_id: str) -> Path:
    """Get and create the directory path for a session.
    
    Args:
        session_id: UUID string identifying the session
        
    Returns:
        Path to the session directory (created if needed)
    """
    base_dir = session_dir(session_id)
    base_dir.mkdir(parents=True, exist_ok=True)
    return base_dir


def chroma_dir(session_id: str) -> Path:
    """Get the ChromaDB directory path for a session.
    
    Args:
        session_id: UUID string identifying the session
        
    Returns:
        Path to the session's ChromaDB directory (created if needed)
    """
    chroma_path = ensure_session_dir(session_id) / "chroma_db"
    chroma_path.mkdir(parents=True, exist_ok=True)
    return chroma_path


def documents_dir(session_id: str) -> Path:
    """Get the documents directory path for a session.
    
    Args:
        session_id: UUID string identifying the session
        
    Returns:
        Path to the session's documents directory (created if needed)
    """
    docs_path = ensure_session_dir(session_id) / "documents"
    docs_path.mkdir(parents=True, exist_ok=True)
    return docs_path


def original_files_dir(session_id: str) -> Path:
    """Get the original files directory path for a session.
    
    Args:
        session_id: UUID string identifying the session
        
    Returns:
        Path to the session's original files directory (created if needed)
    """
    original_path = documents_dir(session_id) / "original"
    original_path.mkdir(parents=True, exist_ok=True)
    return original_path


def extracted_files_dir(session_id: str) -> Path:
    """Get the extracted files directory path for a session.
    
    Args:
        session_id: UUID string identifying the session
        
    Returns:
        Path to the session's extracted files directory (created if needed)
    """
    extracted_path = documents_dir(session_id) / "extracted"
    extracted_path.mkdir(parents=True, exist_ok=True)
    return extracted_path


def session_file_path(session_id: str) -> Path:
    """Get the session metadata file path.
    
    Args:
        session_id: UUID string identifying the session
        
    Returns:
        Path to the session.json file
    """
    return session_dir(session_id) / "session.json"


def chat_file_path(session_id: str) -> Path:
    """Get the chat history file path.
    
    Args:
        session_id: UUID string identifying the session
        
    Returns:
        Path to the chat.json file
    """
    return session_dir(session_id) / "chat.json"


def load_session(session_id: str) -> SessionData:
    """Load session data from JSON file.
    
    Args:
        session_id: UUID string identifying the session
        
    Returns:
        SessionData object with loaded metadata
        
    Raises:
        SessionNotFoundError: If session file doesn't exist or is invalid
    """
    session_path = session_file_path(session_id)
    
    if not session_path.exists():
        raise SessionNotFoundError(f"Session {session_id} not found")
    
    try:
        with open(session_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Convert created_at string back to datetime
        created_at_str = data.get('created_at')
        if created_at_str:
            created_at = datetime.fromisoformat(created_at_str)
        else:
            created_at = datetime.now()
        
        # Reconstruct SessionData (chat messages loaded separately)
        from .models import DocumentMeta
        documents = [
            DocumentMeta(**doc_data) for doc_data in data.get('documents', [])
        ]
        
        return SessionData(
            session_id=data['session_id'],
            created_at=created_at,
            documents=documents,
            chat_messages=[]  # Will be loaded separately when needed
        )
        
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        raise SessionNotFoundError(f"Invalid session data for {session_id}: {exc}") from exc


def save_session(session_data: SessionData) -> None:
    """Save session data to JSON file.
    
    Args:
        session_data: SessionData object to save
        
    Raises:
        OSError: If file write fails
    """
    session_path = session_file_path(session_data.session_id)
    
    # Ensure session directory exists before saving
    ensure_session_dir(session_data.session_id)
    
    # Convert SessionData to JSON-serializable dict
    data = {
        'session_id': session_data.session_id,
        'created_at': session_data.created_at.isoformat(),
        'documents': [
            {
                'filename': doc.filename,
                'chunks': doc.chunks,
                'status': doc.status
            }
            for doc in session_data.documents
        ]
    }
    
    with open(session_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def session_exists(session_id: str) -> bool:
    """Check if a session exists.
    
    Args:
        session_id: UUID string identifying the session
        
    Returns:
        True if session file exists, False otherwise
    """
    return session_file_path(session_id).exists()


def delete_session_files(session_id: str) -> None:
    """Delete all files associated with a session.
    
    Args:
        session_id: UUID string identifying the session
        
    Raises:
        OSError: If deletion fails
    """
    session_path = session_dir(session_id)
    if session_path.exists():
        import shutil
        shutil.rmtree(session_path)


def load_chat_history(session_id: str) -> List:
    """Load chat history from JSON file.
    
    Args:
        session_id: UUID string identifying the session
        
    Returns:
        List of chat message dictionaries
        
    Raises:
        SessionNotFoundError: If session doesn't exist
    """
    # Verify session exists first
    if not session_exists(session_id):
        raise SessionNotFoundError(f"Session {session_id} not found")
    
    chat_path = chat_file_path(session_id)
    
    if not chat_path.exists():
        return []  # No chat history yet
    
    try:
        with open(chat_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Convert timestamp strings back to datetime objects
        messages = []
        for msg_data in data:
            timestamp_str = msg_data.get('timestamp')
            if timestamp_str:
                timestamp = datetime.fromisoformat(timestamp_str)
            else:
                timestamp = datetime.now()
            
            messages.append({
                'role': msg_data['role'],
                'content': msg_data['content'],
                'timestamp': timestamp,
                'details': msg_data.get('details')
            })
        
        return messages
        
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        # If chat file is corrupted, return empty list and log error
        import logging
        logging.error(f"Invalid chat data for {session_id}: {exc}")
        return []


def save_chat_history(session_id: str, messages: List) -> None:
    """Save chat history to JSON file.
    
    Args:
        session_id: UUID string identifying the session
        messages: List of chat message dictionaries
        
    Raises:
        SessionNotFoundError: If session doesn't exist
        OSError: If file write fails
    """
    # Verify session exists first
    if not session_exists(session_id):
        raise SessionNotFoundError(f"Session {session_id} not found")
    
    chat_path = chat_file_path(session_id)
    
    # Ensure session directory exists before saving
    ensure_session_dir(session_id)
    
    # Convert messages to JSON-serializable format
    data = []
    for msg in messages:
        msg_data = {
            'role': msg['role'],
            'content': msg['content'],
            'timestamp': msg['timestamp'].isoformat() if hasattr(msg['timestamp'], 'isoformat') else msg['timestamp'],
        }
        if msg.get('details') is not None:
            msg_data['details'] = msg['details']
        data.append(msg_data)
    
    with open(chat_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
