"""Upload endpoint for RAG Document QA.

Handles multipart file uploads, text extraction, and RAG pipeline ingestion.
Supports both new session creation and adding documents to existing sessions.
"""

import logging
from typing import Annotated, List, Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from fastapi.responses import JSONResponse

from app.config import Settings
from app.schemas.upload import DocumentInfo, UploadResponse
from app.services.extraction import get_extractor
from app.services.extraction.base import ExtractionError
from app.services.rag.pipeline import BaseRAGPipeline
from app.services.session.models import SessionData
from app.services.session import (
    add_document_to_session,
    create_session,
    get_session,
    get_pipeline_for_session
)
from .deps import get_settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/upload", tags=["upload"])


def validate_file(file: UploadFile, app_settings: Settings) -> None:
    """Validate uploaded file.
    
    Args:
        file: UploadFile object to validate
        app_settings: Application settings containing file upload configuration
        
    Raises:
        HTTPException: If file is invalid (unsupported type, too large, etc.)
    """
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="No filename provided"
        )
    
    # Check file extension
    file_ext = file.filename.lower().rsplit(".", 1)[-1]
    supported_extensions = app_settings.supported_extensions.split(",")
    if f".{file_ext}" not in supported_extensions:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Unsupported file type: {file_ext}. Supported types: {', '.join(supported_extensions)}"
        )
    
    # Check file size
    if file.size and file.size > app_settings.max_file_size:
        max_size_mb = app_settings.max_file_size // (1024*1024)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"File too large. Maximum size: {max_size_mb}MB"
        )


async def process_single_file(
    file: UploadFile,
    pipeline: BaseRAGPipeline,
    session_id: str,
    app_settings: Settings
) -> DocumentInfo:
    """Process a single uploaded file.
    
    Args:
        file: UploadFile to process
        pipeline: RAG pipeline for ingestion
        session_id: Session ID for storage
        app_settings: Application settings
        
    Returns:
        DocumentInfo with processing results
        
    Raises:
        HTTPException: If processing fails
    """
    try:
        # Validate file first
        validate_file(file, app_settings)
        
        # Read file bytes
        file_bytes = await file.read()
        
        # Extract text using appropriate extractor
        extractor = get_extractor(file.filename)
        extracted_text = extractor.extract(file_bytes, file.filename)
        
        if not extracted_text.strip():
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"No text could be extracted from {file.filename}"
            )
        
        # Ingest into RAG pipeline
        chunks_created = pipeline.ingest(
            text=extracted_text,
            metadata={"filename": file.filename, "session_id": session_id},
            chunk_size=app_settings.chunk_size,
            chunk_overlap=app_settings.chunk_overlap
        )
        
        # Store file and extracted text in session
        add_document_to_session(
            session_id=session_id,
            filename=file.filename,
            file_bytes=file_bytes,
            extracted_text=extracted_text,
            chunks_created=chunks_created
        )
        
        logger.info(f"Successfully processed {file.filename}: {chunks_created} chunks")
        
        return DocumentInfo(
            filename=file.filename,
            chunks=chunks_created,
            status="processed"
        )
        
    except ExtractionError as exc:
        logger.error(f"Extraction failed for {file.filename}: {exc}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Text extraction failed for {file.filename}: {str(exc)}"
        ) from exc
    except Exception as exc:
        logger.error(f"Processing failed for {file.filename}: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Processing failed for {file.filename}: {str(exc)}"
        ) from exc


@router.post("/", response_model=UploadResponse, status_code=status.HTTP_200_OK)
async def upload_documents(
    files: Annotated[List[UploadFile], File(description="One or more files to upload")],
    session_id: Annotated[Optional[str], Form(description="Optional session ID (creates new if not provided)")] = None,
    app_settings: Annotated[Settings, Depends(get_settings)] = None
) -> UploadResponse:
    """Upload and process documents for question answering.
    
    Accepts one or more files (PDF, PNG, JPG, TIFF, TXT, MD), extracts text,
    creates embeddings, and stores in ChromaDB. Either creates a new
    session or adds to an existing one.
    
    Args:
        files: List of files to upload and process
        session_id: Optional existing session ID
        app_settings: Application settings (injected)
        
    Returns:
        UploadResponse with session ID and document processing info
        
    Raises:
        HTTPException: For various processing errors
    """
    if not files:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="At least one file must be provided"
        )
    
    # Validate all files first
    for file in files:
        validate_file(file, app_settings)
    
    # Determine session ID and get pipeline
    if session_id:
        # Use existing session
        try:
            get_session(session_id)  # Validate session exists
            pipeline = get_pipeline_for_session(session_id)
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Session {session_id} not found"
            ) from exc
    else:
        # Create new session
        session_id = create_session()
        pipeline = get_pipeline_for_session(session_id)
    
    # Process each file
    processed_documents = []
    for file in files:
        document_info = await process_single_file(file, pipeline, session_id, app_settings)
        processed_documents.append(document_info)
    
    logger.info(f"Upload complete: session {session_id}, {len(processed_documents)} files processed")
    
    return UploadResponse(
        session_id=session_id,
        documents=processed_documents
    )
