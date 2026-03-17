"""Ask endpoint for RAG Document QA.

Handles question-answering requests using the RAG pipeline
to retrieve relevant document chunks and generate answers.
"""

import logging
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status

from app.schemas.ask import AskRequest, AskResponse, SourceChunk
from app.services.qa.base import BaseQAEngine
from app.services.rag.pipeline import BaseRAGPipeline
from .deps import get_pipeline_from_request, get_qa_engine

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/ask", tags=["ask"])


@router.post("/", response_model=AskResponse, status_code=status.HTTP_200_OK)
async def ask_question(
    request: AskRequest,
    pipeline: Annotated[BaseRAGPipeline, Depends(get_pipeline_from_request)],
    qa_engine: Annotated[BaseQAEngine, Depends(get_qa_engine)]
) -> AskResponse:
    """Ask a question about uploaded documents.
    
    Uses the RAG pipeline to retrieve relevant document chunks and
    the QA engine to generate an answer based on those chunks.
    
    Args:
        request: AskRequest with session_id and question
        pipeline: RAG pipeline for the session (injected)
        qa_engine: QA engine for answer generation (injected)
        
    Returns:
        AskResponse with answer and source chunks
        
    Raises:
        HTTPException: For various processing errors
    """
    if not request.question.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Question cannot be empty"
        )
    
    try:
        # Check if pipeline has any documents
        doc_count = pipeline.count_documents()
        logger.info(f"Document count in session {request.session_id}: {doc_count}")
        
        if doc_count == 0:
            logger.warning(f"No documents found in session {request.session_id}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No documents found in session. Upload documents first."
            )
        
        logger.info(f"Querying session {request.session_id}: '{request.question}'")
        
        # Retrieve relevant chunks using RAG pipeline
        search_results = pipeline.query(request.question, top_k=5)
        
        if not search_results:
            logger.warning(f"No relevant chunks found for question: {request.question}")
            return AskResponse(
                answer="I couldn't find relevant information in the uploaded documents to answer your question.",
                sources=[]
            )
        
        # Extract chunk texts for QA engine
        context_chunks = [result["chunk"] for result in search_results]
        
        logger.info(f"Retrieved {len(context_chunks)} relevant chunks")
        
        # Generate answer using QA engine
        answer = await qa_engine.answer(request.question, context_chunks)
        
        if not answer.strip():
            logger.warning("QA engine returned empty answer")
            answer = "I couldn't generate a meaningful answer based on the provided documents."
        
        # Convert search results to source chunks for response
        sources = [
            SourceChunk(
                chunk=result["chunk"],
                score=float(result.get("score", 0.0)),
                metadata=result.get("metadata", {})
            )
            for result in search_results
        ]
        
        logger.info(f"Generated answer with {len(sources)} sources")
        
        return AskResponse(
            answer=answer,
            sources=sources,
            qa_engine=qa_engine.get_engine_name()
        )
        
    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as exc:
        logger.error(f"Error processing question: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing question: {str(exc)}"
        ) from exc
