"""Ask API schemas for RAG Document QA.

Defines Pydantic models for question-answering request/response validation
and API documentation generation.
"""

from typing import List
from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    """API request model for asking questions about uploaded documents.
    
    Clients send this JSON payload to query documents within a session.
    """
    session_id: str = Field(..., description="Session ID containing the documents to query")
    question: str = Field(..., description="Question to ask about the uploaded documents")


class SourceChunk(BaseModel):
    """API response model for retrieved source chunks.
    
    Represents a text chunk that was used as context for generating
    the answer, along with its relevance score.
    """
    chunk: str = Field(..., description="Text chunk used as context for the answer")
    score: float = Field(..., description="Relevance score (0.0 to 1.0) indicating similarity to question")
    metadata: dict = Field(default={}, description="Optional metadata about the source chunk")


class AskResponse(BaseModel):
    """API response model for question-answering endpoint.
    
    Returns the generated answer and the source chunks that were used
    as context, providing transparency into the RAG process.
    """
    answer: str = Field(..., description="Generated answer to the question")
    sources: List[SourceChunk] = Field(..., description="Source text chunks used to generate the answer")
    qa_engine: str = Field(..., description="Name of the QA engine used to generate the answer")
