"""Ask API schemas for RAG Document QA.

Defines Pydantic models for question-answering request/response validation
and API documentation generation.
"""

from typing import List, Optional
from pydantic import BaseModel, Field


class NamedEntity(BaseModel):
    """Represents a named entity extracted from the answer."""
    text: str = Field(..., description="The entity text as it appears in the answer")
    label: str = Field(..., description="Entity type (PERSON, ORG, GPE, etc.)")
    start: int = Field(..., description="Character start position in the answer text")
    end: int = Field(..., description="Character end position in the answer text")
    confidence: float = Field(default=0.0, description="Confidence score for the entity detection")


class AskRequest(BaseModel):
    """API request model for question-answering endpoint."""
    
    session_id: str = Field(..., description="Session ID for context")
    question: str = Field(..., description="Question to ask about the documents")
    qa_engine: str = Field(default="cloud", description="QA engine to use: 'cloud' or 'local'")
    ner_enabled: bool = Field(default=True, description="Whether to enable NER processing")


class SourceChunk(BaseModel):
    """API response model for retrieved source chunks.
    
    Represents a text chunk that was used as context for generating
    the answer, along with its relevance score.
    """
    chunk: str = Field(..., description="Text chunk used as context for the answer")
    score: float = Field(..., description="Semantic similarity between the question embedding and this chunk's embedding (cosine similarity, 0.0–1.0)")
    metadata: dict = Field(default={}, description="Optional metadata about the source chunk")


class AskResponse(BaseModel):
    """API response model for question-answering endpoint.
    
    Returns the generated answer, source chunks, and optionally
    named entities extracted from the answer.
    """
    answer: str = Field(..., description="Generated answer to the question")
    entities: List[NamedEntity] = Field(default=[], description="Named entities extracted from the answer")
    sources: List[SourceChunk] = Field(..., description="Source text chunks used to generate the answer")
    qa_engine: str = Field(..., description="Name of the QA engine used to generate the answer")
    qa_model: str = Field(..., description="Name of the QA model used to generate the answer")
    embedding_model: str = Field(..., description="Name of the embedding model used for retrieval")
