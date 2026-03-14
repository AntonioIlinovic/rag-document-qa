"""RAG pipeline orchestration service.

This module provides a unified pipeline that combines chunking, embedding,
and vector storage into a single interface for document ingestion and
question-answering retrieval.
"""

import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional

from app.services.rag.chunker import BaseChunker, get_chunker
from app.services.rag.embedder import BaseEmbedder, get_embedder
from app.services.rag.store import BaseVectorStore, get_store

logger = logging.getLogger(__name__)


class BaseRAGPipeline(ABC):
    """Abstract base class for the RAG pipeline.

    Defines the interface for document ingestion and retrieval,
    enabling the Strategy pattern for interchangeable pipeline implementations.
    """

    @abstractmethod
    def ingest(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
    ) -> int:
        """Chunk, embed, and store a document.

        Args:
            text: Raw document text to ingest
            metadata: Optional metadata to attach to every chunk (e.g. filename, source)
            chunk_size: Maximum size of each chunk in characters
            chunk_overlap: Number of characters to overlap between chunks

        Returns:
            Number of chunks stored

        Raises:
            ValueError: If text is invalid or chunking/embedding parameters are bad
            RuntimeError: If storage fails
        """
        pass

    @abstractmethod
    def query(
        self,
        question: str,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """Embed a question and retrieve the most relevant chunks.

        Args:
            question: The question to search for
            top_k: Number of top results to return

        Returns:
            List of result dicts with keys: chunk, metadata, score, id

        Raises:
            ValueError: If question is empty or top_k is invalid
            RuntimeError: If retrieval fails
        """
        pass

    @abstractmethod
    def count_documents(self) -> int:
        """Return the number of chunks currently stored.

        Returns:
            Number of stored chunks

        Raises:
            RuntimeError: If the count operation fails
        """
        pass

    @abstractmethod
    def clear(self) -> None:
        """Delete all stored documents from the vector store.

        Raises:
            RuntimeError: If deletion fails
        """
        pass


class RAGPipeline(BaseRAGPipeline):
    """Concrete RAG pipeline combining chunking, embedding, and vector storage.

    Orchestrates the three RAG services into a single cohesive interface.
    All dependencies are injected, making the pipeline fully testable and
    swappable via the Strategy pattern.

    Typical usage:
        pipeline = get_pipeline()
        pipeline.ingest(text, metadata={"source": "contract.pdf"})
        results = pipeline.query("What is the termination clause?")
    """

    def __init__(
        self,
        chunker: BaseChunker,
        embedder: BaseEmbedder,
        store: BaseVectorStore,
    ) -> None:
        """Initialize the pipeline with injected dependencies.

        Args:
            chunker: Text chunking implementation
            embedder: Text embedding implementation
            store: Vector store implementation
        """
        self._chunker = chunker
        self._embedder = embedder
        self._store = store

    @property
    def chunker(self) -> BaseChunker:
        return self._chunker

    @property
    def embedder(self) -> BaseEmbedder:
        return self._embedder

    @property
    def store(self) -> BaseVectorStore:
        return self._store

    def ingest(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
    ) -> int:
        """Chunk, embed, and store a document.

        Args:
            text: Raw document text to ingest
            metadata: Optional metadata to attach to every chunk
            chunk_size: Maximum size of each chunk in characters
            chunk_overlap: Number of characters to overlap between chunks

        Returns:
            Number of chunks stored (0 if text was empty)

        Raises:
            ValueError: If chunking or embedding parameters are invalid
            RuntimeError: If storage operation fails
        """
        if not text or not text.strip():
            logger.warning("ingest called with empty text — nothing stored")
            return 0

        logger.info(f"Ingesting document: chunk_size={chunk_size}, chunk_overlap={chunk_overlap}")

        chunks = self._chunker.chunk_text(text, chunk_size, chunk_overlap)

        if not chunks:
            logger.warning("Chunker returned no chunks — nothing stored")
            return 0

        logger.info(f"Created {len(chunks)} chunks, generating embeddings")
        embeddings = self._embedder.embed_texts(chunks)

        chunk_metadata = [dict(metadata) if metadata else None for _ in chunks]

        self._store.add_documents(chunks, embeddings, chunk_metadata)

        logger.info(f"Ingestion complete: {len(chunks)} chunks stored")
        return len(chunks)

    def query(
        self,
        question: str,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """Embed a question and retrieve the most relevant chunks.

        Args:
            question: The question to search for
            top_k: Number of top results to return

        Returns:
            List of result dicts with keys: chunk, metadata, score, id.
            Returns empty list if question is empty or top_k is 0.

        Raises:
            ValueError: If embedding fails
            RuntimeError: If retrieval fails
        """
        if not question or not question.strip():
            logger.warning("query called with empty question — returning empty results")
            return []

        if top_k <= 0:
            return []

        logger.info(f"Querying pipeline: top_k={top_k}")

        query_embedding = self._embedder.embed_texts([question])[0]
        results = self._store.search(query_embedding, top_k)

        logger.info(f"Query returned {len(results)} results")
        return results

    def count_documents(self) -> int:
        """Return the number of chunks currently stored.

        Returns:
            Number of stored chunks

        Raises:
            RuntimeError: If the count operation fails
        """
        return self._store.count_documents()

    def clear(self) -> None:
        """Delete all stored documents from the vector store.

        Raises:
            RuntimeError: If deletion fails
        """
        self._store.delete_collection()
        logger.info("Pipeline cleared — all documents deleted")


def get_pipeline(persist_directory: Optional[str] = None) -> BaseRAGPipeline:
    """Factory function to create a fully wired RAGPipeline instance.

    All components are created via their own factory functions, which
    respect environment variable configuration (EMBEDDING_MODEL, APP_DATA_DIR).

    Args:
        persist_directory: Optional directory for ChromaDB persistence.
                           Passed through to get_store().

    Returns:
        Configured BaseRAGPipeline implementation

    Note:
        This factory is the single entry point for pipeline creation.
        In tests, prefer constructing RAGPipeline directly with mocked
        dependencies rather than calling get_pipeline().
    """
    return RAGPipeline(
        chunker=get_chunker(),
        embedder=get_embedder(),
        store=get_store(persist_directory=persist_directory),
    )