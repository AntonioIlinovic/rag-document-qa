"""Vector store service for RAG pipeline.

This module provides vector database functionality using ChromaDB to store,
retrieve, and manage document embeddings for similarity search and retrieval
in the RAG system.
"""

import os
import uuid
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from pathlib import Path
import logging

from app.config import settings

logger = logging.getLogger(__name__)


class BaseVectorStore(ABC):
    """Abstract base class for vector stores.

    Defines the interface for all vector store implementations,
    enabling the Strategy pattern for interchangeable vector databases.
    """

    @abstractmethod
    def add_documents(self, chunks: List[str], embeddings: List[List[float]], metadata: Optional[List[Dict[str, Any]]] = None) -> None:
        """Add documents with their embeddings to the vector store.

        Args:
            chunks: List of text chunks to store
            embeddings: List of embedding vectors for each chunk
            metadata: Optional list of metadata dictionaries for each chunk

        Raises:
            ValueError: If input validation fails
            RuntimeError: If storage operation fails
        """
        pass

    @abstractmethod
    def search(self, query_embedding: List[float], top_k: int) -> List[Dict[str, Any]]:
        """Search for similar documents using query embedding.

        Args:
            query_embedding: Embedding vector for the query
            top_k: Number of top results to return

        Returns:
            List of dicts with keys: chunk, metadata, score, id

        Raises:
            ValueError: If query embedding is invalid
            RuntimeError: If search operation fails
        """
        pass

    @abstractmethod
    def delete_collection(self) -> None:
        """Delete the entire collection and reset internal state.

        Raises:
            RuntimeError: If deletion fails
        """
        pass

    @abstractmethod
    def count_documents(self) -> int:
        """Count the number of documents in the collection.

        Returns:
            Number of documents stored

        Raises:
            RuntimeError: If the count operation fails
        """
        pass


class ChromaDBStore(BaseVectorStore):
    """ChromaDB implementation of vector store.

    Provides persistent vector storage using ChromaDB with built-in
    persistence and metadata filtering capabilities.

    ChromaDB PersistentClient handles persistence automatically — all writes
    are flushed to disk immediately, so no explicit persist/load calls are needed.
    """

    def __init__(self, collection_name: Optional[str] = None, persist_directory: Optional[str] = None):
        """Initialize the ChromaDB store.

        Args:
            collection_name: Name of the collection (auto-generated if not provided)
            persist_directory: Directory for persistence (uses APP_DATA_DIR/chroma if not provided)

        Raises:
            RuntimeError: If ChromaDB client fails to initialize
        """
        self.collection_name = collection_name or f"collection_{uuid.uuid4().hex[:8]}"
        self.persist_directory = persist_directory or str(Path(settings.app_data_dir) / "chroma")
        self._client = None
        self._collection = None
        self._initialize_client()

    def _initialize_client(self) -> None:
        """Initialize ChromaDB client and get/create collection.

        Raises:
            RuntimeError: If client initialization fails
        """
        try:
            import chromadb
            logger.info(f"Initializing ChromaDB client with persist_directory: {self.persist_directory}")

            Path(self.persist_directory).mkdir(parents=True, exist_ok=True)

            self._client = chromadb.PersistentClient(path=self.persist_directory)
            self._collection = self._client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}  # Use cosine similarity so distances are in [0, 1]
            )
            logger.info(f"ChromaDB initialized. Collection: {self.collection_name}, "
                        f"documents: {self._collection.count()}")
        except ImportError as e:
            raise RuntimeError(f"chromadb package not installed: {e}")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize ChromaDB client: {e}")

    @property
    def collection(self):
        """Get the ChromaDB collection instance."""
        return self._collection

    @collection.setter
    def collection(self, collection):
        """Set the collection instance (useful for testing with mocks)."""
        self._collection = collection

    def add_documents(self, chunks: List[str], embeddings: List[List[float]], metadata: Optional[List[Dict[str, Any]]] = None) -> None:
        """Add documents with their embeddings to the vector store.

        Args:
            chunks: List of text chunks to store
            embeddings: List of embedding vectors for each chunk
            metadata: Optional list of metadata dicts for each chunk

        Raises:
            ValueError: If input validation fails
            RuntimeError: If storage operation fails
        """
        if chunks is None:
            raise ValueError("Chunks cannot be None")
        if embeddings is None:
            raise ValueError("Embeddings cannot be None")
        if not isinstance(chunks, list):
            raise ValueError(f"Chunks must be a list, got {type(chunks)}")
        if not isinstance(embeddings, list):
            raise ValueError(f"Embeddings must be a list, got {type(embeddings)}")
        if len(chunks) != len(embeddings):
            raise ValueError(f"Number of chunks ({len(chunks)}) must match number of embeddings ({len(embeddings)})")
        if metadata is not None and len(metadata) != len(chunks):
            raise ValueError(f"Number of metadata entries ({len(metadata)}) must match number of chunks ({len(chunks)})")

        if len(chunks) == 0:
            logger.warning("add_documents called with empty list — nothing stored")
            return

        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            if not isinstance(chunk, str):
                raise ValueError(f"Chunk at index {i} must be a string, got {type(chunk)}")
            if not isinstance(embedding, list):
                raise ValueError(f"Embedding at index {i} must be a list, got {type(embedding)}")
            if len(embedding) == 0:
                raise ValueError(f"Embedding at index {i} cannot be empty")

        if metadata is None:
            metadata = [None for _ in chunks]

        ids = [f"{self.collection_name}_{uuid.uuid4().hex[:8]}_{i}" for i in range(len(chunks))]

        try:
            self._collection.add(
                ids=ids,
                documents=chunks,
                embeddings=embeddings,
                metadatas=metadata
            )
            logger.info(f"Added {len(chunks)} documents to collection '{self.collection_name}'")
        except Exception as e:
            raise RuntimeError(f"Failed to add documents to ChromaDB: {e}")

    def search(self, query_embedding: List[float], top_k: int) -> List[Dict[str, Any]]:
        """Search for similar documents using query embedding.

        Args:
            query_embedding: Embedding vector for the query
            top_k: Number of top results to return

        Returns:
            List of dicts with keys: chunk, metadata, score, id.
            Score is a cosine similarity in [0, 1] (higher = more similar).

        Raises:
            ValueError: If query embedding is invalid
            RuntimeError: If search operation fails
        """
        if query_embedding is None:
            raise ValueError("Query embedding cannot be None")
        if not isinstance(query_embedding, list):
            raise ValueError(f"Query embedding must be a list, got {type(query_embedding)}")
        if len(query_embedding) == 0:
            raise ValueError("Query embedding cannot be empty")
        if top_k <= 0:
            return []

        try:
            results = self._collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k
            )

            formatted_results = []
            docs = results.get("documents", [[]])[0]
            metadatas = results.get("metadatas", [[]])[0] or [{}] * len(docs)
            distances = results.get("distances", [[]])[0] or [1.0] * len(docs)
            ids = results.get("ids", [[]])[0] or [f"result_{i}" for i in range(len(docs))]

            for i, (doc, meta, distance, doc_id) in enumerate(zip(docs, metadatas, distances, ids)):
                formatted_results.append({
                    "chunk": doc,
                    "metadata": meta or {},
                    "score": 1.0 - distance,  # Cosine distance → cosine similarity
                    "id": doc_id
                })

            logger.info(f"Search returned {len(formatted_results)} results for top_k={top_k}")
            return formatted_results

        except Exception as e:
            raise RuntimeError(f"Failed to search ChromaDB collection: {e}")

    def delete_collection(self) -> None:
        """Delete the entire collection and reset internal collection reference.

        After deletion, the collection reference is set to None. Call
        _initialize_client() to recreate it if needed.

        Raises:
            RuntimeError: If deletion fails
        """
        try:
            self._client.delete_collection(name=self.collection_name)
            self._collection = None
            logger.info(f"Deleted collection '{self.collection_name}'")
        except Exception as e:
            raise RuntimeError(f"Failed to delete collection '{self.collection_name}': {e}")

    def count_documents(self) -> int:
        """Count the number of documents in the collection.

        Returns:
            Number of documents stored

        Raises:
            RuntimeError: If the count operation fails
        """
        try:
            return self._collection.count()
        except Exception as e:
            raise RuntimeError(f"Failed to count documents in collection '{self.collection_name}': {e}")


def get_store(persist_directory: Optional[str] = None, collection_name: Optional[str] = None) -> BaseVectorStore:
    """Factory function to create a vector store instance.

    Args:
        persist_directory: Optional directory for ChromaDB persistence.
                           Defaults to APP_DATA_DIR/chroma env var.
        collection_name: Optional collection name for ChromaDB.
                         Defaults to random UUID if not provided.

    Returns:
        Configured BaseVectorStore implementation

    Note:
        Additional providers (e.g. FAISS, Qdrant) can be added here in the
        future via a VECTOR_STORE_PROVIDER environment variable.
    """
    return ChromaDBStore(collection_name=collection_name, persist_directory=persist_directory)