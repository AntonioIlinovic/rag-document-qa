"""Embedding service for RAG pipeline.

This module provides text embedding functionality using sentence-transformers
to generate vector representations of text chunks for similarity search
and retrieval in the RAG system.
"""

import os
from abc import ABC, abstractmethod
from typing import List
import logging

logger = logging.getLogger(__name__)


class BaseEmbedder(ABC):
    """Abstract base class for embedding models.

    Defines the interface for all embedding implementations,
    enabling the Strategy pattern for interchangeable embedding models.
    """

    @abstractmethod
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts.

        Args:
            texts: List of input texts to embed

        Returns:
            List of embedding vectors, each as list of floats

        Raises:
            ValueError: If texts list is invalid or model fails to embed
        """
        pass


class SentenceTransformerEmbedder(BaseEmbedder):
    """Sentence-transformers based embedding implementation.

    Supports any model available in the sentence-transformers library.
    The specific model is configured via the EMBEDDING_MODEL environment
    variable, defaulting to all-MiniLM-L6-v2.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """Initialize the embedder with specified model.

        Args:
            model_name: Name of the sentence-transformers model to use

        Raises:
            RuntimeError: If model fails to load
        """
        self.model_name = model_name
        self._model = None
        self._load_model()

    def _load_model(self) -> None:
        """Load the sentence-transformers model.

        Raises:
            RuntimeError: If model fails to load
        """
        try:
            from sentence_transformers import SentenceTransformer
            logger.info(f"Loading sentence-transformers model: {self.model_name}")
            self._model = SentenceTransformer(self.model_name)
            logger.info(f"Model loaded successfully. Embedding dimension: {self._model.get_sentence_embedding_dimension()}")
        except ImportError as e:
            raise RuntimeError(f"sentence-transformers package not installed: {e}")
        except Exception as e:
            raise RuntimeError(f"Failed to load embedding model {self.model_name}: {e}")

    @property
    def model(self):
        """Get the underlying model instance.

        Returns:
            The sentence-transformers model instance
        """
        return self._model

    @model.setter
    def model(self, new_model):
        """Set the model instance (useful for testing with mocks).

        Args:
            new_model: New model instance to use
        """
        self._model = new_model

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts.

        Args:
            texts: List of input texts to embed. Pass a single-element list
                   to embed one text: embed_texts(["my text"])[0]

        Returns:
            List of embedding vectors, each as list of floats

        Raises:
            ValueError: If texts list is invalid or model fails to embed
        """
        if texts is None:
            raise ValueError("Texts list cannot be None")

        if not isinstance(texts, list):
            raise ValueError(f"Texts must be a list, got {type(texts)}")

        if len(texts) == 0:
            return []

        for i, text in enumerate(texts):
            if text is None:
                raise ValueError(f"Text at index {i} cannot be None")
            if not isinstance(text, str):
                raise ValueError(f"Text at index {i} must be a string, got {type(text)}")
            if text.strip() == "":
                logger.warning(f"Text at index {i} is empty, embedding may not be meaningful")

        try:
            embeddings = self._model.encode(texts, batch_size=32, convert_to_numpy=True)
            return [emb.tolist() if hasattr(emb, "tolist") else list(emb) for emb in embeddings]
        except Exception as e:
            raise ValueError(f"Failed to embed texts: {e}")

    def get_embedding_dimension(self) -> int:
        """Get the dimension of the embedding vectors.

        Returns:
            The embedding dimension (e.g., 384 for MiniLM)
        """
        if self._model is None:
            raise RuntimeError("Model not loaded")
        return self._model.get_sentence_embedding_dimension()


def get_embedder() -> BaseEmbedder:
    """Factory function to create an embedder instance.

    The model is configured via the EMBEDDING_MODEL environment variable,
    defaulting to all-MiniLM-L6-v2 if not set.

    Returns:
        Configured BaseEmbedder implementation

    Note:
        This factory pattern allows for easy dependency injection and testing
        by providing a single point of embedder creation. Additional providers
        (e.g. OpenAI, Cohere) can be added here in the future via an
        EMBEDDING_PROVIDER environment variable.
    """
    model_name = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    return SentenceTransformerEmbedder(model_name=model_name)