"""Text chunking service for RAG pipeline.

This module provides text splitting functionality using LangChain's
RecursiveCharacterTextSplitter to create overlapping chunks suitable
for embedding and retrieval.
"""

from abc import ABC, abstractmethod
from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter


class BaseChunker(ABC):
    """Abstract base class for text chunking strategies.
    
    Defines the interface for all text chunking implementations,
    enabling the Strategy pattern for interchangeable chunking algorithms.
    """
    
    @abstractmethod
    def chunk_text(self, text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
        """Split text into overlapping chunks.
        
        Args:
            text: The input text to chunk
            chunk_size: Maximum size of each chunk in characters
            chunk_overlap: Number of characters to overlap between chunks
            
        Returns:
            List of text chunks
            
        Raises:
            ValueError: If chunk_size or chunk_overlap are invalid
        """
        pass


class LangChainChunker(BaseChunker):
    """LangChain-based text chunking implementation.
    
    Uses LangChain's RecursiveCharacterTextSplitter to intelligently
    split text into overlapping chunks while preserving semantic
    coherence and respecting word boundaries.
    """
    
    def chunk_text(self, text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
        """Split text into overlapping chunks.
        
        Args:
            text: The input text to chunk
            chunk_size: Maximum size of each chunk in characters
            chunk_overlap: Number of characters to overlap between chunks
            
        Returns:
            List of text chunks
            
        Raises:
            ValueError: If chunk_size or chunk_overlap are invalid
        """
        if not text or not text.strip():
            return []
        
        if chunk_size <= 0:
            raise ValueError("chunk_size must be greater than 0")
        
        if chunk_overlap < 0:
            raise ValueError("chunk_overlap cannot be negative")
        
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")
        
        # Create a new splitter with the provided parameters
        splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", ". ", " ", ""],
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            keep_separator=False,
            strip_whitespace=True
        )
        
        # Split the text
        chunks = splitter.split_text(text)
        
        # Filter out any empty chunks that might result from splitting
        return [chunk.strip() for chunk in chunks if chunk.strip()]


def get_chunker() -> BaseChunker:
    """Factory function to create a chunker instance.
    
    Returns:
        Configured BaseChunker implementation (LangChainChunker)
        
    Note:
        This factory pattern allows for easy dependency injection
        and testing by providing a single point of chunker creation.
        Enables Strategy pattern for interchangeable chunking algorithms.
    """
    return LangChainChunker()