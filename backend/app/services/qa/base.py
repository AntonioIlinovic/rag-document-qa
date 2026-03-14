from abc import ABC, abstractmethod
from typing import List


class BaseQAEngine(ABC):
    """Abstract base class for question answering engines."""
    
    @abstractmethod
    async def answer(self, question: str, context_chunks: List[str]) -> str:
        """
        Generate an answer to a question based on provided context chunks.
        
        Args:
            question: The question to answer
            context_chunks: List of relevant text chunks for context
            
        Returns:
            Generated answer as a string
        """
        pass
