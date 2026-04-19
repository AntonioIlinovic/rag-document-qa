"""Abstract base class for NER extractors.

Defines the interface that all NER implementations must follow.
"""

from abc import ABC, abstractmethod
from typing import List

from app.schemas.ask import NamedEntity


class BaseNERExtractor(ABC):
    """Abstract base class for named entity recognition extractors."""
    
    @abstractmethod
    def extract_entities(self, text: str) -> List[NamedEntity]:
        """
        Extract named entities from the given text.
        
        Args:
            text: The input text to extract entities from
            
        Returns:
            List of NamedEntity objects
        """
        pass
    
    @abstractmethod
    def get_model_name(self) -> str:
        """
        Get the name of the NER model being used.
        
        Returns:
            Model name as a string
        """
        pass
    
    @abstractmethod
    def is_enabled(self) -> bool:
        """
        Check if the NER service is enabled and functional.
        
        Returns:
            True if NER is available, False otherwise
        """
        pass
