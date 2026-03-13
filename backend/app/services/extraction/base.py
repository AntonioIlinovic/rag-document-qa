from abc import ABC, abstractmethod
from typing import Any


class BaseExtractor(ABC):
    """Abstract base class for document text extractors.
 
    This class defines the interface that all document extractors must implement.
    It follows the Strategy pattern, allowing different extraction methods
    to be used interchangeably based on file type.
    """
 
    @abstractmethod
    def extract(self, file_bytes: bytes, filename: str) -> str:
        """Extract text content from a document file.
 
        Args:
            file_bytes: The raw bytes of the document file
            filename: The name of the file (including extension)
 
        Returns:
            The extracted text content as a string
 
        Raises:
            NotImplementedError: If not implemented by subclass
            ExtractionError: If text extraction fails
        """
        pass
