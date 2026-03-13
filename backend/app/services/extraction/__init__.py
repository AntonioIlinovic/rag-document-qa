"""Document extraction services.

This module provides a factory function for creating appropriate document extractors
based on file extensions. It follows the Factory pattern to decouple the calling
code from specific extractor implementations.
"""

from .base import BaseExtractor
from .pdf import PyMuPDFExtractor


def get_extractor(filename: str) -> BaseExtractor:
    """Factory function to return appropriate extractor based on file extension.
    
    Analyzes the file extension and returns the corresponding extractor instance.
    This allows the rest of the application to work with files without needing
    to know which specific extraction method to use.
    
    Args:
        filename: The name of the file including its extension
        
    Returns:
        An instance of the appropriate BaseExtractor subclass
        
    Raises:
        ValueError: If the file extension is not supported
        
    Examples:
        >>> extractor = get_extractor("document.pdf")
        >>> isinstance(extractor, PyMuPDFExtractor)
        True
        
        >>> extractor = get_extractor("image.png")
        ValueError: Unsupported file type: image.png
    """
    if filename.lower().endswith('.pdf'):
        return PyMuPDFExtractor()
    # Later: add OCR for images
    # elif filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff')):
    #     return EasyOCRExtractor()
    else:
        raise ValueError(f"Unsupported file type: {filename}")