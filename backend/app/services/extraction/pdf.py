import fitz  # PyMuPDF
from .base import BaseExtractor


class PyMuPDFExtractor(BaseExtractor):
    """PDF text extractor using PyMuPDF (fitz).
    
    This extractor handles PDF files using the PyMuPDF library, which provides
    fast and reliable text extraction from PDF documents.
    """
    
    def extract(self, file_bytes: bytes, filename: str) -> str:
        """Extract text content from a PDF file.
        
        Opens the PDF from memory, iterates through all pages, and concatenates
        the extracted text. The PDF is properly closed to free resources.
        
        Args:
            file_bytes: The raw bytes of the PDF file
            filename: The name of the PDF file (for logging/reference)
            
        Returns:
            The concatenated text content from all PDF pages
            
        Raises:
            fitz.FitzError: If the PDF cannot be opened or is corrupted
            Exception: For other PDF processing errors
        """
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        text = ""
        for page in doc:
            text += page.get_text()
        doc.close()
        return text