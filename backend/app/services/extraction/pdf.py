import fitz
from .base import BaseExtractor, ExtractionError


class PyMuPDFExtractor(BaseExtractor):
    """PDF text extractor using PyMuPDF (fitz)."""

    def extract(self, file_bytes: bytes, filename: str) -> str:
        """Extract text content from a PDF file.

        Args:
            file_bytes: The raw bytes of the PDF file
            filename: The name of the PDF file (for logging/reference)

        Returns:
            The concatenated text content from all PDF pages

        Raises:
            ExtractionError: If the PDF cannot be opened or text extraction fails
        """
        try:
            with fitz.open(stream=file_bytes, filetype="pdf") as doc:
                return "".join(page.get_text() for page in doc)
        except ExtractionError:
            raise
        except Exception as e:
            raise ExtractionError(f"Failed to extract text from {filename}: {e}")