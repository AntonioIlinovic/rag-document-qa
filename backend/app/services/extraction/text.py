"""Text file extractor for plain text documents.

This module provides a simple extractor for text-based files like .txt and .md
that just reads the file content as UTF-8 text.
"""

from .base import BaseExtractor, ExtractionError


class TextExtractor(BaseExtractor):
    """Text file extractor for plain text documents.
    
    Handles .txt, .md, and other text-based files by simply reading
    the file content as UTF-8 text.
    """

    def extract(self, file_bytes: bytes, filename: str) -> str:
        """Extract text content from a text file.

        Args:
            file_bytes: The raw bytes of the text file
            filename: The name of the text file (for logging/reference)

        Returns:
            The text content as a string

        Raises:
            ExtractionError: If the text cannot be decoded as UTF-8
        """
        try:
            # Try to decode as UTF-8 first
            return file_bytes.decode('utf-8')
        except UnicodeDecodeError:
            try:
                # Fallback to latin-1 if UTF-8 fails
                return file_bytes.decode('latin-1')
            except UnicodeDecodeError as e:
                raise ExtractionError(f"Failed to decode text file {filename}: {e}")
        except Exception as e:
            raise ExtractionError(f"Failed to extract text from {filename}: {e}")
