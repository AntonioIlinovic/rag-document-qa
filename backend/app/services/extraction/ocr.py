import easyocr
import numpy as np
from PIL import Image
import io
from typing import List, Tuple
from .base import BaseExtractor, ExtractionError


class EasyOCRExtractor(BaseExtractor):
    """OCR text extractor using EasyOCR for image files.
    
    This extractor handles image files (PNG, JPG, JPEG, TIFF) using the EasyOCR library,
    which provides multi-language text recognition capabilities. It processes images
    in memory and extracts all detected text regions.
    """
    
    def __init__(self, languages: List[str] = None):
        """Initialize the OCR reader with specified languages.
        
        Args:
            languages: List of language codes for OCR (default: ['en'])
                     Common codes: 'en' (English), 'hr' (Croatian), etc.
        """
        self.languages = languages or ['en', 'hr']
        try:
            self.reader = easyocr.Reader(self.languages, gpu=True)
        except Exception as e:
            raise ExtractionError(f"Failed to initialize EasyOCR reader: {str(e)}")
    
    def extract(self, file_bytes: bytes, filename: str) -> str:
        """Extract text content from an image file using OCR.
        
        Converts the image bytes to a format EasyOCR can process, runs OCR detection,
        and concatenates all detected text regions into a single string.
        
        Args:
            file_bytes: The raw bytes of the image file
            filename: The name of the image file (for logging/reference)
            
        Returns:
            The concatenated text content from all detected text regions
            
        Raises:
            ExtractionError: If OCR processing fails or image cannot be processed
        """
        try:
            # Convert bytes to PIL Image
            image = self._bytes_to_image(file_bytes)
            
            # Convert PIL Image to numpy array for EasyOCR
            image_array = np.array(image)
            
            # Run OCR detection
            results = self.reader.readtext(image_array)
            
            # Extract and concatenate text from results
            extracted_text = self._process_ocr_results(results)
            
            return extracted_text
            
        except Exception as e:
            raise ExtractionError(f"Failed to extract text from {filename}: {str(e)}")
    
    def _bytes_to_image(self, file_bytes: bytes) -> Image.Image:
        """Convert bytes to PIL Image object.
        
        Args:
            file_bytes: Raw image bytes
            
        Returns:
            PIL Image object
            
        Raises:
            ExtractionError: If image cannot be opened or processed
        """
        try:
            image = Image.open(io.BytesIO(file_bytes))
            # Convert to RGB if necessary (EasyOCR works best with RGB)
            if image.mode != 'RGB':
                image = image.convert('RGB')
            return image
        except Exception as e:
            raise ExtractionError(f"Failed to process image bytes: {str(e)}")
    
    def _process_ocr_results(self, results: List[Tuple]) -> str:
        """Process OCR results and extract text.
        
        Args:
            results: List of OCR results from EasyOCR reader.readtext()
                    Each result is a tuple: (bbox, text, confidence)
                    
        Returns:
            Concatenated text from all detected regions
        """
        texts = []
        for bbox, text, confidence in results:
            # Only include text with reasonable confidence (optional filter)
            if confidence > 0.5:  # Confidence threshold can be adjusted
                texts.append(text.strip())
        
        return ' '.join(texts)