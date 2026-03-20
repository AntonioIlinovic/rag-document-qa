"""Named Entity Recognition (NER) services.

This package provides entity extraction capabilities for highlighting
named entities in text responses.
"""

from typing import Optional
from .base import BaseNERExtractor
from .spacy_extractor import SpaCyExtractor
from ...config import Settings
import logging

logger = logging.getLogger(__name__)


def get_ner_extractor(settings: Settings) -> Optional[BaseNERExtractor]:
    """Create and return an NER extractor based on configuration.
    
    Args:
        settings: Application settings (contains spacy_model)
        
    Returns:
        NER extractor instance or None if model fails to load
    """
    try:
        # For now, only spaCy is supported
        extractor = SpaCyExtractor(model_name=settings.spacy_model)
        
        # Test if the model loads successfully
        if extractor.is_enabled():
            logger.info(f"NER extractor created with model: {settings.spacy_model}")
            return extractor
        else:
            error = extractor.get_load_error()
            logger.warning(f"NER model failed to load: {error}")
            return None
            
    except Exception as exc:
        logger.error(f"Error creating NER extractor: {exc}")
        return None


__all__ = ["BaseNERExtractor", "SpaCyExtractor", "get_ner_extractor"]
