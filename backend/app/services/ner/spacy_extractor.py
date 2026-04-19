"""SpaCy implementation of NER extractor.

Provides named entity recognition using spaCy's pre-trained models.
"""

import logging
from typing import List, Optional

from .base import BaseNERExtractor
from app.schemas.ask import NamedEntity

logger = logging.getLogger(__name__)


class SpaCyExtractor(BaseNERExtractor):
    """SpaCy-based named entity recognition extractor."""
    
    def __init__(self, model_name: str = "en_core_web_sm"):
        """Initialize the SpaCy NER extractor.
        
        Args:
            model_name: Name of the spaCy model to use
        """
        self.model_name = model_name
        self._nlp = None
        self._model_loaded = False
        self._load_error = None
    
    def _load_model(self) -> bool:
        """Load the spaCy model lazily.
        
        Returns:
            True if model loaded successfully, False otherwise
        """
        if self._model_loaded:
            return True
        
        try:
            import spacy
            logger.info(f"Loading spaCy model: {self.model_name}")
            
            # Try to load the model, download if not available
            try:
                self._nlp = spacy.load(self.model_name)
            except OSError:
                logger.info(f"Model {self.model_name} not found, downloading...")
                spacy.cli.download(self.model_name)
                self._nlp = spacy.load(self.model_name)
            
            # Disable unnecessary components for performance
            self._nlp.disable_pipes(["parser", "tagger", "lemmatizer"])
            
            self._model_loaded = True
            logger.info("SpaCy model loaded successfully")
            return True
            
        except Exception as exc:
            self._load_error = str(exc)
            logger.error(f"Failed to load spaCy model: {exc}")
            return False
    
    def extract_entities(self, text: str) -> List[NamedEntity]:
        """Extract named entities from the given text."""
        if not text or not text.strip():
            return []
        
        if not self._load_model():
            logger.warning("NER model not loaded, returning empty entities")
            return []
        
        try:
            doc = self._nlp(text)
            entities = []
            
            for ent in doc.ents:
                # Convert spaCy entity to our format
                entity = NamedEntity(
                    text=ent.text,
                    label=ent.label_,
                    start=ent.start_char,
                    end=ent.end_char,
                    confidence=1.0  # spaCy doesn't provide confidence scores by default
                )
                entities.append(entity)
            
            # Sort entities by start position for consistent ordering
            entities.sort(key=lambda x: x.start)
            
            logger.debug(f"Extracted {len(entities)} entities from text")
            return entities
            
        except Exception as exc:
            logger.error(f"Error extracting entities: {exc}")
            return []
    
    def get_model_name(self) -> str:
        """Get the name of the NER model being used."""
        return self.model_name
    
    def is_enabled(self) -> bool:
        """Check if the NER service is enabled and functional."""
        return self._load_model()
    
    def get_load_error(self) -> Optional[str]:
        """Get the error message if model loading failed."""
        return self._load_error
