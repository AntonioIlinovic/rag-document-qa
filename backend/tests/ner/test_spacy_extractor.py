"""Tests for SpaCy NER extractor."""

import pytest
from unittest.mock import patch, Mock
from app.services.ner.spacy_extractor import SpaCyExtractor
from app.services.ner.base import NamedEntity


class TestSpaCyExtractor:
    """Test cases for SpaCyExtractor class."""
    
    def test_init(self):
        """Test SpaCyExtractor initialization."""
        extractor = SpaCyExtractor("en_core_web_sm")
        assert extractor.model_name == "en_core_web_sm"
        assert extractor._nlp is None
        assert not extractor._model_loaded
        assert extractor._load_error is None
    
    def test_init_custom_model(self):
        """Test SpaCyExtractor initialization with custom model."""
        extractor = SpaCyExtractor("en_core_web_md")
        assert extractor.model_name == "en_core_web_md"
    
    def test_extract_entities_empty_text(self):
        """Test entity extraction with empty text."""
        extractor = SpaCyExtractor()
        entities = extractor.extract_entities("")
        assert entities == []
        
        entities = extractor.extract_entities("   ")
        assert entities == []
    
    def test_extract_entities_model_not_loaded(self):
        """Test entity extraction when model fails to load."""
        extractor = SpaCyExtractor()
        # Don't load the model
        entities = extractor.extract_entities("Some text")
        assert entities == []
    
    @patch('app.services.ner.spacy_extractor.spacy')
    def test_load_model_success(self, mock_spacy):
        """Test successful model loading."""
        # Mock spaCy components
        mock_nlp = Mock()
        mock_spacy.load.return_value = mock_nlp
        
        extractor = SpaCyExtractor()
        result = extractor._load_model()
        
        assert result is True
        assert extractor._model_loaded
        assert extractor._nlp == mock_nlp
        mock_nlp.disable_pipes.assert_called_once_with(["parser", "tagger", "lemmatizer"])
    
    @patch('app.services.ner.spacy_extractor.spacy')
    def test_load_model_download_and_load(self, mock_spacy):
        """Test model loading with download fallback."""
        # Mock spaCy components
        mock_nlp = Mock()
        mock_spacy.load.side_effect = [OSError("Model not found"), mock_nlp]
        
        extractor = SpaCyExtractor()
        result = extractor._load_model()
        
        assert result is True
        assert extractor._model_loaded
        assert extractor._nlp == mock_nlp
        assert mock_spacy.cli.download.called
        assert mock_nlp.disable_pipes.called
    
    @patch('app.services.ner.spacy_extractor.spacy')
    def test_load_model_failure(self, mock_spacy):
        """Test model loading failure."""
        mock_spacy.load.side_effect = Exception("Failed to load")
        mock_spacy.cli.download.side_effect = Exception("Failed to download")
        
        extractor = SpaCyExtractor()
        result = extractor._load_model()
        
        assert result is False
        assert not extractor._model_loaded
        assert extractor._load_error is not None
    
    @patch('app.services.ner.spacy_extractor.spacy')
    def test_extract_entities_success(self, mock_spacy):
        """Test successful entity extraction."""
        # Mock spaCy components
        mock_nlp = Mock()
        mock_doc = Mock()
        mock_ent1 = Mock()
        mock_ent1.text = "Apple"
        mock_ent1.label_ = "ORG"
        mock_ent1.start_char = 0
        mock_ent1.end_char = 5
        mock_ent2 = Mock()
        mock_ent2.text = "Cupertino"
        mock_ent2.label_ = "GPE"
        mock_ent2.start_char = 20
        mock_ent2.end_char = 30
        
        mock_doc.ents = [mock_ent1, mock_ent2]
        mock_nlp.return_value = mock_doc
        mock_spacy.load.return_value = mock_nlp
        
        extractor = SpaCyExtractor()
        extractor._load_model()
        
        entities = extractor.extract_entities("Apple is in Cupertino")
        
        assert len(entities) == 2
        assert entities[0].text == "Apple"
        assert entities[0].label == "ORG"
        assert entities[0].start == 0
        assert entities[0].end == 5
        assert entities[1].text == "Cupertino"
        assert entities[1].label == "GPE"
        assert entities[1].start == 20
        assert entities[1].end == 30
    
    @patch('app.services.ner.spacy_extractor.spacy')
    def test_extract_entities_error(self, mock_spacy):
        """Test entity extraction with error."""
        mock_nlp = Mock()
        mock_nlp.side_effect = Exception("Processing error")
        mock_spacy.load.return_value = mock_nlp
        
        extractor = SpaCyExtractor()
        extractor._load_model()
        
        entities = extractor.extract_entities("Some text")
        assert entities == []
    
    def test_get_model_name(self):
        """Test getting model name."""
        extractor = SpaCyExtractor("en_core_web_md")
        assert extractor.get_model_name() == "en_core_web_md"
    
    def test_is_enabled_not_loaded(self):
        """Test is_enabled when model not loaded."""
        extractor = SpaCyExtractor()
        assert not extractor.is_enabled()
    
    @patch('app.services.ner.spacy_extractor.spacy')
    def test_is_enabled_loaded(self, mock_spacy):
        """Test is_enabled when model is loaded."""
        mock_nlp = Mock()
        mock_spacy.load.return_value = mock_nlp
        
        extractor = SpaCyExtractor()
        extractor._load_model()
        
        assert extractor.is_enabled()
    
    def test_get_load_error_no_error(self):
        """Test get_load_error when no error occurred."""
        extractor = SpaCyExtractor()
        assert extractor.get_load_error() is None
    
    @patch('app.services.ner.spacy_extractor.spacy')
    def test_get_load_error_with_error(self, mock_spacy):
        """Test get_load_error when error occurred."""
        mock_spacy.load.side_effect = Exception("Failed to load")
        
        extractor = SpaCyExtractor()
        extractor._load_model()
        
        assert extractor.get_load_error() is not None
        assert "Failed to load" in extractor.get_load_error()


@pytest.mark.integration
class TestSpaCyExtractorIntegration:
    """Integration tests for SpaCyExtractor (requires real spaCy model)."""
    
    def test_real_extractor_initialization(self):
        """Test real extractor initialization."""
        extractor = SpaCyExtractor()
        assert extractor.model_name == "en_core_web_sm"
        assert not extractor._model_loaded
    
    @pytest.mark.slow
    def test_real_model_loading(self):
        """Test loading real spaCy model (marked as slow)."""
        extractor = SpaCyExtractor()
        
        # This might take time or fail if model not available
        result = extractor._load_model()
        
        if result:
            assert extractor.is_enabled()
            assert extractor.get_model_name() == "en_core_web_sm"
        else:
            # Model loading failed, check error is set
            assert extractor.get_load_error() is not None
    
    @pytest.mark.slow
    def test_real_entity_extraction(self, real_spacy_extractor, sample_text):
        """Test real entity extraction (marked as slow)."""
        if not real_spacy_extractor.is_enabled():
            pytest.skip("SpaCy model not available")
        
        entities = real_spacy_extractor.extract_entities(sample_text)
        
        # Should find some entities
        assert len(entities) > 0
        
        # Check entity structure
        for entity in entities:
            assert isinstance(entity.text, str)
            assert isinstance(entity.label, str)
            assert isinstance(entity.start, int)
            assert isinstance(entity.end, int)
            assert entity.start < entity.end
            assert entity.start >= 0
            assert entity.end <= len(sample_text)
