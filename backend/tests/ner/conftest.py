"""Test fixtures for NER tests."""

import pytest
from unittest.mock import Mock
from app.services.ner.spacy_extractor import SpaCyExtractor
from app.schemas.ask import NamedEntity


@pytest.fixture
def sample_entities():
    """Sample named entities for testing."""
    return [
        NamedEntity(
            text="Apple",
            label="ORG",
            start=0,
            end=5,
            confidence=0.95
        ),
        NamedEntity(
            text="$3 billion",
            label="MONEY",
            start=20,
            end=30,
            confidence=0.88
        ),
        NamedEntity(
            text="Cupertino",
            label="GPE",
            start=51,
            end=60,
            confidence=0.89
        )
    ]


@pytest.fixture
def sample_text():
    """Sample text containing named entities."""
    return "Apple announced a $3 billion deal in Cupertino last Monday."


@pytest.fixture
def mock_spacy_extractor():
    """Mock spaCy extractor for testing."""
    extractor = Mock(spec=SpaCyExtractor)
    extractor.extract_entities.return_value = []
    extractor.highlight_entities.return_value = ""
    extractor.get_model_name.return_value = "en_core_web_sm"
    extractor.is_enabled.return_value = True
    return extractor


@pytest.fixture
def real_spacy_extractor():
    """Real spaCy extractor for integration tests."""
    return SpaCyExtractor(model_name="en_core_web_sm")
