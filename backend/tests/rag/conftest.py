"""Fixtures for RAG pipeline tests."""

import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch

# ---------------------------------------------------------------------------
# Data fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_text_content():
    """Sample product description text for RAG tests."""
    return """The SolarSolutions SC-100 is a high-performance, ultra-portable monocrystalline solar charger designed for outdoor enthusiasts, hikers, and emergency preparedness. Utilizing high-efficiency SunPower cells, the SC-100 provides reliable power for smartphones, tablets, and other USB-powered devices even in challenging lighting conditions.

Technical Specifications:
- Peak Power Output: 15 Watts
- Transformation Efficiency: 22% - 25%
- Output Ports: Dual USB-A (5V / 2.1A Max per port)
- Weight: 12.8 oz (360g)

Frequently Asked Questions:
How long does it take to charge a standard phone? Under ideal direct sunlight, the SC-100 can charge a typical 3000mAh smartphone from 0% to 100% in approximately 2 to 2.5 hours.

Is the SC-100 waterproof? The solar panels themselves and the outer fabric are IP65 water-resistant, meaning they can handle light rain and splashes. However, the USB output controller box is NOT waterproof.
"""


@pytest.fixture
def rag_test_file():
    """Path to the single RAG test document."""
    return Path(__file__).parent.parent.parent.parent / "dummy_docs" / "rag_test" / "example_document.txt"


@pytest.fixture
def sample_text_from_file(rag_test_file):
    """Text content of the RAG test document."""
    with open(rag_test_file, "r", encoding="utf-8") as f:
        return f.read()


# ---------------------------------------------------------------------------
# Infrastructure fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def temp_app_data_dir():
    """Temporary app_data directory, cleaned up after each test."""
    temp_dir = Path(tempfile.mkdtemp(prefix="test_app_data_"))
    (temp_dir / "chroma").mkdir(exist_ok=True)
    (temp_dir / "sessions").mkdir(exist_ok=True)

    yield temp_dir

    shutil.rmtree(temp_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Embedder fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_embedding_model():
    """Mock sentence-transformers model returning 384-dim embeddings."""
    EMBEDDING_DIM = 384

    mock = MagicMock()
    mock.encode.side_effect = lambda texts, **kwargs: [
        [float(i) / EMBEDDING_DIM] * EMBEDDING_DIM for i in range(len(texts))
    ]
    mock.get_sentence_embedding_dimension.return_value = EMBEDDING_DIM
    return mock


@pytest.fixture
def embedder(mock_embedding_model):
    """SentenceTransformerEmbedder with model loading patched out."""
    from app.services.rag.embedder import SentenceTransformerEmbedder

    with patch.object(SentenceTransformerEmbedder, "_load_model"):
        instance = SentenceTransformerEmbedder()

    instance.model = mock_embedding_model
    return instance


# ---------------------------------------------------------------------------
# Vector store fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_chroma_collection():
    """Mock ChromaDB collection returning 3 canned results."""
    mock = Mock()
    mock.add.return_value = None
    mock.count.return_value = 2
    mock.query.return_value = {
        "ids": [["doc1", "doc2", "doc3"]],
        "documents": [["Chunk 1 content", "Chunk 2 content", "Chunk 3 content"]],
        "metadatas": [[{"source": "doc1"}, {"source": "doc2"}, {"source": "doc3"}]],
        "distances": [[0.1, 0.2, 0.3]],
    }
    mock.get.return_value = {
        "ids": ["doc1", "doc2"],
        "documents": ["Content 1", "Content 2"],
        "metadatas": [{"source": "doc1"}, {"source": "doc2"}],
    }
    return mock


@pytest.fixture
def mock_chroma_client(mock_chroma_collection):
    """Mock ChromaDB PersistentClient wired to mock_chroma_collection."""
    mock = Mock()
    mock.get_or_create_collection.return_value = mock_chroma_collection
    mock.delete_collection.return_value = None
    return mock


@pytest.fixture
def store(mock_chroma_collection, mock_chroma_client):
    """ChromaDBStore with client initialization patched out."""
    from app.services.rag.store import ChromaDBStore

    with patch.object(ChromaDBStore, "_initialize_client"):
        instance = ChromaDBStore()

    instance._client = mock_chroma_client
    instance._collection = mock_chroma_collection
    return instance
