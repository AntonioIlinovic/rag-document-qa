"""Tests for the RAG pipeline orchestration layer.

This module contains unit tests for the RAGPipeline, verifying that it
correctly orchestrates chunking, embedding, and vector storage — and
handles edge cases gracefully.
"""

import pytest
from unittest.mock import Mock, MagicMock, call, patch

from app.services.rag.pipeline import RAGPipeline, get_pipeline


# ---------------------------------------------------------------------------
# Ingest tests
# ---------------------------------------------------------------------------

def test_ingest_calls_all_three_services(pipeline, mock_chunker, mock_embedder, mock_store):
    """Test that ingest orchestrates chunk → embed → store in order."""
    text = "Some document text to ingest."

    result = pipeline.ingest(text)

    mock_chunker.chunk_text.assert_called_once_with(text, 512, 64)
    mock_embedder.embed_texts.assert_called_once_with(["Chunk one content", "Chunk two content"])
    mock_store.add_documents.assert_called_once()
    assert result == 2


def test_ingest_passes_custom_chunk_parameters(pipeline, mock_chunker):
    """Test that custom chunk_size and chunk_overlap are forwarded to the chunker."""
    pipeline.ingest("Some text", chunk_size=256, chunk_overlap=32)

    mock_chunker.chunk_text.assert_called_once_with("Some text", 256, 32)


def test_ingest_attaches_metadata_to_every_chunk(pipeline, mock_store):
    """Test that metadata is copied and attached to each chunk individually."""
    metadata = {"source": "contract.pdf", "page": 1}

    pipeline.ingest("Some text", metadata=metadata)

    call_kwargs = mock_store.add_documents.call_args[1] if mock_store.add_documents.call_args.kwargs else mock_store.add_documents.call_args[0]
    # Support both positional and keyword call styles
    args, kwargs = mock_store.add_documents.call_args
    stored_metadata = kwargs.get("metadata") or args[2]

    assert len(stored_metadata) == 2
    assert all(m == metadata for m in stored_metadata)


def test_ingest_metadata_copies_are_independent(pipeline, mock_store):
    """Test that each chunk gets its own metadata copy, not a shared reference."""
    metadata = {"source": "doc.pdf"}

    pipeline.ingest("Some text", metadata=metadata)

    args, kwargs = mock_store.add_documents.call_args
    stored_metadata = kwargs.get("metadata") or args[2]

    # Mutating original should not affect stored copies
    metadata["source"] = "mutated"
    assert all(m["source"] == "doc.pdf" for m in stored_metadata)


def test_ingest_uses_empty_metadata_when_none_provided(pipeline, mock_store):
    """Test that missing metadata defaults to None, not empty dicts."""
    pipeline.ingest("Some text")

    args, kwargs = mock_store.add_documents.call_args
    stored_metadata = kwargs.get("metadata") or args[2]

    assert all(m is None for m in stored_metadata)


def test_ingest_returns_chunk_count(pipeline):
    """Test that ingest returns the exact number of chunks created."""
    result = pipeline.ingest("Some text")

    assert result == 2


def test_ingest_empty_text_returns_zero(pipeline, mock_chunker, mock_embedder, mock_store):
    """Test that empty text short-circuits and returns 0 without calling services."""
    result = pipeline.ingest("")

    assert result == 0
    mock_chunker.chunk_text.assert_not_called()
    mock_embedder.embed_texts.assert_not_called()
    mock_store.add_documents.assert_not_called()


def test_ingest_whitespace_only_text_returns_zero(pipeline, mock_chunker, mock_store):
    """Test that whitespace-only text is treated as empty."""
    result = pipeline.ingest("   \n\t  ")

    assert result == 0
    mock_chunker.chunk_text.assert_not_called()
    mock_store.add_documents.assert_not_called()


def test_ingest_when_chunker_returns_empty_list(pipeline, mock_chunker, mock_embedder, mock_store):
    """Test that a chunker returning no chunks skips embedding and storage."""
    mock_chunker.chunk_text.return_value = []

    result = pipeline.ingest("Some text")

    assert result == 0
    mock_embedder.embed_texts.assert_not_called()
    mock_store.add_documents.assert_not_called()


def test_ingest_passes_correct_chunks_and_embeddings_to_store(pipeline, mock_store):
    """Test that store receives the exact chunks and their corresponding embeddings."""
    pipeline.ingest("Some text")

    args, kwargs = mock_store.add_documents.call_args
    chunks = kwargs.get("chunks") or args[0]
    embeddings = kwargs.get("embeddings") or args[1]

    assert chunks == ["Chunk one content", "Chunk two content"]
    assert len(embeddings) == 2
    assert all(isinstance(e, list) for e in embeddings)


# ---------------------------------------------------------------------------
# Query tests
# ---------------------------------------------------------------------------

def test_query_embeds_question_and_searches_store(pipeline, mock_embedder, mock_store):
    """Test that query embeds the question and passes the result to store.search."""
    results = pipeline.query("What is the product weight?", top_k=2)

    mock_embedder.embed_texts.assert_called_once_with(["What is the product weight?"])
    mock_store.search.assert_called_once()
    assert len(results) == 2


def test_query_passes_top_k_to_store(pipeline, mock_store):
    """Test that top_k is forwarded correctly to the vector store."""
    pipeline.query("Some question", top_k=7)

    _, kwargs = mock_store.search.call_args
    n = kwargs.get("top_k") or mock_store.search.call_args[0][1]
    assert n == 7


def test_query_result_shape(pipeline):
    """Test that query results contain the expected keys."""
    results = pipeline.query("Any question?", top_k=5)

    for result in results:
        assert "chunk" in result
        assert "metadata" in result
        assert "score" in result
        assert "id" in result


def test_query_empty_question_returns_empty_list(pipeline, mock_embedder, mock_store):
    """Test that an empty question short-circuits without calling services."""
    results = pipeline.query("")

    assert results == []
    mock_embedder.embed_texts.assert_not_called()
    mock_store.search.assert_not_called()


def test_query_whitespace_question_returns_empty_list(pipeline, mock_embedder, mock_store):
    """Test that a whitespace-only question is treated as empty."""
    results = pipeline.query("   ")

    assert results == []
    mock_embedder.embed_texts.assert_not_called()
    mock_store.search.assert_not_called()


def test_query_top_k_zero_returns_empty_list(pipeline, mock_embedder, mock_store):
    """Test that top_k=0 short-circuits and returns empty without calling services."""
    results = pipeline.query("Some question", top_k=0)

    assert results == []
    mock_embedder.embed_texts.assert_not_called()
    mock_store.search.assert_not_called()


def test_query_uses_only_first_embedding(pipeline, mock_embedder, mock_store):
    """Test that query extracts only the first embedding from embed_texts output."""
    # embed_texts returns a list; query should use index [0]
    pipeline.query("Test question", top_k=5)

    _, search_kwargs = mock_store.search.call_args
    query_emb = search_kwargs.get("query_embedding") or mock_store.search.call_args[0][0]

    # The fixture produces [0.0, 0.0, 0.0] for index 0
    assert query_emb == [0.0, 0.0, 0.0]


# ---------------------------------------------------------------------------
# Count and clear tests
# ---------------------------------------------------------------------------

def test_count_documents_delegates_to_store(pipeline, mock_store):
    """Test that count_documents returns the store's count directly."""
    count = pipeline.count_documents()

    assert count == 2
    mock_store.count_documents.assert_called_once()


def test_clear_delegates_to_store(pipeline, mock_store):
    """Test that clear calls delete_collection on the store."""
    pipeline.clear()

    mock_store.delete_collection.assert_called_once()


# ---------------------------------------------------------------------------
# Property access tests
# ---------------------------------------------------------------------------

def test_pipeline_exposes_injected_dependencies(pipeline, mock_chunker, mock_embedder, mock_store):
    """Test that the pipeline correctly exposes its injected components."""
    assert pipeline.chunker is mock_chunker
    assert pipeline.embedder is mock_embedder
    assert pipeline.store is mock_store


# ---------------------------------------------------------------------------
# Factory function test
# ---------------------------------------------------------------------------

def test_get_pipeline_returns_rag_pipeline_instance(temp_app_data_dir):
    """Test that the factory function returns a properly wired RAGPipeline."""
    from app.services.rag.pipeline import RAGPipeline
    from app.services.rag.chunker import BaseChunker
    from app.services.rag.embedder import BaseEmbedder
    from app.services.rag.store import BaseVectorStore
    from unittest.mock import patch

    with (
        patch("app.services.rag.pipeline.get_chunker") as mock_get_chunker,
        patch("app.services.rag.pipeline.get_embedder") as mock_get_embedder,
        patch("app.services.rag.pipeline.get_store") as mock_get_store,
    ):
        mock_get_chunker.return_value = Mock(spec=BaseChunker)
        mock_get_embedder.return_value = Mock(spec=BaseEmbedder)
        mock_get_store.return_value = Mock(spec=BaseVectorStore)

        pipeline = get_pipeline(persist_directory=str(temp_app_data_dir / "chroma"))

        assert isinstance(pipeline, RAGPipeline)
        mock_get_store.assert_called_once_with(
            persist_directory=str(temp_app_data_dir / "chroma"),
            collection_name=None
        )


# ---------------------------------------------------------------------------
# Integration test
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_pipeline_full_ingest_and_query(temp_app_data_dir, sample_text_content):
    """Integration test: full ingest → query cycle with real services."""
    pipeline = get_pipeline(persist_directory=str(temp_app_data_dir / "chroma"))

    chunk_count = pipeline.ingest(
        sample_text_content,
        metadata={"source": "test_document.txt"},
        chunk_size=300,
        chunk_overlap=50,
    )

    assert chunk_count > 0
    assert pipeline.count_documents() == chunk_count

    results = pipeline.query("What is the peak power output?", top_k=3)

    assert isinstance(results, list)
    assert len(results) > 0
    for result in results:
        assert "chunk" in result
        assert "score" in result
        assert 0.0 <= result["score"] <= 1.0
        assert result["metadata"].get("source") == "test_document.txt"


@pytest.mark.integration
def test_pipeline_clear_removes_all_documents(temp_app_data_dir, sample_text_content):
    """Integration test: clear removes all documents from the store."""
    pipeline = get_pipeline(persist_directory=str(temp_app_data_dir / "chroma"))

    pipeline.ingest(sample_text_content)
    assert pipeline.count_documents() > 0

    pipeline.clear()

    # After clearing, re-initialize to confirm persistence is gone
    fresh_pipeline = get_pipeline(persist_directory=str(temp_app_data_dir / "chroma"))
    assert fresh_pipeline.count_documents() == 0