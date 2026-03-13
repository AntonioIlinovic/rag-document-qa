"""Tests for RAG pipeline integration.

This module contains integration tests for the complete RAG pipeline,
including end-to-end tests for document processing and retrieval.
"""

import pytest
from unittest.mock import Mock, patch
from app.services.rag import create_rag_pipeline


def test_rag_pipeline_document_processing(mock_embedding_model, mock_chroma_collection):
    """Test end-to-end document processing through RAG pipeline.
    
    Verifies that the RAG pipeline can process a complete document
    through chunking, embedding, and storage.
    
    Args:
        mock_embedding_model: Fixture providing mock embedding model
        mock_chroma_collection: Fixture providing mock ChromaDB collection
    """
    # Arrange
    pipeline = create_rag_pipeline()
    pipeline.embedder.model = mock_embedding_model  # Inject mock
    pipeline.store.collection = mock_chroma_collection  # Inject mock
    test_text = "This is a test document for the RAG pipeline. It contains multiple sentences and should be processed into chunks, embedded, and stored in the vector database for later retrieval."
    
    # Act
    result = pipeline.process_document(test_text, document_id="test_doc_1")
    
    # Assert
    assert result is not None
    assert "chunks_processed" in result
    assert "document_id" in result
    assert result["document_id"] == "test_doc_1"
    assert result["chunks_processed"] > 1  # Should create multiple chunks
    
    # Verify that components were called
    mock_chroma_collection.add.assert_called()


def test_rag_pipeline_search_functionality(mock_embedding_model, mock_chroma_collection):
    """Test RAG pipeline search functionality.
    
    Verifies that the RAG pipeline can search for relevant documents
    based on a query.
    
    Args:
        mock_embedding_model: Fixture providing mock embedding model
        mock_chroma_collection: Fixture providing mock ChromaDB collection
    """
    # Arrange
    pipeline = create_rag_pipeline()
    pipeline.embedder.model = mock_embedding_model  # Inject mock
    pipeline.store.collection = mock_chroma_collection  # Inject mock
    query = "What is the warranty information?"
    
    # Act
    results = pipeline.search(query, top_k=3)
    
    # Assert
    assert isinstance(results, list)
    assert len(results) == 3  # Should return top_k results
    
    # Verify result structure
    for result in results:
        assert "content" in result
        assert "score" in result
        assert "metadata" in result
        assert isinstance(result["content"], str)
        assert isinstance(result["score"], (int, float))
        assert isinstance(result["metadata"], dict)


def test_rag_pipeline_with_real_text(sample_text_from_file):
    """Test RAG pipeline with real document content.
    
    Verifies that the RAG pipeline can handle real document content
    from the test files.
    
    Args:
        sample_text_from_file: Fixture providing real text content
    """
    # Arrange
    pipeline = create_rag_pipeline()
    
    # Mock the components to avoid real model loading
    pipeline.embedder = Mock()
    pipeline.embedder.embed_text.return_value = [[0.1, 0.2, 0.3, 0.4, 0.5]]
    pipeline.embedder.embed_texts.return_value = [[0.1, 0.2, 0.3, 0.4, 0.5] for _ in range(10)]
    
    pipeline.store = Mock()
    pipeline.store.add_documents.return_value = None
    pipeline.store.search.return_value = [
        {"content": "Sample content 1", "score": 0.9, "metadata": {"source": "doc1"}},
        {"content": "Sample content 2", "score": 0.8, "metadata": {"source": "doc2"}}
    ]
    
    # Act
    result = pipeline.process_document(sample_text_from_file, document_id="real_doc")
    
    # Assert
    assert result is not None
    assert result["document_id"] == "real_doc"
    assert result["chunks_processed"] > 0
    
    # Verify components were called
    pipeline.embedder.embed_texts.assert_called()
    pipeline.store.add_documents.assert_called()


def test_rag_pipeline_search_with_real_content(sample_text_from_file):
    """Test RAG pipeline search with real document content.
    
    Verifies that the RAG pipeline can search within real document content.
    
    Args:
        sample_text_from_file: Fixture providing real text content
    """
    # Arrange
    pipeline = create_rag_pipeline()
    
    # Mock components
    pipeline.embedder = Mock()
    pipeline.embedder.embed_text.return_value = [0.1, 0.2, 0.3, 0.4, 0.5]
    
    pipeline.store = Mock()
    pipeline.store.search.return_value = [
        {
            "content": "The SC-100 provides reliable power for smartphones and tablets.",
            "score": 0.95,
            "metadata": {"source": "solar_doc", "chunk_id": 1}
        },
        {
            "content": "The solar panels are IP65 water-resistant.",
            "score": 0.87,
            "metadata": {"source": "solar_doc", "chunk_id": 2}
        }
    ]
    
    # Act
    results = pipeline.search("Is the solar charger waterproof?", top_k=2)
    
    # Assert
    assert len(results) == 2
    assert all("content" in result for result in results)
    assert all("score" in result for result in results)
    assert all("metadata" in result for result in results)
    
    # Verify search was called
    pipeline.store.search.assert_called_once()


def test_rag_pipeline_error_handling():
    """Test RAG pipeline error handling.
    
    Verifies that the RAG pipeline handles errors gracefully
    and provides meaningful error messages.
    """
    # Arrange
    pipeline = create_rag_pipeline()
    
    # Test with empty document
    result = pipeline.process_document("", document_id="empty_doc")
    assert result["chunks_processed"] == 0
    
    # Test with None document
    with pytest.raises((ValueError, TypeError)):
        pipeline.process_document(None, document_id="none_doc")
    
    # Test search with empty query
    with pytest.raises((ValueError, TypeError)):
        pipeline.search("", top_k=3)


def test_rag_pipeline_component_integration():
    """Test integration between RAG pipeline components.
    
    Verifies that all components work together correctly and that
    data flows properly through the pipeline.
    """
    # Arrange
    pipeline = create_rag_pipeline()
    
    # Mock with realistic data flow
    mock_chunks = ["Chunk 1 content", "Chunk 2 content", "Chunk 3 content"]
    mock_embeddings = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]]
    
    pipeline.chunker = Mock()
    pipeline.chunker.chunk_text.return_value = mock_chunks
    
    pipeline.embedder = Mock()
    pipeline.embedder.embed_texts.return_value = mock_embeddings
    
    pipeline.store = Mock()
    pipeline.store.add_documents.return_value = None
    pipeline.store.search.return_value = [
        {"content": chunk, "score": 0.9, "metadata": {"chunk_id": i}}
        for i, chunk in enumerate(mock_chunks)
    ]
    
    # Act
    result = pipeline.process_document("Test document", document_id="integration_test")
    search_results = pipeline.search("Test query", top_k=2)
    
    # Assert
    assert result["chunks_processed"] == 3
    assert len(search_results) == 2
    
    # Verify data flow
    pipeline.chunker.chunk_text.assert_called_once_with("Test document")
    pipeline.embedder.embed_texts.assert_called_once_with(mock_chunks)
    pipeline.store.add_documents.assert_called_once_with(
        documents=mock_chunks,
        embeddings=mock_embeddings,
        metadatas=[{"chunk_id": i, "document_id": "integration_test"} for i in range(3)]
    )


def test_rag_pipeline_with_different_chunk_sizes():
    """Test RAG pipeline with different chunking configurations.
    
    Verifies that the pipeline respects different chunking parameters.
    """
    # Arrange
    pipeline = create_rag_pipeline()
    
    # Mock components
    pipeline.chunker = Mock()
    pipeline.chunker.chunk_text.return_value = ["Chunk 1", "Chunk 2"]
    
    pipeline.embedder = Mock()
    pipeline.embedder.embed_texts.return_value = [[0.1, 0.2], [0.3, 0.4]]
    
    pipeline.store = Mock()
    pipeline.store.add_documents.return_value = None
    
    # Act
    pipeline.process_document("Test document", document_id="chunk_test", chunk_size=500, chunk_overlap=100)
    
    # Assert
    pipeline.chunker.chunk_text.assert_called_once_with("Test document", chunk_size=500, chunk_overlap=100)


def test_rag_pipeline_metadata_handling():
    """Test RAG pipeline metadata handling.
    
    Verifies that metadata is properly attached to chunks and
    preserved through the pipeline.
    """
    # Arrange
    pipeline = create_rag_pipeline()
    
    # Mock components
    pipeline.chunker = Mock()
    pipeline.chunker.chunk_text.return_value = ["Chunk 1", "Chunk 2"]
    
    pipeline.embedder = Mock()
    pipeline.embedder.embed_texts.return_value = [[0.1, 0.2], [0.3, 0.4]]
    
    pipeline.store = Mock()
    pipeline.store.add_documents.return_value = None
    
    # Act
    pipeline.process_document(
        "Test document", 
        document_id="metadata_test",
        metadata={"source": "test_file.txt", "type": "manual"}
    )
    
    # Assert
    pipeline.store.add_documents.assert_called_once()
    call_args = pipeline.store.add_documents.call_args
    metadatas = call_args[1]["metadatas"]
    
    # Verify metadata includes document info
    assert all("chunk_id" in meta for meta in metadatas)
    assert all("document_id" in meta for meta in metadatas)
    assert metadatas[0]["document_id"] == "metadata_test"
    assert metadatas[0]["source"] == "test_file.txt"
    assert metadatas[0]["type"] == "manual"
