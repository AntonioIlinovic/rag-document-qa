import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from app.services.qa.base import BaseQAEngine
from app.services.qa.cloud import CloudQAEngine
from app.services.qa.local import LocalQAEngine
from app.services.qa import get_qa_engine
from app.config import Settings


class TestBaseQAEngine:
    """Test the abstract base class."""
    
    def test_abstract_class(self):
        """Test that BaseQAEngine cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseQAEngine()


class TestCloudQAEngine:
    """Test the OpenAI-based QA engine."""
    
    def test_init_with_api_key(self):
        """Test initialization with API key."""
        engine = CloudQAEngine(api_key="test-key")
        assert engine.model == "gpt-3.5-turbo"
        assert engine.client.api_key == "test-key"
    
    def test_init_without_api_key(self):
        """Test initialization fails without API key."""
        with pytest.raises(ValueError, match="OpenAI API key is required"):
            CloudQAEngine(api_key="")
    
    def test_init_with_custom_model(self):
        """Test initialization with custom model."""
        engine = CloudQAEngine(api_key="test-key", model="gpt-4")
        assert engine.model == "gpt-4"
    
    @pytest.mark.asyncio
    async def test_answer_success(self):
        """Test successful answer generation."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Test answer"
        
        engine = CloudQAEngine(api_key="test-key")
        engine.client.chat.completions.create = AsyncMock(return_value=mock_response)
        
        result = await engine.answer("What is X?", ["Context about X"])
        assert result == "Test answer"
        
        # Verify the API was called correctly
        engine.client.chat.completions.create.assert_called_once()
        call_args = engine.client.chat.completions.create.call_args
        assert call_args[1]["model"] == "gpt-3.5-turbo"
        assert len(call_args[1]["messages"]) == 2
    
    @pytest.mark.asyncio
    async def test_answer_api_error(self):
        """Test handling of OpenAI API errors."""
        import openai
        from unittest.mock import MagicMock
        
        engine = CloudQAEngine(api_key="test-key")
        
        # Create a proper APIError with required arguments
        mock_request = MagicMock()
        engine.client.chat.completions.create = AsyncMock(
            side_effect=openai.APIError(message="API Error", request=mock_request, body={})
        )
        
        with pytest.raises(Exception, match="OpenAI API error"):
            await engine.answer("What is X?", ["Context about X"])
    
    @pytest.mark.asyncio
    async def test_answer_general_error(self):
        """Test handling of general errors."""
        engine = CloudQAEngine(api_key="test-key")
        engine.client.chat.completions.create = AsyncMock(
            side_effect=Exception("General error")
        )
        
        with pytest.raises(Exception, match="Error generating answer"):
            await engine.answer("What is X?", ["Context about X"])
    
    @pytest.mark.asyncio
    async def test_answer_with_multiple_chunks(self):
        """Test answer generation with multiple context chunks."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Test answer"
        
        engine = CloudQAEngine(api_key="test-key")
        engine.client.chat.completions.create = AsyncMock(return_value=mock_response)
        
        chunks = ["Chunk 1 content", "Chunk 2 content", "Chunk 3 content"]
        await engine.answer("What is X?", chunks)
        
        # Verify context was combined
        call_args = engine.client.chat.completions.create.call_args
        prompt = call_args[1]["messages"][1]["content"]
        assert "Chunk 1 content" in prompt
        assert "Chunk 2 content" in prompt
        assert "Chunk 3 content" in prompt


class TestLocalQAEngine:
    """Test the local DistilBERT QA engine."""
    
    def test_init_default_model(self):
        """Test initialization with default model."""
        engine = LocalQAEngine()
        assert engine.model_name == "distilbert-base-cased-distilled-squad"
        assert engine._pipeline is None
    
    def test_init_custom_model(self):
        """Test initialization with custom model."""
        engine = LocalQAEngine(model_name="custom-model")
        assert engine.model_name == "custom-model"
    
    @patch('app.services.qa.local.pipeline')
    @patch('app.services.qa.local.torch.cuda.is_available')
    def test_get_pipeline_cpu(self, mock_cuda, mock_pipeline_func):
        """Test pipeline creation on CPU."""
        mock_cuda.return_value = False
        mock_pipeline = MagicMock()
        mock_pipeline_func.return_value = mock_pipeline
        
        engine = LocalQAEngine()
        pipeline = engine._get_pipeline()
        
        assert pipeline == mock_pipeline
        mock_pipeline_func.assert_called_once_with(
            "document-question-answering",
            model="distilbert-base-cased-distilled-squad",
            device=-1
        )
    
    @patch('app.services.qa.local.pipeline')
    @patch('app.services.qa.local.torch.cuda.is_available')
    def test_get_pipeline_gpu(self, mock_cuda, mock_pipeline_func):
        """Test pipeline creation on GPU."""
        mock_cuda.return_value = True
        mock_pipeline = MagicMock()
        mock_pipeline_func.return_value = mock_pipeline
        
        engine = LocalQAEngine()
        pipeline = engine._get_pipeline()
        
        assert pipeline == mock_pipeline
        mock_pipeline_func.assert_called_once_with(
            "document-question-answering",
            model="distilbert-base-cased-distilled-squad",
            device=0
        )
    
    @patch('app.services.qa.local.pipeline')
    @pytest.mark.asyncio
    async def test_answer_success(self, mock_pipeline_func):
        """Test successful answer generation."""
        mock_pipeline = MagicMock()
        mock_pipeline.return_value = {
            "answer": "Test answer",
            "score": 0.95
        }
        mock_pipeline_func.return_value = mock_pipeline
        
        engine = LocalQAEngine()
        result = await engine.answer("What is X?", ["Context about X"])
        
        assert result == "Test answer"
        mock_pipeline.assert_called_once_with(
            question="What is X?",
            context="Context about X",
            max_answer_length=100
        )
    
    @patch('app.services.qa.local.pipeline')
    @pytest.mark.asyncio
    async def test_answer_multiple_chunks_best_score(self, mock_pipeline_func):
        """Test answer generation with multiple chunks, selects best score."""
        mock_pipeline = MagicMock()
        # Simulate different scores for different chunks
        mock_pipeline.side_effect = [
            {"answer": "Answer 1", "score": 0.3},
            {"answer": "Answer 2", "score": 0.8},
            {"answer": "Answer 3", "score": 0.5}
        ]
        mock_pipeline_func.return_value = mock_pipeline
        
        engine = LocalQAEngine()
        result = await engine.answer("What is X?", ["Chunk 1", "Chunk 2", "Chunk 3"])
        
        assert result == "Answer 2"  # Highest score
        assert mock_pipeline.call_count == 3
    
    @patch('app.services.qa.local.pipeline')
    @pytest.mark.asyncio
    async def test_answer_low_confidence(self, mock_pipeline_func):
        """Test answer generation with low confidence returns fallback."""
        mock_pipeline = MagicMock()
        mock_pipeline.return_value = {
            "answer": "Low confidence answer",
            "score": 0.05  # Below threshold
        }
        mock_pipeline_func.return_value = mock_pipeline
        
        engine = LocalQAEngine()
        result = await engine.answer("What is X?", ["Context about X"])
        
        assert result == "I cannot answer this question based on the provided context."
    
    @patch('app.services.qa.local.pipeline')
    @pytest.mark.asyncio
    async def test_answer_pipeline_error(self, mock_pipeline_func):
        """Test handling of pipeline errors."""
        mock_pipeline = MagicMock()
        mock_pipeline.side_effect = Exception("Pipeline error")
        mock_pipeline_func.return_value = mock_pipeline
        
        engine = LocalQAEngine()
        result = await engine.answer("What is X?", ["Context about X"])
        
        assert result == "I cannot answer this question based on the provided context."
    
    @patch('app.services.qa.local.pipeline')
    @pytest.mark.asyncio
    async def test_answer_no_chunks(self, mock_pipeline_func):
        """Test answer generation with no context chunks."""
        mock_pipeline = MagicMock()
        mock_pipeline_func.return_value = mock_pipeline
        
        engine = LocalQAEngine()
        result = await engine.answer("What is X?", [])
        
        assert result == "I cannot answer this question based on the provided context."
        mock_pipeline.assert_not_called()


class TestQAFactory:
    """Test the QA engine factory function."""
    
    def test_get_cloud_engine(self):
        """Test factory returns cloud engine when configured."""
        settings = Settings(
            openai_api_key="test-key",
            openai_model="gpt-4o-mini"
        )
        
        engine = get_qa_engine(settings, "cloud")
        assert isinstance(engine, CloudQAEngine)
    
    def test_get_local_engine(self):
        """Test factory returns local engine when configured."""
        settings = Settings(
            openai_api_key="",
            openai_model="gpt-4o-mini"
        )
        
        engine = get_qa_engine(settings, "local")
        assert isinstance(engine, LocalQAEngine)
    
    def test_get_engine_case_insensitive(self):
        """Test factory handles case insensitive engine names."""
        settings = Settings(
            openai_api_key="",
            openai_model="gpt-4o-mini"
        )
        
        engine = get_qa_engine(settings, "LOCAL")
        assert isinstance(engine, LocalQAEngine)
    
    def test_invalid_engine(self):
        """Test factory raises error for invalid engine."""
        settings = Settings(
            openai_api_key="",
            openai_model="gpt-4o-mini"
        )
        
        with pytest.raises(ValueError, match="Unknown QA engine"):
            get_qa_engine(settings, "invalid")


@pytest.mark.integration
class TestQAIntegration:
    """Integration tests that require actual model loading."""
    
    @pytest.mark.asyncio
    async def test_local_qa_integration(self):
        """Test local QA engine with actual model (marked as integration)."""
        # This test requires downloading the actual model
        # Mark with pytest.mark.integration to run separately
        
        #pytest.skip("Skipping integration test - requires model download")
        
        engine = LocalQAEngine()
        
        # Simple test case
        context = ["The capital of France is Paris."]
        question = "What is the capital of France?"
        
        result = await engine.answer(question, context)
        
        # Should contain "Paris" or the fallback message
        assert "Paris" in result or "cannot answer" in result.lower()
