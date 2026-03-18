import asyncio
import logging
from typing import List

import torch
from transformers import pipeline

from .base import BaseQAEngine

logger = logging.getLogger(__name__)


class LocalQAEngine(BaseQAEngine):
    """Local DistilBERT-based question answering engine.
    
    Uses a Hugging Face extractive QA model to find answers within
    provided context chunks. Model is lazy-loaded on first use.
    Inference runs in a thread pool to avoid blocking the event loop.
    """

    NO_ANSWER_RESPONSE = "I cannot answer this question based on the provided context."
    MIN_CONFIDENCE_THRESHOLD = 0.3


    def __init__(self, model_name: str = "distilbert-base-cased-distilled-squad"):
        """Initialize the local QA engine.

        Args:
            model_name: Hugging Face model name to use for QA
        """
        self.model_name = model_name
        self._pipeline = None

    def _get_pipeline(self):
        """Lazy-load the QA pipeline on first use.

        Returns:
            Loaded Hugging Face QA pipeline

        Raises:
            RuntimeError: If the model fails to load
        """
        if self._pipeline is not None:
            return self._pipeline

        device = 0 if torch.cuda.is_available() else -1
        logger.info(f"Loading QA model '{self.model_name}' on device {device}")

        try:
            self._pipeline = pipeline(
                "question-answering",
                model=self.model_name,
                device=device,
            )
            logger.info("QA model loaded successfully")
        except Exception as e:
            raise RuntimeError(f"Failed to load QA model '{self.model_name}': {e}")

        return self._pipeline

    def _run_inference(self, question: str, context_chunks: List[str]) -> str:
        """Run extractive QA inference across all context chunks synchronously.

        Iterates over each chunk and keeps the answer with the highest
        confidence score. Intended to be called via asyncio.to_thread.

        Args:
            question: The question to answer
            context_chunks: List of text chunks to search for an answer

        Returns:
            Best answer found above the confidence threshold, or a fallback
            message if no confident answer was found
        """
        qa_pipeline = self._get_pipeline()
        best_answer = ""
        best_score = 0.0

        for chunk in context_chunks:
            if not chunk or not chunk.strip():
                continue
            try:
                result = qa_pipeline(
                    question=question,
                    context=chunk,
                    max_answer_length=100,
                )
                if result["score"] > best_score:
                    best_answer = result["answer"]
                    best_score = result["score"]
            except Exception as e:
                logger.warning(f"Inference failed on chunk, skipping: {e}")
                continue

        if best_answer and best_score > self.MIN_CONFIDENCE_THRESHOLD:
            logger.info(f"Answer found with confidence {best_score:.3f}")
            return best_answer

        logger.info("No answer found above confidence threshold")
        return self.NO_ANSWER_RESPONSE

    async def answer(self, question: str, context_chunks: List[str]) -> str:
        """Generate an answer using the local DistilBERT model.

        Runs inference in a thread pool to avoid blocking the event loop.

        Args:
            question: The question to answer
            context_chunks: List of relevant text chunks for context

        Returns:
            Extracted answer string, or a fallback message if no confident
            answer was found in the provided context
        """
        if not question or not question.strip():
            return self.NO_ANSWER_RESPONSE

        if not context_chunks:
            return self.NO_ANSWER_RESPONSE

        return await asyncio.to_thread(self._run_inference, question, context_chunks)
    
    def get_engine_name(self) -> str:
        """Get the name of the QA engine for display purposes.
        
        Returns:
            Engine name as a string
        """
        return f"Local ({self.model_name})"
    
    def get_model_name(self) -> str:
        """Get the name of the QA model used.
        
        Returns:
            Model name as a string
        """
        return self.model_name