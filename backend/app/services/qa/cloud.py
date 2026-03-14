import openai
from typing import List
from .base import BaseQAEngine


class CloudQAEngine(BaseQAEngine):
    """OpenAI-based question answering engine."""
    
    def __init__(self, api_key: str, model: str = "gpt-3.5-turbo"):
        """
        Initialize the OpenAI QA engine.
        
        Args:
            api_key: OpenAI API key
            model: OpenAI model to use (default: gpt-3.5-turbo)
        """
        if not api_key:
            raise ValueError("OpenAI API key is required for CloudQAEngine")
        
        self.client = openai.AsyncOpenAI(api_key=api_key)
        self.model = model
    
    async def answer(self, question: str, context_chunks: List[str]) -> str:
        """
        Generate an answer using OpenAI's API.
        
        Args:
            question: The question to answer
            context_chunks: List of relevant text chunks for context
            
        Returns:
            Generated answer as a string
        """
        # Combine context chunks into a single context
        context = "\n\n".join(context_chunks)
        
        # Create the prompt
        prompt = f"""Based on the following context, please answer the question. 
If the answer cannot be found in the context, say "I cannot answer this question based on the provided context."

Context:
{context}

Question: {question}

Answer:"""
        
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that answers questions based on provided context."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500,
                temperature=0.1
            )
            
            return response.choices[0].message.content.strip()
            
        except openai.APIError as e:
            raise Exception(f"OpenAI API error: {e}")
        except Exception as e:
            raise Exception(f"Error generating answer: {e}")
