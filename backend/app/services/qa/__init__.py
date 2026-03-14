# __init__.py
from .base import BaseQAEngine
from .cloud import CloudQAEngine
from .local import LocalQAEngine
from ...config import Settings

__all__ = ["BaseQAEngine", "CloudQAEngine", "LocalQAEngine", "get_qa_engine"]


def get_qa_engine(settings: Settings) -> BaseQAEngine:
    """Factory function to create QA engine based on configuration."""
    if settings.qa_engine.lower() == "cloud":
        return CloudQAEngine(api_key=settings.openai_api_key)
    elif settings.qa_engine.lower() == "local":
        return LocalQAEngine()
    else:
        raise ValueError(f"Unknown QA engine: {settings.qa_engine}. Use 'cloud' or 'local'.")