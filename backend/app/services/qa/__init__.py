# __init__.py
from .base import BaseQAEngine
from .cloud import CloudQAEngine
from .local import LocalQAEngine
from ...config import Settings

__all__ = ["BaseQAEngine", "CloudQAEngine", "LocalQAEngine", "get_qa_engine"]


def get_qa_engine(settings: Settings, qa_engine_type: str) -> BaseQAEngine:
    """Factory function to create QA engine based on configuration.
    
    Args:
        settings: Application settings (contains api_key and model)
        qa_engine_type: QA engine type ('cloud' or 'local') from user selection
    
    Returns:
        QA engine instance
    """
    if qa_engine_type.lower() == "cloud":
        return CloudQAEngine(api_key=settings.openai_api_key, model=settings.openai_model)
    elif qa_engine_type.lower() == "local":
        return LocalQAEngine()
    else:
        raise ValueError(f"Unknown QA engine: {qa_engine_type}. Use 'cloud' or 'local'.")