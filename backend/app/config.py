from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path

class Settings(BaseSettings):
    """Application settings with Pydantic v2 configuration."""

    # RAG Configuration
    chunk_size: int = 1000
    chunk_overlap: int = 200
    embedding_model: str = "all-MiniLM-L6-v2"

    # QA Configuration
    openai_api_key: str = ""
    qa_engine: str = "local"  # cloud or local

    # Application Data Directory (runtime data storage, e.g. sessions, chromaDB, etc.)
    # Default to ../app_data to store at project root level
    app_data_dir: str = "../app_data"

    model_config = SettingsConfigDict(
        env_file=str(Path(__file__).parent.parent.parent / ".env"),
        env_file_encoding="utf-8"
    )

settings = Settings()