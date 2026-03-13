from pydantic_settings import BaseSettings
from pathlib import Path

class Settings(BaseSettings):

    # RAG Configuration
    chunk_size: int = 1000
    chunk_overlap: int = 200
    embedding_model: str = "all-MiniLM-L6-v2"

    # Application Data Directory (runtime data storage, e.g. sessions, chromaDB, etc.)
    app_data_dir: str = "app_data"

    class Config:
        env_file = str(Path(__file__).parent.parent.parent / ".env")

settings = Settings()