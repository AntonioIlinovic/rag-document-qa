from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path
import yaml
from typing import List

class Settings(BaseSettings):
    """Application settings with Pydantic v2 configuration."""

    # RAG Configuration
    chunk_size: int = 1000
    chunk_overlap: int = 200
    embedding_model: str = "all-MiniLM-L6-v2"

    # QA Configuration
    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"
    qa_engine: str = "local"  # cloud or local

    # File Upload Configuration (loaded from shared_config.yaml)
    supported_extensions: List[str] = []

    # Application Data Directory (runtime data storage, e.g. sessions, chromaDB, etc.)
    # Default to ../app_data to store at project root level
    app_data_dir: str = "../app_data"

    model_config = SettingsConfigDict(
        env_file=str(Path(__file__).parent.parent / ".env"),
        env_file_encoding="utf-8"
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._load_shared_config()

    def _load_shared_config(self):
        """Load file upload configuration from shared_config.yaml."""
        try:
            config_path = Path(__file__).parent.parent.parent / "shared_config.yaml"
            with open(config_path, 'r') as f:
                shared_config = yaml.safe_load(f)
            
            file_config = shared_config.get('file_upload', {})
            self.supported_extensions = file_config.get('supported_extensions', [])
        except (FileNotFoundError, yaml.YAMLError, KeyError):
            # Fallback to defaults if shared config is not available
            self.supported_extensions = ["pdf", "png", "jpg", "jpeg", "tiff", "txt", "md"]

settings = Settings()