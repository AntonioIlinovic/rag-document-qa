from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path
import yaml
from typing import List, Union
import os

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

    # NER Configuration
    ner_enabled: bool = True
    spacy_model: str = "en_core_web_sm"

    # File Upload Configuration (loaded from shared_config.yaml)
    supported_extensions: List[str] = []

    # Application Data Directory (runtime data storage, e.g. sessions, chromaDB, etc.)
    # Resolves relative to project root regardless of execution context
    app_data_dir: Union[str, Path] = "./app_data"

    model_config = SettingsConfigDict(
        env_file=str(Path(__file__).parent.parent / ".env"),
        env_file_encoding="utf-8"
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Resolve app_data_dir relative to project root
        self._resolve_app_data_dir()
        self._load_shared_config()

    def _resolve_app_data_dir(self):
        """Resolve app_data_dir relative to project root regardless of execution context."""
        # Get project root (go up from app/ -> backend/ -> project root)
        base_dir = Path(__file__).resolve().parent.parent
        
        # Always default to app_data in project root
        self.app_data_dir = base_dir / "app_data"
    
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