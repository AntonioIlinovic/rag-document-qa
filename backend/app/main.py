# app/main.py
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.api.router import api_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup logic here
    # Ensure app_data directory exists
    app_data_path = Path(settings.app_data_dir)
    app_data_path.mkdir(parents=True, exist_ok=True)
    
    # Ensure sessions directory exists
    sessions_path = app_data_path / "sessions"
    sessions_path.mkdir(parents=True, exist_ok=True)
    
    yield
    # Shutdown logic here


def create_app() -> FastAPI:
    app = FastAPI(
        title="RAG Document QA API",
        version="0.1.0",
        lifespan=lifespan
    )
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure appropriately for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Include API router
    app.include_router(api_router)
    
    # Health check endpoint
    @app.get("/health/")
    async def health_check():
        return {"status": "ok"}
    
    return app

# Create the app instance
app = create_app()