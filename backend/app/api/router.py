"""API router aggregation for RAG Document QA.

Combines all route modules into a single router for inclusion
in the main FastAPI application.
"""

from fastapi import APIRouter

from .upload import router as upload_router
from .ask import router as ask_router
from .sessions import router as sessions_router

# Create main API router
api_router = APIRouter()

# Include route modules with proper prefixes
api_router.include_router(upload_router)
api_router.include_router(ask_router)
api_router.include_router(sessions_router)
