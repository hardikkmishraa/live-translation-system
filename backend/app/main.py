import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from starlette.exceptions import HTTPException as StarletteHTTPException
from fastapi.exceptions import RequestValidationError

from backend.app.config import (
    BASE_DIR,
    CORS_ORIGINS,
    cleanup_temp_dir
)
from backend.app.api.api_v1 import api_v1_router
from backend.app.api.endpoints import websocket as ws_endpoint
from backend.app.middleware.request_logging import PrivacyPreservingLoggingMiddleware
from backend.app.middleware.error_handler import (
    http_exception_handler,
    validation_exception_handler,
    generic_exception_handler
)
from backend.app.utils.logger import logger

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and graceful shutdown lifecycle."""
    logger.info("Initializing VoiceAffect translation backend...")
    cleanup_temp_dir()
    yield
    logger.info("Shutting down VoiceAffect translation backend...")
    cleanup_temp_dir()

app = FastAPI(
    title="VoiceAffect: Bilingual Voice Translation System With Emotion Awareness",
    description=(
        "Research implementation of Bilingual Voice Translation with Emotion Awareness "
        "(Ayush Singh Mehta, Hardik Mishra, Hardik Sishodia, Amit Sharma, Sharda University). "
        "Provides real-time Hindi <-> English speech translation with affective prosody preservation."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan
)

# 1. Privacy-Preserving Request Logging Middleware
app.add_middleware(PrivacyPreservingLoggingMiddleware)

# 2. CORS Middleware for Local Development
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS if CORS_ORIGINS != ["*"] else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 3. Exception Handlers
app.add_exception_handler(StarletteHTTPException, http_exception_handler)
app.add_exception_handler(RequestValidationError, validation_exception_handler)
app.add_exception_handler(Exception, generic_exception_handler)

# 4. API v1 Router Mount
app.include_router(api_v1_router, prefix="/api/v1")

# 5. Root WebSocket Endpoint (docs/api-contract.md)
app.include_router(ws_endpoint.router, prefix="/ws", tags=["Live Streaming"])

# 6. Backward Compatibility Aliases (legacy endpoints)
from backend.app.api.endpoints import health, translation
app.include_router(health.router, prefix="/api", include_in_schema=False)
app.include_router(translation.router, prefix="/api", include_in_schema=False)

# 7. Frontend Static Files & SPA Route
FRONTEND_DIR = BASE_DIR / "frontend"
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

    @app.get("/", include_in_schema=False)
    async def serve_index():
        return FileResponse(FRONTEND_DIR / "index.html")
