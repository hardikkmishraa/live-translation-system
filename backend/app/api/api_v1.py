from fastapi import APIRouter
from backend.app.api.endpoints import health, models, capabilities, translation, websocket, evaluation, research

api_v1_router = APIRouter()

api_v1_router.include_router(health.router, tags=["Health"])
api_v1_router.include_router(models.router, prefix="/models", tags=["Models"])
api_v1_router.include_router(capabilities.router, tags=["Capabilities"])
api_v1_router.include_router(translation.router, tags=["Translation"])
api_v1_router.include_router(evaluation.router, tags=["Evaluation"])
api_v1_router.include_router(research.router, tags=["Research"])
api_v1_router.include_router(websocket.router, prefix="/ws", tags=["Live Streaming"])
