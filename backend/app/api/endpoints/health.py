import time
from fastapi import APIRouter
from backend.app.config import DEVICE
from backend.app.core.ffmpeg_utils import FFMPEG_EXE
from backend.app.schemas.health import HealthResponse
from backend.app.services.model_lifecycle import model_lifecycle

router = APIRouter()
START_TIME = time.time()

@router.get("/health", response_model=HealthResponse, summary="Basic Application Health")
async def get_health():
    """
    Returns fundamental application health status, compute device acceleration,
    FFmpeg binary availability, server uptime, and model summary.
    """
    uptime = time.time() - START_TIME
    model_status = model_lifecycle.get_detailed_status()
    return HealthResponse(
        status="healthy",
        environment="development",
        device=DEVICE,
        ffmpeg_available=bool(FFMPEG_EXE),
        ffmpeg_path=FFMPEG_EXE or "Not Found",
        uptime_seconds=round(uptime, 2),
        supported_pairs=["Hindi -> English", "English -> Hindi"],
        modules=model_status.modules,
        timestamp=time.time()
    )
