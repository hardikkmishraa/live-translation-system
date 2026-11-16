from fastapi import APIRouter
from backend.app.schemas.models import ModelsStatusResponse
from backend.app.services.model_lifecycle import model_lifecycle

router = APIRouter()

@router.get("/status", response_model=ModelsStatusResponse, summary="Model Readiness & Capabilities")
async def get_models_status():
    """
    Returns truthful, non-fabricated readiness and capability statuses across all cognitive modules.
    Distinguishes:
    - available_and_ready
    - not_installed
    - still_loading
    - failed_to_load
    - unsupported_for_feature
    """
    return model_lifecycle.get_detailed_status()
