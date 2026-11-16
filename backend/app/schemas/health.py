import time
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field

class HealthResponse(BaseModel):
    status: str = Field("healthy", description="Application operational health")
    environment: str = Field("development", description="Current operating environment")
    device: str = Field(..., description="Active compute device (mps, cuda, cpu)")
    ffmpeg_available: bool = Field(..., description="Whether FFmpeg binary was discovered")
    ffmpeg_path: str = Field(..., description="Path to active FFmpeg binary")
    uptime_seconds: float = Field(..., description="Uptime of backend server process in seconds")
    supported_pairs: List[str] = Field(..., description="Supported bilingual translation language directions")
    modules: Optional[Dict[str, Any]] = Field(None, description="Optional brief model capability map")
    timestamp: float = Field(default_factory=time.time, description="Unix timestamp of check")
