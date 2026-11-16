from typing import List, Dict
from pydantic import BaseModel, Field

class LanguageInfo(BaseModel):
    code: str
    name: str
    native_name: str
    script: str
    supported_as_source: bool
    supported_as_target: bool

class MediaLimits(BaseModel):
    max_upload_size_mb: int = Field(25, description="Maximum allowed media upload size in megabytes")
    max_duration_seconds: int = Field(180, description="Maximum audio duration in seconds")
    sample_rate_hz: int = Field(16000, description="Internal processing sampling rate")
    channels: int = Field(1, description="Audio channels (Mono)")
    pcm_encoding: str = Field("16-bit signed integer linear PCM", description="Streaming PCM encoding")

class CapabilitiesResponse(BaseModel):
    supported_languages: List[LanguageInfo] = Field(..., description="Bilingual languages supported")
    supported_translation_directions: List[str] = Field(..., description="Valid translation language pairs")
    allowed_audio_extensions: List[str] = Field(..., description="Accepted audio file formats")
    allowed_video_extensions: List[str] = Field(..., description="Accepted video containers for demuxing")
    supported_emotions: List[str] = Field(..., description="Canonical recognized emotional states")
    media_limits: MediaLimits = Field(..., description="Configured input limits and audio specifications")
