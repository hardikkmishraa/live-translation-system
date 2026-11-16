from fastapi import APIRouter
from backend.app.schemas.capabilities import CapabilitiesResponse, LanguageInfo, MediaLimits
from backend.app.config import MAX_UPLOAD_SIZE_MB, SAMPLE_RATE, ALLOWED_AUDIO_EXTENSIONS

router = APIRouter()

@router.get("/capabilities", response_model=CapabilitiesResponse, summary="Supported Languages & Media Formats")
@router.get("/languages-formats", response_model=CapabilitiesResponse, include_in_schema=False)
async def get_capabilities():
    """
    Returns comprehensive inventory of supported languages, valid translation directions,
    accepted media container formats, and operational upload limits.
    """
    languages = [
        LanguageInfo(
            code="hi",
            name="Hindi",
            native_name="हिन्दी",
            script="Devanagari",
            supported_as_source=True,
            supported_as_target=True
        ),
        LanguageInfo(
            code="en",
            name="English",
            native_name="English",
            script="Latin",
            supported_as_source=True,
            supported_as_target=True
        )
    ]

    return CapabilitiesResponse(
        supported_languages=languages,
        supported_translation_directions=[
            "Hindi -> English",
            "English -> Hindi"
        ],
        allowed_audio_extensions=[".wav", ".mp3", ".m4a", ".ogg", ".flac"],
        allowed_video_extensions=[".mp4", ".webm"],
        supported_emotions=["neutral", "happy", "sad", "angry", "fearful", "surprised"],
        media_limits=MediaLimits(
            max_upload_size_mb=MAX_UPLOAD_SIZE_MB,
            max_duration_seconds=180,
            sample_rate_hz=SAMPLE_RATE,
            channels=1,
            pcm_encoding="16-bit signed integer linear PCM"
        )
    )
