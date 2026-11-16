import os
from pathlib import Path
from typing import Literal, List, Set
import torch
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent.parent
load_dotenv(BASE_DIR / ".env")

DOCS_DIR = BASE_DIR / "docs"
TEMP_STORAGE_DIR = BASE_DIR / os.getenv("TEMP_STORAGE_DIR", "temp_storage")
TEMP_STORAGE_DIR.mkdir(parents=True, exist_ok=True)

# Hardware Device Detection
def get_optimal_device() -> str:
    device_override = os.getenv("DEVICE", "").lower()
    if device_override in ["cpu", "cuda", "mps"]:
        return device_override
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    return "cpu"

DEVICE = get_optimal_device()

# Audio Settings
SAMPLE_RATE: int = int(os.getenv("SAMPLE_RATE", "16000"))
MAX_UPLOAD_SIZE_MB: int = int(os.getenv("MAX_UPLOAD_SIZE_MB", "25"))
MAX_DURATION_SECONDS: float = float(os.getenv("MAX_DURATION_SECONDS", "180.0"))
ALLOWED_AUDIO_EXTENSIONS: Set[str] = {".wav", ".mp3", ".m4a", ".ogg", ".flac", ".webm", ".mp4"}

# Model Identifiers
ASR_MODEL_ID: str = os.getenv("ASR_MODEL_ID", "openai/whisper-tiny")
SER_MODEL_ID: str = os.getenv("SER_MODEL_ID", "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition")
SER_MODEL_PROVIDER: str = os.getenv("SER_MODEL_PROVIDER", "router").lower()
SER_IEMOCAP_CHECKPOINT_DIR: Path = Path(os.getenv("SER_IEMOCAP_CHECKPOINT_DIR", str(BASE_DIR / "models/ser_iemocap")))
SER_HINDI_CHECKPOINT_DIR: Path = Path(os.getenv("SER_HINDI_CHECKPOINT_DIR", str(BASE_DIR / "models/ser_hindi")))
SER_RAVDESS_MODEL_ID: str = os.getenv("SER_RAVDESS_MODEL_ID", "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition")
# NMT Settings
# NMT_BACKEND: "sarvam" (default, Sarvam AI cloud API)
#            | "local"  (Helsinki-NLP MarianMT, fully offline, ~300 MB download)
NMT_BACKEND: str = os.getenv("NMT_BACKEND", "sarvam")

# Sarvam AI API key — required when NMT_BACKEND=sarvam.
# Set via SARVAM_API_KEY in .env. Never hardcode or expose in frontend JS.
SARVAM_API_KEY: str = os.getenv("SARVAM_API_KEY", "")

# Local MarianMT model identifiers (used only when NMT_BACKEND=local)
NMT_HI_TO_EN_MODEL: str = os.getenv("NMT_HI_TO_EN_MODEL", "Helsinki-NLP/opus-mt-hi-en")
NMT_EN_TO_HI_MODEL: str = os.getenv("NMT_EN_TO_HI_MODEL", "Helsinki-NLP/opus-mt-en-hi")

# TTS Settings
TTS_ENGINE: str = os.getenv("TTS_ENGINE", "sarvam").lower()
SARVAM_TTS_MODEL: str = os.getenv("SARVAM_TTS_MODEL", "bulbul:v4-flash")

TTS_VOICE_HI_FEMALE: str = os.getenv("TTS_VOICE_HI_FEMALE", "kavya_hi_conversational")
TTS_VOICE_HI_MALE: str = os.getenv("TTS_VOICE_HI_MALE", "aditya_hi_conversational")
TTS_VOICE_EN_FEMALE: str = os.getenv("TTS_VOICE_EN_FEMALE", "amelia_en_conversational")
TTS_VOICE_EN_MALE: str = os.getenv("TTS_VOICE_EN_MALE", "dev_en_conversational")

# Edge-TTS voice defaults
EDGE_TTS_VOICE_HI_FEMALE: str = os.getenv("EDGE_TTS_VOICE_HI_FEMALE", "hi-IN-SwaraNeural")
EDGE_TTS_VOICE_HI_MALE: str = os.getenv("EDGE_TTS_VOICE_HI_MALE", "hi-IN-MadhurNeural")
EDGE_TTS_VOICE_EN_FEMALE: str = os.getenv("EDGE_TTS_VOICE_EN_FEMALE", "en-US-JennyNeural")
EDGE_TTS_VOICE_EN_MALE: str = os.getenv("EDGE_TTS_VOICE_EN_MALE", "en-US-GuyNeural")

# macOS native voice defaults (fallback)
MACOS_VOICE_HI_FEMALE: str = "Lekha"
MACOS_VOICE_HI_MALE: str = "Rishi"
MACOS_VOICE_EN_FEMALE: str = "Samantha"
MACOS_VOICE_EN_MALE: str = "Daniel"

# Backwards compatibility aliases
TTS_VOICE_HI: str = os.getenv("TTS_DEFAULT_VOICE_HI", TTS_VOICE_HI_FEMALE)
TTS_VOICE_EN: str = os.getenv("TTS_DEFAULT_VOICE_EN", TTS_VOICE_EN_FEMALE)

# Emotion Modulation & Gating Thresholds
ENABLE_EMOTION_GATING: bool = os.getenv("ENABLE_EMOTION_GATING", "True").lower() in ("true", "1")
VOICE_PITCH_MODULATION: bool = os.getenv("VOICE_PITCH_MODULATION", "True").lower() in ("true", "1")
VOICE_RATE_MODULATION: bool = os.getenv("VOICE_RATE_MODULATION", "True").lower() in ("true", "1")
TTS_CONFIDENCE_THRESHOLD: float = float(os.getenv("TTS_CONFIDENCE_THRESHOLD", "0.50"))
TTS_MIN_CONFIDENCE_THRESHOLD: float = float(os.getenv("TTS_MIN_CONFIDENCE_THRESHOLD", "0.35"))

# CORS Settings
CORS_ORIGINS: List[str] = [
    origin.strip() for origin in os.getenv("CORS_ORIGINS", "*").split(",") if origin.strip()
]

# Cleanup temp directory
def cleanup_temp_dir():
    try:
        for file in TEMP_STORAGE_DIR.glob("*"):
            if file.is_file():
                file.unlink(missing_ok=True)
    except Exception as e:
        print(f"Warning: Failed to clean temp storage: {e}")
