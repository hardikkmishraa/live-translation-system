"""
Microsoft Edge-TTS Provider.
============================
Integrates Microsoft Azure Neural voices via edge-tts with SSML prosodic modulation
(pitch, rate, and volume adjustments). Free, no API keys required.
"""

import asyncio
import time
import uuid
from typing import Dict, Any, Optional
from pathlib import Path
import soundfile as sf
import numpy as np

try:
    import edge_tts
    _EDGE_TTS_AVAILABLE = True
except ImportError:
    _EDGE_TTS_AVAILABLE = False

from backend.app.config import (
    TEMP_STORAGE_DIR,
    EDGE_TTS_VOICE_HI_FEMALE,
    EDGE_TTS_VOICE_HI_MALE,
    EDGE_TTS_VOICE_EN_FEMALE,
    EDGE_TTS_VOICE_EN_MALE,
)
from backend.app.models.base import TTSResult, EmotionPreservationPlan, ModelHealthStatus
from backend.app.models.tts_base import BaseTTSProvider
from backend.app.core.ffmpeg_utils import convert_to_wav
from backend.app.utils.logger import logger


class EdgeTTSProvider(BaseTTSProvider):
    """
    High-quality free neural speech synthesis using Microsoft Edge-TTS.
    Supports Hindi ('hi-IN') and English ('en-US' / 'en-IN') with expressive prosody controls.
    """

    def __init__(self):
        self.temp_dir = TEMP_STORAGE_DIR
        self._is_available = _EDGE_TTS_AVAILABLE

    def _select_voice(self, target_lang: str, gender: str) -> str:
        is_hindi = target_lang.lower().startswith("hi") or target_lang.lower() == "hin"
        is_male = gender.lower() == "male"

        if is_hindi:
            return EDGE_TTS_VOICE_HI_MALE if is_male else EDGE_TTS_VOICE_HI_FEMALE
        return EDGE_TTS_VOICE_EN_MALE if is_male else EDGE_TTS_VOICE_EN_FEMALE

    def synthesize(
        self,
        text: str,
        target_lang: str,
        preservation_plan: EmotionPreservationPlan,
        speaker_gender: Optional[str] = None
    ) -> TTSResult:
        t_start = time.perf_counter()
        clean_text = text.strip()

        if not self._is_available:
            raise RuntimeError("edge_tts package is not installed.")

        if not clean_text:
            latency_ms = (time.perf_counter() - t_start) * 1000.0
            return TTSResult(
                audio_bytes=b"",
                audio_format="wav",
                sample_rate=16000,
                duration_seconds=0.0,
                voice_used="none",
                latency_ms=round(latency_ms, 2),
                model_name="edge_tts_noop",
                emotion_applied=preservation_plan.emotion,
            )

        gender = (speaker_gender or preservation_plan.detected_gender or "female").strip().lower()
        voice = self._select_voice(target_lang, gender)

        # Convert prosody multipliers to SSML prosody strings
        rate_pct = int(round((preservation_plan.rate_multiplier - 1.0) * 100))
        rate_str = f"{rate_pct:+d}%"

        pitch_pct = int(round((preservation_plan.pitch_multiplier - 1.0) * 100))
        pitch_str = f"{pitch_pct:+d}Hz"

        volume_pct = int(round((preservation_plan.volume_multiplier - 1.0) * 100))
        volume_str = f"{volume_pct:+d}%"

        uid = uuid.uuid4().hex
        mp3_out = self.temp_dir / f"edge_{uid}.mp3"
        wav_out = self.temp_dir / f"tts_{uid}.wav"

        async def _synthesize_async():
            communicate = edge_tts.Communicate(
                text=clean_text,
                voice=voice,
                rate=rate_str,
                pitch=pitch_str,
                volume=volume_str
            )
            await communicate.save(str(mp3_out))

        try:
            # Run async edge_tts in event loop
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    import concurrent.futures
                    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                        pool.submit(lambda: asyncio.run(_synthesize_async())).result()
                else:
                    loop.run_until_complete(_synthesize_async())
            except RuntimeError:
                asyncio.run(_synthesize_async())

            if not mp3_out.exists() or mp3_out.stat().st_size == 0:
                raise RuntimeError("Edge-TTS generated empty audio file.")

            # Convert MP3 to standard 16kHz mono WAV
            convert_to_wav(mp3_out, wav_out, sample_rate=16000)

            # Cleanup mp3
            mp3_out.unlink(missing_ok=True)

            with open(wav_out, "rb") as f:
                audio_bytes = f.read()

            try:
                info = sf.info(str(wav_out))
                duration_sec = info.duration
                sr = info.samplerate
            except Exception:
                duration_sec = max(0.1, (len(audio_bytes) - 44) / 32000.0)
                sr = 16000

            latency_ms = (time.perf_counter() - t_start) * 1000.0

            return TTSResult(
                audio_bytes=audio_bytes,
                audio_format="wav",
                sample_rate=sr,
                duration_seconds=round(duration_sec, 2),
                voice_used=voice,
                latency_ms=round(latency_ms, 2),
                model_name=f"edge_tts_{voice}",
                emotion_applied=preservation_plan.emotion,
                is_demo=False,
                details={
                    "provider": "edge_tts",
                    "voice": voice,
                    "gender": gender,
                    "rate_str": rate_str,
                    "pitch_str": pitch_str,
                    "volume_str": volume_str,
                    "audio_filename": wav_out.name,
                    "download_ready": True,
                    "expressive_synthesis": True,
                }
            )
        except Exception as e:
            mp3_out.unlink(missing_ok=True)
            wav_out.unlink(missing_ok=True)
            logger.error(f"[EdgeTTSProvider] Synthesis failed: {e}")
            raise

    def get_health(self) -> ModelHealthStatus:
        return ModelHealthStatus(
            module="TTS (Edge-TTS Neural)",
            model_name="Microsoft Edge Neural Voices",
            is_loaded=self._is_available,
            device="cloud_websocket",
            status_message=(
                "Edge-TTS neural speech client ready (free, no API key needed)."
                if self._is_available
                else "edge_tts library is not available."
            ),
            details={
                "provider": "edge_tts",
                "is_available": self._is_available,
            }
        )
