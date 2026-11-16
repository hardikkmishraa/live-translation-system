"""
macOS Native Speech Provider (/usr/bin/say).
============================================
Provides an offline local fallback using macOS built-in system voices
(e.g., Lekha for Hindi, Samantha / Rishi for English).
"""

import shutil
import subprocess
import sys
import time
import uuid
from typing import Dict, Any, Optional
import numpy as np
import scipy.signal
import soundfile as sf

from backend.app.config import (
    TEMP_STORAGE_DIR,
    MACOS_VOICE_HI_FEMALE,
    MACOS_VOICE_HI_MALE,
    MACOS_VOICE_EN_FEMALE,
    MACOS_VOICE_EN_MALE,
)
from backend.app.models.base import TTSResult, EmotionPreservationPlan, ModelHealthStatus
from backend.app.models.tts_base import BaseTTSProvider
from backend.app.utils.logger import logger


class MacOSTTSProvider(BaseTTSProvider):
    """
    Offline fallback speech synthesis using the macOS native 'say' command.
    Applies heuristic pitch shifting via resampling and tempo adjustments via WPM flag.
    """

    SUPPORTED_LANGUAGES = {"hi", "en"}

    def __init__(self):
        self.temp_dir = TEMP_STORAGE_DIR
        self._say_bin = shutil.which("say")
        self._has_native = bool(
            self._say_bin
            and sys.platform == "darwin"
        )

    def _select_voice(self, target_lang: str, gender: str) -> str:
        is_hindi = target_lang.lower().startswith("hi") or target_lang.lower() == "hin"
        is_male = gender.lower() == "male"

        if is_hindi:
            return MACOS_VOICE_HI_MALE if is_male else MACOS_VOICE_HI_FEMALE
        return MACOS_VOICE_EN_MALE if is_male else MACOS_VOICE_EN_FEMALE

    def synthesize(
        self,
        text: str,
        target_lang: str,
        preservation_plan: EmotionPreservationPlan,
        speaker_gender: Optional[str] = None
    ) -> TTSResult:
        t_start = time.perf_counter()
        clean_text = text.strip()

        if not self._has_native:
            raise RuntimeError("macOS native 'say' is not available on this platform.")

        if not clean_text:
            latency_ms = (time.perf_counter() - t_start) * 1000.0
            return TTSResult(
                audio_bytes=b"",
                audio_format="wav",
                sample_rate=16000,
                duration_seconds=0.0,
                voice_used="none",
                latency_ms=round(latency_ms, 2),
                model_name="macos_noop",
                emotion_applied=preservation_plan.emotion,
            )

        gender = (speaker_gender or preservation_plan.detected_gender or "female").strip().lower()
        voice = self._select_voice(target_lang, gender)

        base_wpm = 175
        target_wpm = int(base_wpm * preservation_plan.rate_multiplier)
        target_wpm = max(100, min(300, target_wpm))

        uid = uuid.uuid4().hex
        raw_out = self.temp_dir / f"tts_raw_{uid}.aiff"
        wav_out = self.temp_dir / f"tts_{uid}.wav"

        try:
            cmd = [
                self._say_bin,
                "-v", voice,
                "-r", str(target_wpm),
                "-o", str(raw_out)
            ]
            subprocess.run(
                cmd,
                input=clean_text.encode("utf-8"),
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )

            if not raw_out.exists() or raw_out.stat().st_size <= 44:
                raise RuntimeError(f"Generated macOS TTS file is invalid or empty: {raw_out}")

            audio_data, sr = sf.read(str(raw_out), dtype="float32")
            if audio_data.ndim > 1:
                audio_data = np.mean(audio_data, axis=1)

            raw_out.unlink(missing_ok=True)

            if sr != 16000:
                num_samples = int(round(len(audio_data) * 16000.0 / sr))
                audio_data = scipy.signal.resample(audio_data, num_samples).astype(np.float32)
                sr = 16000

            # Volume scaling
            if abs(preservation_plan.volume_multiplier - 1.0) > 0.02:
                audio_data = audio_data * preservation_plan.volume_multiplier

            # Pitch shifting
            if abs(preservation_plan.pitch_multiplier - 1.0) > 0.05:
                audio_data = self._pitch_shift(audio_data, sr, preservation_plan.pitch_multiplier)

            # Safe normalization
            max_val = np.max(np.abs(audio_data))
            if max_val > 1e-4:
                audio_data = audio_data / max_val * 0.95

            sf.write(str(wav_out), audio_data, sr, subtype="PCM_16")

            with open(wav_out, "rb") as f:
                audio_bytes = f.read()

            duration_sec = float(len(audio_data) / sr)
            latency_ms = (time.perf_counter() - t_start) * 1000.0

            return TTSResult(
                audio_bytes=audio_bytes,
                audio_format="wav",
                sample_rate=sr,
                duration_seconds=round(duration_sec, 2),
                voice_used=voice,
                latency_ms=round(latency_ms, 2),
                model_name=f"macos_say_{voice}",
                emotion_applied=preservation_plan.emotion,
                is_demo=False,
                details={
                    "provider": "macos_native",
                    "pitch_multiplier": preservation_plan.pitch_multiplier,
                    "rate_multiplier": preservation_plan.rate_multiplier,
                    "target_wpm": target_wpm,
                    "gating_weight_z": preservation_plan.gating_weight_z,
                    "audio_filename": wav_out.name,
                    "download_ready": True,
                    "expressive_synthesis": False,
                }
            )
        except Exception as e:
            raw_out.unlink(missing_ok=True)
            wav_out.unlink(missing_ok=True)
            logger.error(f"[MacOSTTSProvider] Synthesis failed: {e}")
            raise

    @staticmethod
    def _pitch_shift(audio: np.ndarray, sr: int, pitch_ratio: float) -> np.ndarray:
        if len(audio) < 256:
            return audio
        new_len = int(len(audio) / pitch_ratio)
        resampled = scipy.signal.resample(audio, new_len)
        final_audio = scipy.signal.resample(resampled, len(audio))
        return final_audio.astype(np.float32)

    def get_health(self) -> ModelHealthStatus:
        return ModelHealthStatus(
            module="TTS (macOS Native)",
            model_name="macOS System Speech (say)",
            is_loaded=self._has_native,
            device="system",
            status_message=(
                "macOS native 'say' command is available as local offline fallback."
                if self._has_native
                else "macOS native 'say' command is unavailable on this environment."
            ),
            details={
                "provider": "macos_native",
                "is_available": self._has_native,
            }
        )
