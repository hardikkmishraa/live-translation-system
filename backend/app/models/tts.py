"""
Emotion-Preserving Text-to-Speech (TTS) Service.
================================================
Orchestrates primary and fallback TTS providers (Sarvam AI bulbul:v4-flash, Microsoft Edge-TTS,
and local macOS native 'say') with emotion prosody modulation and gender matching.
"""

import time
from typing import Dict, Optional, Any, List
import numpy as np

from backend.app.config import (
    TTS_ENGINE,
    SARVAM_API_KEY,
    TEMP_STORAGE_DIR,
)
from backend.app.models.base import BaseTTSModel, TTSResult, EmotionPreservationPlan, ModelHealthStatus
from backend.app.models.tts_base import BaseTTSProvider
from backend.app.models.tts_sarvam import SarvamTTSProvider
from backend.app.models.tts_edge import EdgeTTSProvider
from backend.app.models.tts_macos import MacOSTTSProvider
from backend.app.utils.logger import logger


class EmotionPreservingTTS(BaseTTSModel):
    """
    Modular, Multi-Provider Emotion-Preserving TTS Engine.
    Executes primary provider (default: Sarvam bulbul:v4-flash) with automatic cascading
    fallback to Edge-TTS or macOS native 'say' on network or authentication failure.
    """

    SUPPORTED_LANGUAGES = {"hi", "en"}

    def __init__(self, primary_engine: Optional[str] = None):
        self.engine_name = (primary_engine or TTS_ENGINE or "sarvam").lower()
        self.temp_dir = TEMP_STORAGE_DIR

        # Initialize providers
        self.sarvam_provider = SarvamTTSProvider()
        self.edge_provider = EdgeTTSProvider()
        self.macos_provider = MacOSTTSProvider()

        # Backward-compatibility flag for pipeline/health checks
        self._has_native = bool(
            (self.engine_name == "sarvam" and SARVAM_API_KEY)
            or (self.engine_name == "edge_tts" and self.edge_provider._is_available)
            or self.macos_provider._has_native
        )

    def _get_provider_chain(self) -> List[BaseTTSProvider]:
        """Returns ordered list of providers according to configured preference."""
        if self.engine_name == "sarvam":
            return [self.sarvam_provider, self.edge_provider, self.macos_provider]
        elif self.engine_name in ("edge_tts", "edge"):
            return [self.edge_provider, self.sarvam_provider, self.macos_provider]
        else:  # macos_native / system
            return [self.macos_provider, self.edge_provider, self.sarvam_provider]

    def synthesize(
        self,
        text: str,
        target_lang: str,
        preservation_plan: Optional[EmotionPreservationPlan] = None,
        speaker_gender: Optional[str] = None
    ) -> TTSResult:
        t_start = time.perf_counter()
        clean_text = (text or "").strip()

        # Fallback to neutral baseline prosody if no preservation plan provided
        if preservation_plan is None:
            preservation_plan = EmotionPreservationPlan(
                emotion="neutral",
                pitch_multiplier=1.0,
                rate_multiplier=1.0,
                volume_multiplier=1.0,
                gating_weight_z=0.5,
                applied_to_tts=False,
                prosody_notes="Baseline neutral prosody applied"
            )

        if not getattr(self, "_has_native", True):
            latency_ms = (time.perf_counter() - t_start) * 1000.0
            return TTSResult(
                audio_bytes=b"",
                audio_format="wav",
                sample_rate=16000,
                duration_seconds=0.0,
                voice_used="none",
                latency_ms=round(latency_ms, 2),
                model_name="tts_error",
                emotion_applied=preservation_plan.emotion,
                is_demo=False,
                details={
                    "error": (
                        f"TTS engine '{self.engine_name}' is not available in this environment. "
                        "No speech audio was generated."
                    )
                }
            )

        # 1. Validate non-empty text
        if not clean_text:
            latency_ms = (time.perf_counter() - t_start) * 1000.0
            return TTSResult(
                audio_bytes=b"",
                audio_format="wav",
                sample_rate=16000,
                duration_seconds=0.0,
                voice_used="none",
                latency_ms=round(latency_ms, 2),
                model_name="noop",
                emotion_applied=preservation_plan.emotion
            )

        # 2. Validate target language
        t_code = target_lang.lower().strip()
        is_hindi = t_code.startswith("hi") or t_code == "hin"
        is_english = t_code.startswith("en") or t_code == "eng"

        if not (is_hindi or is_english):
            latency_ms = (time.perf_counter() - t_start) * 1000.0
            return TTSResult(
                audio_bytes=b"",
                audio_format="wav",
                sample_rate=16000,
                duration_seconds=0.0,
                voice_used="none",
                latency_ms=round(latency_ms, 2),
                model_name="tts_unsupported_language",
                emotion_applied="none",
                is_demo=False,
                details={"error": f"Target language '{target_lang}' is unsupported. Supported: ['hi', 'en']"}
            )

        provider_chain = self._get_provider_chain()
        last_error = None
        attempted_providers = []

        for idx, provider in enumerate(provider_chain):
            provider_name = provider.__class__.__name__
            attempted_providers.append(provider_name)
            try:
                res = provider.synthesize(
                    text=clean_text,
                    target_lang=target_lang,
                    preservation_plan=preservation_plan,
                    speaker_gender=speaker_gender
                )
                if res.audio_bytes and len(res.audio_bytes) > 44:
                    if idx > 0:
                        res.details["fallback_used"] = True
                        res.details["primary_provider_failed"] = attempted_providers[0]
                        logger.warning(
                            f"[TTS] Primary provider '{attempted_providers[0]}' failed; "
                            f"fallback '{provider_name}' succeeded."
                        )
                    return res
            except Exception as e:
                last_error = e
                logger.warning(f"[TTS] Provider '{provider_name}' failed: {e}")
                continue

        # All providers in chain failed
        latency_ms = (time.perf_counter() - t_start) * 1000.0
        err_msg = f"All TTS providers failed ({attempted_providers}): {last_error}"
        logger.error(f"[TTS] {err_msg}")
        return TTSResult(
            audio_bytes=b"",
            audio_format="wav",
            sample_rate=16000,
            duration_seconds=0.0,
            voice_used="none",
            latency_ms=round(latency_ms, 2),
            model_name="tts_error",
            emotion_applied=preservation_plan.emotion,
            is_demo=False,
            details={
                "error": err_msg,
                "attempted_providers": attempted_providers,
            }
        )

    def get_health(self) -> ModelHealthStatus:
        active_chain = self._get_provider_chain()
        primary = active_chain[0]
        primary_health = primary.get_health()

        return ModelHealthStatus(
            module="TTS (Text-to-Speech Engine)",
            model_name=f"TTS Provider [{self.engine_name}] -> {primary_health.model_name}",
            is_loaded=primary_health.is_loaded,
            device=primary_health.device,
            status_message=f"Primary: {primary_health.status_message} (Cascading fallback enabled)",
            details={
                "configured_engine": self.engine_name,
                "primary_health": primary_health.details,
                "has_native_fallback": self.macos_provider._has_native,
                "edge_available": self.edge_provider._is_available,
            }
        )
