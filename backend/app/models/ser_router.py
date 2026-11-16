"""
Bilingual Speech Emotion Recognition Router (Phase 18)
======================================================
Dedicated language-aware SER router that dynamically dispatches incoming speech
to the language-matched emotion recognition model:
  - English (en) -> IEMOCAP 4-Class Model (angry, happy, sad, neutral)
  - Hindi (hi)   -> Vishal B. 8-Class Model (angry, disgust, fear, happy, neutral, sad, sarcastic, surprise)
  - Unsupported/Uncertain Language -> Documented fallback to English IEMOCAP baseline with explicit warning.

Architectural Guarantees:
  - Models load once at startup and are reused across requests (no per-request reloads).
  - Preserves distinct label spaces without destructive label merging.
  - Full end-to-end latency instrumentation including acoustic encoder forward pass.
"""

import time
import threading
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np

from backend.app.config import (
    DEVICE,
    SER_IEMOCAP_CHECKPOINT_DIR,
    SER_MODEL_PROVIDER,
    BASE_DIR
)
from backend.app.models.base import BaseSERModel, SERResult, ModelHealthStatus
from backend.app.models.ser_iemocap import IEMOCAPSpeechEmotionRecognizer
from backend.app.models.ser_hindi import HindiSpeechEmotionRecognizer, DEFAULT_HINDI_CHECKPOINT_DIR
from backend.app.utils.logger import logger


def normalize_language_code(lang_str: Optional[str]) -> str:
    """Normalizes language string (e.g., 'hi-IN', 'Hindi', 'en_US') to 2-letter canonical ISO code."""
    if not lang_str:
        return "unknown"
    cleaned = lang_str.strip().lower().replace("_", "-")
    primary = cleaned.split("-")[0]
    if primary in ("hi", "hin", "hindi"):
        return "hi"
    if primary in ("en", "eng", "english"):
        return "en"
    return primary


class BilingualSERRouter(BaseSERModel):
    """
    Bilingual SER Router directing speech audio to the dedicated English or Hindi SER model.
    """

    SUPPORTED_LANGUAGES = ["hi", "en"]

    def __init__(
        self,
        device: Optional[str] = None,
        iemocap_dir: Optional[Union[str, Path]] = None,
        hindi_dir: Optional[Union[str, Path]] = None,
        preload_at_startup: bool = True
    ):
        self.device = device or DEVICE
        self.iemocap_dir = iemocap_dir or SER_IEMOCAP_CHECKPOINT_DIR
        self.hindi_dir = hindi_dir or DEFAULT_HINDI_CHECKPOINT_DIR

        self._lock = threading.Lock()
        self.english_ser = IEMOCAPSpeechEmotionRecognizer(
            checkpoint_dir=self.iemocap_dir,
            device=self.device
        )
        self.hindi_ser = HindiSpeechEmotionRecognizer(
            checkpoint_dir=self.hindi_dir,
            device=self.device
        )

        self._is_loaded = False
        self._is_loading = False
        self._load_error = None
        self._provider_name = "bilingual_router"
        self.model_name = "bilingual-ser-router-v1"

        if preload_at_startup:
            self._ensure_loaded()

    def _ensure_loaded(self):
        """Preloads and warms up both models once at startup."""
        if self._is_loaded:
            return

        with self._lock:
            if self._is_loaded:
                return

            logger.info(f"[Bilingual SER Router] Initializing models on {self.device}...")
            t0 = time.perf_counter()

            self._is_loading = True
            try:
                # Load English IEMOCAP model
                self.english_ser._ensure_loaded()

                # Load Hindi model
                self.hindi_ser._ensure_loaded()

                # Warmup pass with dummy 1-second audio to eliminate first-request JIT/Metal latency
                dummy_wav = np.zeros(16000, dtype=np.float32)
                try:
                    self.english_ser.predict_emotion(dummy_wav, 16000, source_language="en")
                    self.hindi_ser.predict_emotion(dummy_wav, 16000, source_language="hi")
                    logger.info("[Bilingual SER Router] Warmup forward pass completed on both models.")
                except Exception as e:
                    logger.warning(f"[Bilingual SER Router] Warmup pass encountered non-fatal notice: {e}")

                elapsed = time.perf_counter() - t0
                self._is_loaded = True
                self._load_error = None
                logger.info(f"[Bilingual SER Router] Both models ready in {elapsed:.2f}s on {self.device}.")
            except Exception as e:
                self._load_error = str(e)
                logger.error(f"[Bilingual SER Router] Failed to initialize models: {e}")
                raise
            finally:
                self._is_loading = False

    def route_model(self, source_language: Optional[str]) -> Tuple[BaseSERModel, str, bool, Optional[str]]:
        """
        Determines target SER model based on source language.
        Returns: (selected_model, canonical_lang, is_fallback, fallback_reason)
        """
        norm_lang = normalize_language_code(source_language)

        if norm_lang == "hi":
            return self.hindi_ser, "hi", False, None
        elif norm_lang == "en":
            return self.english_ser, "en", False, None
        else:
            # Documented fallback for unknown or unsupported languages
            fallback_reason = (
                f"Source language '{source_language}' is unsupported or uncertain. "
                "Defaulting to English IEMOCAP 4-class SER baseline."
            )
            logger.warning(f"[Bilingual SER Router] {fallback_reason}")
            return self.english_ser, norm_lang, True, fallback_reason

    def predict_emotion(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
        source_language: Optional[str] = None,
        acoustic_features: Optional[Dict[str, Any]] = None
    ) -> SERResult:
        """
        Routes incoming audio to the appropriate model based on source language,
        executes emotion recognition, and records complete routing telemetry.
        """
        self._ensure_loaded()

        t_router_start = time.perf_counter()
        model, canonical_lang, is_fallback, fallback_reason = self.route_model(source_language)

        # Execute model inference
        result = model.predict_emotion(
            audio=audio,
            sample_rate=sample_rate,
            source_language=canonical_lang,
            acoustic_features=acoustic_features
        )

        total_router_latency_ms = round((time.perf_counter() - t_router_start) * 1000.0, 2)

        # Augment result details with routing telemetry
        result.details.update({
            "routed_by": "bilingual_ser_router",
            "routing_source_language": source_language,
            "routing_canonical_language": canonical_lang,
            "routed_model_id": getattr(model, "model_name", "unknown"),
            "routed_model_provider": getattr(model, "_provider_name", "unknown"),
            "is_routing_fallback": is_fallback,
            "routing_fallback_reason": fallback_reason,
            "total_router_latency_ms": total_router_latency_ms,
            "native_label_count": len(getattr(model, "NATIVE_LABELS", []))
        })

        # Update model_name to reflect both router and routed architecture
        result.model_name = f"{self.model_name} -> {getattr(model, 'model_name', 'submodel')}"
        return result

    def get_health(self) -> ModelHealthStatus:
        """Aggregates health across both underlying SER models."""
        en_health = self.english_ser.get_health()
        hi_health = self.hindi_ser.get_health()

        all_loaded = en_health.is_loaded and hi_health.is_loaded
        return ModelHealthStatus(
            module="ser_router",
            model_name=self.model_name,
            is_loaded=all_loaded,
            device=self.device,
            status_message=(
                "Bilingual SER Router operational (both English & Hindi models active)."
                if all_loaded else "One or more SER submodels are not loaded."
            ),
            details={
                "provider": self._provider_name,
                "supported_languages": self.SUPPORTED_LANGUAGES,
                "english_submodel": {
                    "model_name": en_health.model_name,
                    "is_loaded": en_health.is_loaded,
                    "classes": getattr(self.english_ser, "NATIVE_LABELS", [])
                },
                "hindi_submodel": {
                    "model_name": hi_health.model_name,
                    "is_loaded": hi_health.is_loaded,
                    "classes": getattr(self.hindi_ser, "NATIVE_LABELS", [])
                }
            }
        )
