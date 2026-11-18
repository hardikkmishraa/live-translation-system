"""
NMT Models — Hindi <-> English Neural Machine Translation
==========================================================

Models available:
  SarvamNMTModel    — Primary. Translates via Sarvam AI REST API (mayura:v1 / sarvam-translate:v1).
                      Requires SARVAM_API_KEY in backend environment.
                      Does NOT silently fall back to other engines on failure.
                      Preserves original transcription and returns explicit error status.

  BilingualNMTModel — Local offline model. Helsinki-NLP MarianMT (opus-mt-hi-en / opus-mt-en-hi).
                      Used only when explicitly configured via NMT_BACKEND=local.
                      Never used as an automatic fallback when Sarvam fails.

Both implement BaseNMTModel and return standardized NMTResult objects.
"""

import threading
import time
from typing import Optional

import torch

from backend.app.config import DEVICE, NMT_HI_TO_EN_MODEL, NMT_EN_TO_HI_MODEL, SARVAM_API_KEY
from backend.app.models.base import BaseNMTModel, ModelHealthStatus, NMTResult
from backend.app.services.sarvam_translation import (
    SarvamTranslationService,
    SarvamTranslationError,
    sanitize_sensitive_data,
)
from backend.app.utils.logger import logger


# ---------------------------------------------------------------------------
# SarvamNMTModel — Primary Production Translation Model
# ---------------------------------------------------------------------------

class SarvamNMTModel(BaseNMTModel):
    """
    Production Neural Machine Translation model adapter powered by Sarvam AI.

    Supported Pairs:
      - Hindi -> English (hi-IN -> en-IN)
      - English -> Hindi (en-IN -> hi-IN)

    Contract:
      - Extracts transcription from upstream ASR.
      - Dispatches request to dedicated SarvamTranslationService.
      - Returns validated translation and provider metadata.
      - On API failure or rate limit: returns an explicit failure result
        with translated_text="" and preserved source transcription.
      - Never invents simulated translations or triggers unapproved fallbacks.
    """

    SUPPORTED_LANGUAGES = {"hi", "en"}

    def __init__(
        self,
        api_key: Optional[str] = None,
        timeout: float = 10.0,
        max_retries: int = 2,
    ):
        resolved_key = (api_key or SARVAM_API_KEY or "").strip()
        self._api_key = resolved_key
        if not resolved_key:
            self._service: Optional[SarvamTranslationService] = None
            self._is_loaded = False
            self._load_error: Optional[str] = (
                "SARVAM_API_KEY is not configured. Set it in backend environment variables or .env file."
            )
        else:
            self._service = SarvamTranslationService(
                api_key=self._api_key,
                timeout=timeout,
                max_retries=max_retries,
            )
            self._is_loaded = True
            self._load_error = None

    def _normalize_lang_code(self, lang: Optional[str]) -> str:
        """Normalizes various language code representations into short form ('hi' or 'en')."""
        if not lang:
            return ""
        code = lang.lower().strip()
        if code.startswith("hi") or code == "hin":
            return "hi"
        if code.startswith("en") or code == "eng":
            return "en"
        return code

    def translate(self, text: str, source_lang: str, target_lang: str) -> NMTResult:
        """
        Translates text using Sarvam AI.

        Preserves source text under all conditions.
        If translation fails, marks translation status as failed without inventing text.
        """
        t_start = time.perf_counter()
        clean_text = (text or "").strip()

        # 1. Guard: empty or non-alphanumeric input
        if not clean_text or not any(c.isalnum() for c in clean_text):
            return NMTResult(
                source_text=clean_text,
                translated_text="",
                source_lang=source_lang,
                target_lang=target_lang,
                latency_ms=0.0,
                model_name="noop",
                is_demo=False,
                details={"status": "noop", "reason": "empty_or_non_alphanumeric_input"},
            )

        s_lang = self._normalize_lang_code(source_lang)
        t_lang = self._normalize_lang_code(target_lang)

        # 2. Guard: unsupported language pair
        if s_lang not in self.SUPPORTED_LANGUAGES or t_lang not in self.SUPPORTED_LANGUAGES:
            latency_ms = (time.perf_counter() - t_start) * 1000.0
            error_msg = (
                f"Unsupported language pair: '{source_lang}' -> '{target_lang}'. "
                f"Sarvam translation supports Hindi <-> English only."
            )
            logger.warning(f"[Sarvam] {error_msg}")
            return NMTResult(
                source_text=clean_text,
                translated_text="",
                source_lang=s_lang or source_lang,
                target_lang=t_lang or target_lang,
                latency_ms=round(latency_ms, 2),
                model_name="unsupported_language_pair",
                is_demo=False,
                details={
                    "status": "translation_failed",
                    "error": error_msg,
                    "error_type": "UnsupportedLanguagePairError",
                    "provider": "sarvam_ai",
                },
            )

        # 3. Guard: identity translation
        if s_lang == t_lang:
            latency_ms = (time.perf_counter() - t_start) * 1000.0
            return NMTResult(
                source_text=clean_text,
                translated_text=clean_text,
                source_lang=s_lang,
                target_lang=t_lang,
                latency_ms=round(latency_ms, 2),
                model_name="identity",
                is_demo=False,
                details={
                    "status": "identity",
                    "note": "Source and target languages are identical; no translation required.",
                    "provider": "identity",
                },
            )

        # 4. Guard: missing API key configuration
        if not self._service:
            latency_ms = (time.perf_counter() - t_start) * 1000.0
            error_msg = (
                "SARVAM_API_KEY is not configured in backend environment variables. "
                "Translation service cannot execute."
            )
            logger.error(f"[Sarvam] {error_msg}")
            return NMTResult(
                source_text=clean_text,
                translated_text="",
                source_lang=s_lang,
                target_lang=t_lang,
                latency_ms=round(latency_ms, 2),
                model_name="sarvam_ai",
                is_demo=False,
                details={
                    "status": "translation_failed",
                    "provider": "sarvam_ai",
                    "error": error_msg,
                    "error_type": "MissingApiKeyError",
                    "automatic_fallback_used": False,
                },
            )

        # 5. Dispatch to Sarvam Translation Service
        try:
            result = self._service.translate(
                text=clean_text,
                source_lang=s_lang,
                target_lang=t_lang,
                enable_preprocessing=True,
            )

            translated = result["translated_text"]
            # Deduplicate runaway word repetitions (NMT looping hallucination guard)
            words = translated.split()
            if len(words) > 5:
                cleaned_words = []
                consecutive_count = 1
                last_w = None
                for w in words:
                    if w.lower() == (last_w.lower() if last_w else None):
                        consecutive_count += 1
                        if consecutive_count <= 2:
                            cleaned_words.append(w)
                    else:
                        consecutive_count = 1
                        last_w = w
                        cleaned_words.append(w)
                translated = " ".join(cleaned_words)

            return NMTResult(
                source_text=clean_text,
                translated_text=translated,
                source_lang=s_lang,
                target_lang=t_lang,
                latency_ms=result["latency_ms"],
                model_name=result["model"],
                is_demo=False,
                details={
                    "status": "success",
                    "provider": result["provider"],
                    "model": result["model"],
                    "request_id": result["request_id"],
                    "attempts": result.get("attempts", 1),
                    "source_language_code_resolved": result["source_language_code"],
                    "target_language_code_resolved": result["target_language_code"],
                },
            )

        except Exception as exc:
            # Preserve original transcription, mark translation status as failed,
            # return meaningful error details without exposing credentials.
            latency_ms = (time.perf_counter() - t_start) * 1000.0
            clean_error_msg = sanitize_sensitive_data(str(exc), self._api_key)
            error_type = type(exc).__name__

            logger.error(
                f"[Sarvam] Translation failed ({error_type}): {clean_error_msg}"
            )

            return NMTResult(
                source_text=clean_text,
                translated_text="",
                source_lang=s_lang,
                target_lang=t_lang,
                latency_ms=round(latency_ms, 2),
                model_name="sarvam_ai",
                is_demo=False,
                details={
                    "status": "translation_failed",
                    "provider": "sarvam_ai",
                    "error": clean_error_msg,
                    "error_type": error_type,
                    "automatic_fallback_used": False,
                },
            )

    def get_health(self) -> ModelHealthStatus:
        is_ready = self._service is not None
        endpoint = self._service.endpoint if self._service else SarvamTranslationService.API_ENDPOINT
        timeout_val = self._service.timeout if self._service else SarvamTranslationService.DEFAULT_TIMEOUT
        retries_val = self._service.max_retries if self._service else SarvamTranslationService.DEFAULT_MAX_RETRIES

        return ModelHealthStatus(
            module="NMT (Sarvam AI)",
            model_name="mayura:v1 / sarvam-translate:v1",
            is_loaded=is_ready,
            device="sarvam_cloud",
            status_message=(
                "Sarvam AI translation service ready. Endpoint: POST https://api.sarvam.ai/translate"
                if is_ready
                else "SARVAM_API_KEY is not configured in backend environment variables."
            ),
            details={
                "provider": "sarvam_ai",
                "primary_model": "mayura:v1",
                "extended_model": "sarvam-translate:v1",
                "supported_pairs": ["hi-IN -> en-IN", "en-IN -> hi-IN"],
                "api_endpoint": endpoint,
                "api_key_configured": bool(self._api_key),
                "timeout_seconds": timeout_val,
                "max_retries": retries_val,
            },
        )


# ---------------------------------------------------------------------------
# BilingualNMTModel — Local MarianMT (used ONLY when NMT_BACKEND=local)
# ---------------------------------------------------------------------------

class BilingualNMTModel(BaseNMTModel):
    """
    Local offline Neural Machine Translation model (Helsinki-NLP MarianMT).
    Used ONLY when explicitly activated with NMT_BACKEND=local in environment.
    Never activated as an automatic fallback when Sarvam fails.
    """

    SUPPORTED_LANGUAGES = {"hi", "en"}

    def __init__(
        self,
        hi_to_en_model: str = NMT_HI_TO_EN_MODEL,
        en_to_hi_model: str = NMT_EN_TO_HI_MODEL,
        device: Optional[str] = None,
    ):
        self.hi_to_en_model_id = hi_to_en_model
        self.en_to_hi_model_id = en_to_hi_model
        self.device = device or DEVICE
        self._lock = threading.Lock()

        self._hi_to_en_tokenizer = None
        self._hi_to_en_model = None
        self._en_to_hi_tokenizer = None
        self._en_to_hi_model = None

        self._is_loaded = False
        self._is_loading = False
        self._load_error: Optional[str] = None

    def _normalize_lang_code(self, lang: Optional[str]) -> str:
        if not lang:
            return ""
        code = lang.lower().strip()
        if code.startswith("hi") or code == "hin":
            return "hi"
        if code.startswith("en") or code == "eng":
            return "en"
        return code

    def _ensure_loaded(self, direction: str):
        with self._lock:
            self._is_loading = True
            try:
                from transformers import MarianTokenizer, MarianMTModel
                target_device = torch.device(
                    self.device if self.device in ("cuda", "mps") else "cpu"
                )
                if direction == "hi_to_en":
                    if self._hi_to_en_tokenizer is None:
                        logger.info(f"[MarianMT] Loading tokenizer: {self.hi_to_en_model_id}")
                        self._hi_to_en_tokenizer = MarianTokenizer.from_pretrained(
                            self.hi_to_en_model_id
                        )
                    if self._hi_to_en_model is None:
                        logger.info(
                            f"[MarianMT] Loading model: {self.hi_to_en_model_id} -> {self.device}"
                        )
                        self._hi_to_en_model = MarianMTModel.from_pretrained(
                            self.hi_to_en_model_id
                        ).to(target_device)
                        self._hi_to_en_model.eval()

                elif direction == "en_to_hi":
                    if self._en_to_hi_tokenizer is None:
                        logger.info(f"[MarianMT] Loading tokenizer: {self.en_to_hi_model_id}")
                        self._en_to_hi_tokenizer = MarianTokenizer.from_pretrained(
                            self.en_to_hi_model_id
                        )
                    if self._en_to_hi_model is None:
                        logger.info(
                            f"[MarianMT] Loading model: {self.en_to_hi_model_id} -> {self.device}"
                        )
                        self._en_to_hi_model = MarianMTModel.from_pretrained(
                            self.en_to_hi_model_id
                        ).to(target_device)
                        self._en_to_hi_model.eval()

                self._is_loaded = True
                self._load_error = None
            except Exception as e:
                self._load_error = str(e)
                logger.error(f"[MarianMT] Model load failed for direction {direction}: {e}")
                raise RuntimeError(f"MarianMT model loading failed for {direction}: {e}")
            finally:
                self._is_loading = False

    def translate(self, text: str, source_lang: str, target_lang: str) -> NMTResult:
        t_start = time.perf_counter()
        clean_text = (text or "").strip()

        if not clean_text or not any(c.isalnum() for c in clean_text):
            return NMTResult(
                source_text=clean_text,
                translated_text="",
                source_lang=source_lang,
                target_lang=target_lang,
                latency_ms=0.0,
                model_name="noop",
                is_demo=False,
            )

        s_lang = self._normalize_lang_code(source_lang)
        t_lang = self._normalize_lang_code(target_lang)

        if s_lang not in self.SUPPORTED_LANGUAGES or t_lang not in self.SUPPORTED_LANGUAGES:
            latency_ms = (time.perf_counter() - t_start) * 1000.0
            return NMTResult(
                source_text=clean_text,
                translated_text="",
                source_lang=s_lang or source_lang,
                target_lang=t_lang or target_lang,
                latency_ms=round(latency_ms, 2),
                model_name="unsupported_language_pair",
                is_demo=False,
                details={
                    "error": f"Unsupported language pair: '{source_lang}' -> '{target_lang}'."
                },
            )

        if s_lang == t_lang:
            latency_ms = (time.perf_counter() - t_start) * 1000.0
            return NMTResult(
                source_text=clean_text,
                translated_text=clean_text,
                source_lang=s_lang,
                target_lang=t_lang,
                latency_ms=round(latency_ms, 2),
                model_name="identity",
                is_demo=False,
                details={"note": "Source and target languages are identical."},
            )

        direction = f"{s_lang}_to_{t_lang}"
        model_id = (
            self.hi_to_en_model_id if direction == "hi_to_en" else self.en_to_hi_model_id
        )

        try:
            self._ensure_loaded(direction)
            target_device = torch.device(
                self.device if self.device in ("cuda", "mps") else "cpu"
            )

            tok = (
                self._hi_to_en_tokenizer
                if direction == "hi_to_en"
                else self._en_to_hi_tokenizer
            )
            model = (
                self._hi_to_en_model if direction == "hi_to_en" else self._en_to_hi_model
            )

            inputs = tok(
                clean_text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            )
            inputs = {k: v.to(target_device) for k, v in inputs.items()}

            with torch.no_grad():
                generated_tokens = model.generate(**inputs, max_length=512)

            translated = tok.batch_decode(generated_tokens, skip_special_tokens=True)[0].strip()
            translated = translated.strip(" _\t\n\r")
            if not any(c.isalnum() for c in translated):
                translated = ""

            latency_ms = (time.perf_counter() - t_start) * 1000.0
            return NMTResult(
                source_text=clean_text,
                translated_text=translated,
                source_lang=s_lang,
                target_lang=t_lang,
                latency_ms=round(latency_ms, 2),
                model_name=model_id,
                is_demo=False,
                details={"engine": "MarianMT_Transformer", "device": self.device},
            )

        except Exception as e:
            latency_ms = (time.perf_counter() - t_start) * 1000.0
            logger.error(f"[MarianMT] Translation failed: {e}")
            return NMTResult(
                source_text=clean_text,
                translated_text="",
                source_lang=s_lang,
                target_lang=t_lang,
                latency_ms=round(latency_ms, 2),
                model_name=model_id,
                is_demo=False,
                details={"status": "translation_failed", "error": str(e)},
            )

    def get_health(self) -> ModelHealthStatus:
        return ModelHealthStatus(
            module="NMT (MarianMT — Local Offline)",
            model_name=f"{self.hi_to_en_model_id} / {self.en_to_hi_model_id}",
            is_loaded=self._is_loaded,
            device=self.device,
            status_message=(
                "Helsinki-NLP MarianMT loaded and ready"
                if self._is_loaded
                else "Ready — lazy load on first translation call"
            ),
            details={
                "hi_to_en_model": self.hi_to_en_model_id,
                "en_to_hi_model": self.en_to_hi_model_id,
                "engine": "MarianMT_Transformer",
                "note": "Local offline model. Used ONLY when NMT_BACKEND=local.",
            },
        )
