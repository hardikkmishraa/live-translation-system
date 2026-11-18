import re
import threading
import time
from pathlib import Path
from typing import Optional, Dict, Any
import numpy as np
import torch
from backend.app.config import DEVICE, ASR_MODEL_ID
from backend.app.models.base import BaseASRModel, ASRResult, ModelHealthStatus
from backend.app.utils.logger import logger

class WhisperASRModel(BaseASRModel):
    """
    Automatic Speech Recognition using Whisper Transformer architecture (Section 3, Eq. 3 & 4).
    Supports Hindi (hi) and English (en) speech recognition. The standard
    Transformers ASR pipeline output used here does not expose token log-probabilities,
    so this adapter deliberately reports utterance confidence as unavailable.

    Inference Paradigm:
      Whisper is an offline/chunked encoder-decoder model that computes 80-channel log-Mel
      spectrograms and performs cross-attention over full utterance frames. It is NOT an
      autoregressive frame-by-frame streaming model. Live streaming is achieved via sliding
      accumulation chunks (1.5-2.0s) processed at utterance boundaries.
    """

    PUNCTUATION_ONLY_PATTERN = re.compile(
        r"^[\s\.\,\!\?।\-\_\;\:\'\"\(\)\[\]\{\}\<\>\/\\\|~`@#\$%\^&\*\+=]*$"
    )

    TRAILING_OUTRO_PATTERN = re.compile(
        r"(?:[\.\,\!\?।]\s*)?(?:thank\s+you\s+for\s+watching|thanks\s+for\s+watching|subtitles\s+by[^\.\!\?]*|please\s+subscribe|like\s+and\s+subscribe|देखने\s+के\s+लिए\s+धन्यवाद)[\.\!\?।]*$",
        re.IGNORECASE
    )

    STANDALONE_HALLUCINATIONS = {
        "you", "thank you", "thanks", "thank you for watching", "thanks for watching",
        "thank you very much", "thank you so much", "bye", "goodbye",
        "subtitles by", "amara.org", "subscribe", "please subscribe", "like and subscribe",
        "धन्यवाद", "धन्यवाद आपका", "आपका धन्यवाद", "देखने के लिए धन्यवाद",
        "बहुत बहुत धन्यवाद", "अलविदा"
    }

    def __init__(self, model_id: str = ASR_MODEL_ID, device: Optional[str] = None):
        self.model_id = model_id
        self.device = device or DEVICE
        self.pipeline = None
        self._is_loaded = False
        self._is_loading = False
        self._load_error: Optional[str] = None
        self._load_lock = threading.Lock()

    def _locate_checkpoint_dir(self) -> Optional[Path]:
        """Return the newest complete local Hugging Face snapshot, if present."""
        hub_dir = (
            Path.home() / ".cache" / "huggingface" / "hub"
            / f"models--{self.model_id.replace('/', '--')}"
        )
        snapshots_dir = hub_dir / "snapshots"
        if not snapshots_dir.is_dir():
            return None

        snapshots = [p for p in snapshots_dir.iterdir() if p.is_dir()]
        snapshots.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        for snapshot in snapshots:
            has_config = (snapshot / "config.json").is_file()
            has_weights = any(
                (snapshot / filename).is_file()
                for filename in (
                    "model.safetensors",
                    "pytorch_model.bin",
                    "model.safetensors.index.json",
                    "pytorch_model.bin.index.json",
                )
            )
            if has_config and has_weights:
                return snapshot
        return None

    def _ensure_loaded(self):
        """Thread-safe lazy loader for the Whisper pipeline."""
        if self._is_loaded:
            return
        with self._load_lock:
            if self._is_loaded:
                return
            self._is_loading = True
            try:
                from transformers import pipeline
                torch_dtype = torch.float32
                # For CUDA, float16 is advantageous; MPS and CPU prefer float32 for Whisper stability
                if self.device == "cuda":
                    torch_dtype = torch.float16

                device_arg = 0 if self.device == "cuda" else (-1 if self.device == "cpu" else "mps")
                snapshot_dir = self._locate_checkpoint_dir()
                model_source = str(snapshot_dir) if snapshot_dir else self.model_id
                self.pipeline = pipeline(
                    "automatic-speech-recognition",
                    model=model_source,
                    dtype=torch_dtype,
                    device=device_arg,
                )
                self._is_loaded = True
                self._load_error = None
            except Exception as e:
                self._load_error = str(e)
                self._is_loaded = False
                raise RuntimeError(f"Failed to load ASR model '{self.model_id}': {e}")
            finally:
                self._is_loading = False

    @staticmethod
    def _trim_audio_silence(audio: np.ndarray, threshold: float = 0.008, sample_rate: int = 16000) -> np.ndarray:
        """Trims leading and trailing silent/noise frames before inference."""
        frame_len = 512
        hop_len = 256
        if len(audio) < frame_len:
            return audio

        frames = [audio[i:i + frame_len] for i in range(0, len(audio) - frame_len + 1, hop_len)]
        rms = np.array([float(np.sqrt(np.mean(f**2))) for f in frames], dtype=np.float32)
        active = np.where(rms >= threshold)[0]
        if len(active) == 0:
            return np.zeros(0, dtype=np.float32)

        pre_roll = int(sample_rate * 0.15)   # 150ms pre-roll
        post_roll = int(sample_rate * 0.15)  # 150ms post-roll

        start_idx = max(0, int(active[0] * hop_len - pre_roll))
        end_idx = min(len(audio), int((active[-1] + 1) * hop_len + frame_len + post_roll))
        return audio[start_idx:end_idx]

    def transcribe(self, audio: np.ndarray, sample_rate: int = 16000, language: Optional[str] = None) -> ASRResult:
        t_start = time.perf_counter()

        # 1. Multi-channel to mono conversion & float32 normalization
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)

        # 2. Silence, empty, and short recording handling
        duration_sec = len(audio) / float(sample_rate) if sample_rate > 0 else 0.0
        if len(audio) < 1600 or duration_sec < 0.1:  # Less than 100ms
            latency_ms = (time.perf_counter() - t_start) * 1000.0
            return ASRResult(
                text="",
                language=language or "unknown",
                confidence=0.0,
                latency_ms=round(latency_ms, 2),
                model_name=self.model_id,
                is_demo=False,
                details={"status": "too_short_or_empty", "duration_sec": round(duration_sec, 3)}
            )

        rms = float(np.sqrt(np.mean(audio**2)))
        max_amp = float(np.max(np.abs(audio))) if len(audio) > 0 else 0.0

        if rms < 0.005 and max_amp < 0.015:
            # Below audible threshold; avoid sending to Whisper to prevent hallucination
            latency_ms = (time.perf_counter() - t_start) * 1000.0
            return ASRResult(
                text="",
                language=language or "unknown",
                confidence=0.0,
                latency_ms=round(latency_ms, 2),
                model_name=self.model_id,
                is_demo=False,
                details={"status": "silence_detected", "rms": round(rms, 5), "max_amplitude": round(max_amp, 5)}
            )

        # 3. Trim leading and trailing silence to avoid Whisper hallucinating on silence margins
        trimmed_audio = self._trim_audio_silence(audio, threshold=0.007, sample_rate=sample_rate)
        if len(trimmed_audio) < 1600:
            latency_ms = (time.perf_counter() - t_start) * 1000.0
            return ASRResult(
                text="",
                language=language or "unknown",
                confidence=0.0,
                latency_ms=round(latency_ms, 2),
                model_name=self.model_id,
                is_demo=False,
                details={"status": "trimmed_to_silence", "original_len": len(audio)}
            )

        trimmed_rms = float(np.sqrt(np.mean(trimmed_audio**2)))
        trimmed_dur = len(trimmed_audio) / float(sample_rate)

        # 4. Model acquisition
        try:
            self._ensure_loaded()
        except Exception as e:
            latency_ms = (time.perf_counter() - t_start) * 1000.0
            return ASRResult(
                text="",
                language="unknown",
                confidence=None,
                latency_ms=round(latency_ms, 2),
                model_name=self.model_id,
                is_demo=False,
                details={
                    "error": str(e),
                    "status": "model_unavailable",
                    "confidence_status": "unavailable_model_not_loaded",
                }
            )

        # 5. Language identification & Generate kwargs
        if language and language in ("hi", "en"):
            detected_lang = language
        else:
            detected_lang = self.detect_language(trimmed_audio, sample_rate=sample_rate)

        generate_kwargs = {"task": "transcribe", "language": detected_lang}

        try:
            result = self.pipeline(
                {"raw": trimmed_audio, "sampling_rate": sample_rate},
                generate_kwargs=generate_kwargs,
                return_timestamps=False
            )
            raw_text = result.get("text", "").strip()

            # 6. Hallucination and punctuation filtering
            if self.PUNCTUATION_ONLY_PATTERN.match(raw_text):
                raw_text = ""
            else:
                # Strip trailing YouTube outro artifacts (e.g. "I am going to market. Thank you for watching!")
                cleaned_without_outro = self.TRAILING_OUTRO_PATTERN.sub("", raw_text).strip()
                if cleaned_without_outro:
                    if raw_text != cleaned_without_outro and not cleaned_without_outro.endswith((".", "!", "?", "।")):
                        cleaned_without_outro += "."
                    raw_text = cleaned_without_outro
                else:
                    raw_text = ""

            # Check standalone hallucinations (e.g. "Thank you", "धन्यवाद आपका")
            if raw_text:
                norm_text = re.sub(r"[^\w\s]", "", raw_text.lower()).strip()
                if norm_text in self.STANDALONE_HALLUCINATIONS:
                    # Filter if acoustic energy or duration indicates noise hallucination rather than genuine speech
                    if trimmed_rms < 0.022 or trimmed_dur < 0.45 or max_amp < 0.07:
                        raw_text = ""

            # Final language verification: if text has Devanagari characters, enforce Hindi
            if self._has_devanagari(raw_text):
                detected_lang = "hi"

            # The Transformers ASR pipeline does not expose token log-probabilities
            # through this call, so a calibrated utterance confidence is unavailable.
            confidence = None

            latency_ms = (time.perf_counter() - t_start) * 1000.0

            return ASRResult(
                text=raw_text,
                language=detected_lang,
                confidence=round(confidence, 3) if confidence is not None else None,
                latency_ms=round(latency_ms, 2),
                model_name=self.model_id,
                is_demo=False,
                details={
                    "sample_count": len(audio),
                    "duration_sec": round(duration_sec, 2),
                    "trimmed_duration_sec": round(trimmed_dur, 2),
                    "rms_energy": round(rms, 5),
                    "trimmed_rms": round(trimmed_rms, 5),
                    "inference_type": "chunked_offline",
                    "confidence_status": "unavailable_from_pipeline_output"
                }
            )
        except Exception as e:
            latency_ms = (time.perf_counter() - t_start) * 1000.0
            logger.error(f"Whisper transcription failure: {e}")
            return ASRResult(
                text="",
                language="unknown",
                confidence=None,
                latency_ms=round(latency_ms, 2),
                model_name=self.model_id,
                is_demo=False,
                details={
                    "error": str(e),
                    "status": "transcription_error",
                    "confidence_status": "unavailable_transcription_error",
                }
            )

    def detect_language(self, audio: np.ndarray, sample_rate: int = 16000) -> str:
        """
        Acoustically identifies whether input speech is Hindi ('hi') or English ('en')
        from Whisper's audio encoder and decoder step-1 language token logits.
        """
        try:
            self._ensure_loaded()
            model = self.pipeline.model
            feature_extractor = self.pipeline.feature_extractor
            tokenizer = self.pipeline.tokenizer

            # Guard on input length
            if len(audio) < 1600:
                return "en"

            inputs = feature_extractor(audio, sampling_rate=sample_rate, return_tensors="pt")
            input_features = inputs.input_features.to(model.device)

            hi_token_id = tokenizer.convert_tokens_to_ids("<|hi|>")
            en_token_id = tokenizer.convert_tokens_to_ids("<|en|>")
            sot_token_id = tokenizer.convert_tokens_to_ids("<|startoftranscript|>")

            decoder_input_ids = torch.tensor([[sot_token_id]], device=model.device)
            with torch.no_grad():
                logits = model(input_features, decoder_input_ids=decoder_input_ids).logits
                first_step_logits = logits[0, 0]
                p_hi = torch.softmax(first_step_logits, dim=-1)[hi_token_id].item()
                p_en = torch.softmax(first_step_logits, dim=-1)[en_token_id].item()

            if p_hi > p_en:
                return "hi"
            return "en"
        except Exception as e:
            logger.warning(f"[WhisperASR] Acoustic language detection failed, fallback to English: {e}")
            return "en"

    @staticmethod
    def _has_devanagari(text: str) -> bool:
        """Returns True if string contains any Devanagari script characters."""
        if not text:
            return False
        return any('\u0900' <= c <= '\u097F' for c in text)

    @staticmethod
    def _heuristic_lang_detect(text: str) -> str:
        """Detects whether recognized text is Devanagari (Hindi) or Latin (English)."""
        if not text:
            return "en"
        devanagari_chars = sum(1 for c in text if '\u0900' <= c <= '\u097F')
        if devanagari_chars > len(text) * 0.2:
            return "hi"
        return "en"

    def get_health(self) -> ModelHealthStatus:
        return ModelHealthStatus(
            module="ASR (Automatic Speech Recognition)",
            model_name=self.model_id,
            is_loaded=self._is_loaded,
            device=self.device,
            status_message="Whisper ASR Loaded & Ready" if self._is_loaded else (f"Error: {self._load_error}" if self._load_error else "Ready (Lazy load on demand)"),
            details={"model_id": self.model_id, "inference_paradigm": "chunked_offline_transformer"}
        )
