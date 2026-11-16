"""
IEMOCAP Speech Emotion Recognition Model (Phase 11)
===================================================
Dedicated 4-class Speech Emotion Recognition (SER) architecture fine-tuned
on the speaker-independent IEMOCAP dataset (Angry, Happy, Sad, Neutral).

Architecture:
  - Pretrained speech encoder: Wav2Vec2 / XLSR-53 (1024-dim hidden representations)
  - Time-distributed temporal pooling with attention masking
  - Classification head: Linear(1024 -> 256) -> GELU -> Dropout(0.3) -> Linear(256 -> 4)
  - Supports both Experiment A (frozen backbone) and Experiment B (partial top-layer fine-tuning)
"""

import os
import json
import time
import threading
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Wav2Vec2Model, AutoConfig

from backend.app.config import DEVICE, SER_IEMOCAP_CHECKPOINT_DIR
from backend.app.models.base import BaseSERModel, SERResult, ModelHealthStatus
from backend.app.utils.logger import logger

LABEL_LIST = ["angry", "happy", "sad", "neutral"]
LABEL_TO_ID = {name: idx for idx, name in enumerate(LABEL_LIST)}
ID_TO_LABEL = {idx: name for idx, name in enumerate(LABEL_LIST)}


def _locate_encoder_checkpoint(model_name: str = "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition") -> str:
    """Locates cached HuggingFace snapshot to prevent blocking network calls."""
    cache_hub = Path.home() / ".cache/huggingface/hub"
    repo_folder = f"models--{model_name.replace('/', '--')}"
    snapshot_dir = cache_hub / repo_folder / "snapshots"
    if snapshot_dir.exists():
        subdirs = [p for p in snapshot_dir.iterdir() if p.is_dir()]
        if subdirs:
            return str(subdirs[0])
    return model_name


class IEMOCAPEmotionClassifier(nn.Module):
    """
    Dedicated 4-class SER neural architecture for IEMOCAP speech emotion recognition.
    """

    def __init__(
        self,
        pretrained_name_or_path: str = "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition",
        num_labels: int = 4,
        hidden_dropout_prob: float = 0.3,
        classifier_hidden_size: int = 256,
        pooling_mode: str = "mean",
    ):
        super().__init__()
        self.num_labels = num_labels
        self.pooling_mode = pooling_mode
        self.pretrained_name_or_path = pretrained_name_or_path

        resolved_path = _locate_encoder_checkpoint(pretrained_name_or_path)

        # 1. Speech Encoder Backbone (force eager attention for flawless MPS backpropagation)
        try:
            self.encoder = Wav2Vec2Model.from_pretrained(resolved_path, attn_implementation="eager")
            self.hidden_size = self.encoder.config.hidden_size
        except Exception as e:
            logger.warning(f"Could not load pretrained Wav2Vec2 from '{resolved_path}': {e}. Using config fallback.")
            config = AutoConfig.from_pretrained(resolved_path) if os.path.exists(resolved_path) else AutoConfig.from_pretrained("facebook/wav2vec2-base")
            config._attn_implementation = "eager"
            self.encoder = Wav2Vec2Model(config)
            self.hidden_size = config.hidden_size

        # 2. Classification Projection Head
        self.dropout = nn.Dropout(p=hidden_dropout_prob)
        self.dense = nn.Linear(self.hidden_size, classifier_hidden_size)
        self.act_fn = nn.GELU()
        self.out_proj = nn.Linear(classifier_hidden_size, num_labels)

    def freeze_encoder(self):
        """Freezes all speech encoder parameters for Experiment A."""
        for param in self.encoder.parameters():
            param.requires_grad = False
        logger.info("[SER IEMOCAP] Speech encoder frozen completely (Experiment A).")

    def unfreeze_top_layers(self, num_layers: int = 2):
        """
        Unfreezes the top N transformer encoder layers for Experiment B (Partial fine-tuning).
        Leaves feature extractor and lower layers strictly frozen.
        """
        # Ensure all frozen first
        self.freeze_encoder()
        
        # Unfreeze top N layers of transformer encoder
        if hasattr(self.encoder, "encoder") and hasattr(self.encoder.encoder, "layers"):
            total_layers = len(self.encoder.encoder.layers)
            start_idx = max(0, total_layers - num_layers)
            for i in range(start_idx, total_layers):
                for param in self.encoder.encoder.layers[i].parameters():
                    param.requires_grad = True
            logger.info(f"[SER IEMOCAP] Unfroze top {num_layers} encoder layers ({start_idx} to {total_layers - 1}) for Experiment B.")
        else:
            logger.warning("[SER IEMOCAP] Encoder does not contain standard layers list; skipping layer unfreezing.")

    def forward(
        self,
        input_values: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        class_weights: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """
        Forward pass.
        Args:
            input_values: (batch_size, sequence_length) 16kHz float32 audio
            attention_mask: (batch_size, sequence_length) binary mask (1 for audio, 0 for pad)
            labels: (batch_size,) integer class indices (0: angry, 1: happy, 2: sad, 3: neutral)
            class_weights: Optional tensor of shape (num_labels,)
        """
        # Encoder forward pass
        encoder_requires_grad = any(p.requires_grad for p in self.encoder.parameters())
        if encoder_requires_grad:
            outputs = self.encoder(input_values=input_values, attention_mask=attention_mask)
            hidden_states = outputs.last_hidden_state  # (batch_size, time_steps, hidden_size)
        else:
            with torch.no_grad():
                outputs = self.encoder(input_values=input_values, attention_mask=attention_mask)
                hidden_states = outputs.last_hidden_state.detach()

        # Temporal pooling over time dimension
        if self.pooling_mode == "mean":
            if attention_mask is not None:
                # Downsample attention mask to match encoder frame rate (roughly 320x downsampling in Wav2Vec2)
                downsampled_mask = self._downsample_attention_mask(attention_mask, hidden_states.shape[1])
                mask_expanded = downsampled_mask.unsqueeze(-1)
                sum_hidden = torch.sum(hidden_states * mask_expanded, dim=1)
                sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-6)
                pooled_output = sum_hidden / sum_mask
            else:
                pooled_output = torch.mean(hidden_states, dim=1)
        elif self.pooling_mode == "max":
            pooled_output = torch.max(hidden_states, dim=1)[0]
        else:
            pooled_output = hidden_states[:, 0, :]

        # Classification Head
        x = self.dropout(pooled_output)
        x = self.dense(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        logits = self.out_proj(x)
        probabilities = F.softmax(logits, dim=-1)

        loss = None
        if labels is not None:
            if class_weights is not None:
                class_weights = class_weights.to(logits.device)
                loss_fct = nn.CrossEntropyLoss(weight=class_weights)
            else:
                loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        return {
            "loss": loss,
            "logits": logits,
            "probabilities": probabilities,
            "pooled_output": pooled_output
        }

    def _downsample_attention_mask(self, mask: torch.Tensor, target_len: int) -> torch.Tensor:
        """Interpolates raw 16kHz binary mask to encoder temporal step resolution."""
        float_mask = mask.unsqueeze(1).float()
        downsampled = F.interpolate(float_mask, size=target_len, mode="nearest").squeeze(1)
        return downsampled

    def predict_emotion(self, audio: np.ndarray, sr: int = 16000, device: Optional[str] = None) -> Dict[str, Any]:
        """Inference helper predicting 4-class emotion probabilities from raw audio array."""
        self.eval()
        dev = device or ("mps" if torch.backends.mps.is_available() else "cpu")
        self.to(dev)

        # Standard zero-mean unit-variance normalization
        if audio.ndim > 1:
            audio = np.mean(audio, axis=-1)
        mean = np.mean(audio)
        std = np.std(audio) + 1e-7
        norm_audio = (audio - mean) / std

        # Max 6.0 seconds (96,000 samples)
        max_samples = 96000
        if len(norm_audio) > max_samples:
            norm_audio = norm_audio[:max_samples]
            mask = np.ones(max_samples, dtype=np.float32)
        else:
            pad_len = max_samples - len(norm_audio)
            mask = np.zeros(max_samples, dtype=np.float32)
            mask[:len(norm_audio)] = 1.0
            norm_audio = np.pad(norm_audio, (0, pad_len), mode="constant")

        input_tensor = torch.tensor(norm_audio, dtype=torch.float32).unsqueeze(0).to(dev)
        mask_tensor = torch.tensor(mask, dtype=torch.float32).unsqueeze(0).to(dev)

        t0 = time.perf_counter()
        with torch.no_grad():
            res = self.forward(input_values=input_tensor, attention_mask=mask_tensor)
            probs = res["probabilities"][0].cpu().numpy()
        latency_ms = (time.perf_counter() - t0) * 1000.0

        prob_dict = {LABEL_LIST[i]: float(probs[i]) for i in range(len(LABEL_LIST))}
        top_idx = int(np.argmax(probs))
        top_emotion = LABEL_LIST[top_idx]
        top_conf = float(probs[top_idx])

        return {
            "top_emotion": top_emotion,
            "confidence": round(top_conf, 4),
            "probabilities": {k: round(v, 4) for k, v in prob_dict.items()},
            "latency_ms": round(latency_ms, 2),
            "device": dev
        }

    predict_waveform = predict_emotion

    def save_pretrained(self, save_directory: Union[str, Path]):
        """Saves model weights and configuration."""
        save_dir = Path(save_directory)
        save_dir.mkdir(parents=True, exist_ok=True)

        # Weights
        weights_path = save_dir / "pytorch_model.bin"
        torch.save(self.state_dict(), weights_path)

        # Config
        config_data = {
            "num_labels": self.num_labels,
            "labels": LABEL_LIST,
            "label_to_id": LABEL_TO_ID,
            "id_to_label": ID_TO_LABEL,
            "hidden_size": self.hidden_size,
            "pooling_mode": self.pooling_mode,
            "pretrained_name_or_path": self.pretrained_name_or_path,
        }
        with open(save_dir / "config.json", "w") as f:
            json.dump(config_data, f, indent=2)

        logger.info(f"[SER IEMOCAP] Model checkpoint saved to: {save_dir}")

    @classmethod
    def from_pretrained(cls, load_directory: Union[str, Path], device: str = "cpu") -> "IEMOCAPEmotionClassifier":
        """Loads a saved checkpoint and configuration."""
        load_dir = Path(load_directory)
        config_path = load_dir / "config.json"
        weights_path = load_dir / "pytorch_model.bin"

        if not config_path.exists() or not weights_path.exists():
            raise FileNotFoundError(f"Missing config.json or pytorch_model.bin in {load_dir}")

        with open(config_path, "r") as f:
            config_data = json.load(f)

        model = cls(
            pretrained_name_or_path=config_data.get("pretrained_name_or_path", "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"),
            num_labels=config_data.get("num_labels", 4),
            pooling_mode=config_data.get("pooling_mode", "mean")
        )
        state_dict = torch.load(weights_path, map_location=device)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        return model


class IEMOCAPSpeechEmotionRecognizer(BaseSERModel):
    """
    Dedicated Pretrained & Fine-Tuned IEMOCAP Speech Emotion Recognition Service Adapter.
    Integrates the 4-class IEMOCAP model (Angry, Happy, Sad, Neutral) into the VoiceAffect system.
    
    Model Specifications:
      - Dataset: IEMOCAP (5,531 clean audited utterances across Sessions 1–5)
      - Backbone: Wav2Vec2-XLSR-53 (1024-dim hidden states)
      - Head: Linear(1024 -> 256) -> GELU -> Dropout(0.3) -> Linear(256 -> 4)
      - Native Classes: 0: angry, 1: happy (hap+exc), 2: sad, 3: neutral
      - Normalization: Zero-mean, unit-variance float32 waveform (16 kHz mono)
      - Hardware Support: Apple Silicon Metal Performance Shaders (MPS) or CPU
      - Thread Safety: Double-checked locking singleton initialization
    """

    NATIVE_LABELS = LABEL_LIST  # ["angry", "happy", "sad", "neutral"]
    CANONICAL_6_LABELS = ["angry", "calm", "fearful", "happy", "neutral", "sad"]
    CANONICAL_8_LABELS = ["angry", "calm", "disgust", "fearful", "happy", "neutral", "sad", "surprised"]
    AFFECT_COORDINATES = {
        "neutral": (0.50, 0.40, 0.50),
        "happy":   (0.85, 0.70, 0.65),
        "sad":     (0.20, 0.25, 0.30),
        "angry":   (0.20, 0.85, 0.80)
    }

    def __init__(
        self,
        checkpoint_dir: Union[str, Path] = SER_IEMOCAP_CHECKPOINT_DIR,
        device: Optional[str] = None
    ):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.device = device or DEVICE
        self._lock = threading.Lock()
        self.classifier_model: Optional[IEMOCAPEmotionClassifier] = None
        self._is_loaded = False
        self._is_loading = False
        self._load_error: Optional[str] = None
        self._provider_name = "iemocap"
        self.model_name = "iemocap-wav2vec2-xlsr-4class"

    def _ensure_loaded(self):
        """Thread-safe lazy initialization and verification of IEMOCAP checkpoint."""
        if self._is_loaded and self.classifier_model is not None:
            return

        with self._lock:
            if self._is_loaded and self.classifier_model is not None:
                return

            self._is_loading = True
            try:
                if not self.checkpoint_dir.exists():
                    raise FileNotFoundError(
                        f"IEMOCAP checkpoint directory not found at: {self.checkpoint_dir}. "
                        "Verify Phase 11 artifacts in models/ser_iemocap/."
                    )

                config_path = self.checkpoint_dir / "config.json"
                weights_path = self.checkpoint_dir / "pytorch_model.bin"
                if not config_path.exists() or not weights_path.exists():
                    raise FileNotFoundError(
                        f"Incomplete IEMOCAP checkpoint in {self.checkpoint_dir}. "
                        f"Found config.json={config_path.exists()}, pytorch_model.bin={weights_path.exists()}."
                    )

                logger.info(f"[SER IEMOCAP] Loading checkpoint from: {self.checkpoint_dir} onto device: {self.device}")
                t0 = time.perf_counter()
                self.classifier_model = IEMOCAPEmotionClassifier.from_pretrained(
                    self.checkpoint_dir,
                    device=self.device
                )
                load_time = time.perf_counter() - t0
                self._is_loaded = True
                self._load_error = None
                logger.info(f"[SER IEMOCAP] Model successfully initialized in {load_time:.2f}s on {self.device}.")
            except Exception as e:
                self._is_loaded = False
                self._load_error = str(e)
                logger.error(f"[SER IEMOCAP] Failed to load checkpoint: {e}")
                raise RuntimeError(f"IEMOCAP SER loading failed: {e}")
            finally:
                self._is_loading = False

    def predict_emotion(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
        source_language: Optional[str] = None,
        acoustic_features: Optional[Dict[str, Any]] = None
    ) -> SERResult:
        """
        Executes genuine neural acoustic Speech Emotion Recognition on input speech.
        Returns actual predicted labels and softmax posterior probabilities without fabrication.
        """
        t_start = time.perf_counter()

        # 1. Multi-channel to mono & float32 normalization
        if audio is None or len(audio) == 0:
            latency_ms = (time.perf_counter() - t_start) * 1000.0
            return SERResult(
                top_emotion="neutral",
                probabilities={e: (1.0 if e == "neutral" else 0.0) for e in self.NATIVE_LABELS},
                valence=0.50,
                arousal=0.40,
                dominance=0.50,
                latency_ms=round(latency_ms, 2),
                model_name=self.model_name,
                is_demo=False,
                details={
                    "status": "invalid_audio_empty",
                    "insufficient_audio": True,
                    "confidence": 0.0,
                    "model_provider": "iemocap"
                }
            )

        if audio.ndim > 1:
            audio = np.mean(audio, axis=-1)
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)

        # Check for NaN / Inf
        if np.isnan(audio).any() or np.isinf(audio).any():
            latency_ms = (time.perf_counter() - t_start) * 1000.0
            return SERResult(
                top_emotion="neutral",
                probabilities={e: (1.0 if e == "neutral" else 0.0) for e in self.NATIVE_LABELS},
                valence=0.50,
                arousal=0.40,
                dominance=0.50,
                latency_ms=round(latency_ms, 2),
                model_name=self.model_name,
                is_demo=False,
                details={
                    "status": "invalid_audio_nan_or_inf",
                    "insufficient_audio": True,
                    "confidence": 0.0,
                    "model_provider": "iemocap"
                }
            )

        # 2. Silence & minimal duration gating
        duration_sec = len(audio) / float(sample_rate) if sample_rate > 0 else 0.0
        rms_energy = float(np.sqrt(np.mean(audio ** 2))) if len(audio) > 0 else 0.0

        if len(audio) < 1600 or duration_sec < 0.1 or rms_energy < 0.005:
            latency_ms = (time.perf_counter() - t_start) * 1000.0
            return SERResult(
                top_emotion="neutral",
                probabilities={e: (1.0 if e == "neutral" else 0.0) for e in self.NATIVE_LABELS},
                valence=0.50,
                arousal=0.40,
                dominance=0.50,
                latency_ms=round(latency_ms, 2),
                model_name=self.model_name,
                is_demo=False,
                details={
                    "status": "silence_or_below_vad_threshold",
                    "insufficient_audio": True,
                    "silence_fallback": True,
                    "confidence": 0.0,
                    "rms_energy": round(rms_energy, 5),
                    "duration_sec": round(duration_sec, 3),
                    "model_provider": "iemocap",
                    "supported_input": "16kHz mono float32 PCM"
                }
            )

        # 3. Model acquisition
        try:
            self._ensure_loaded()
        except Exception as e:
            latency_ms = (time.perf_counter() - t_start) * 1000.0
            return SERResult(
                top_emotion="unavailable",
                probabilities={},
                valence=0.5,
                arousal=0.5,
                dominance=0.5,
                latency_ms=round(latency_ms, 2),
                model_name=self.model_name,
                is_demo=False,
                details={
                    "status": "failed",
                    "error": str(e),
                    "model_provider": "iemocap"
                }
            )

        # 4. Neural inference
        try:
            inf_out = self.classifier_model.predict_emotion(
                audio=audio,
                sr=sample_rate,
                device=self.device
            )
            top_emotion = inf_out["top_emotion"]
            confidence = inf_out["confidence"]
            prob_dict = inf_out["probabilities"]
            latency_ms = inf_out["latency_ms"]

            v, a, d = self.AFFECT_COORDINATES.get(top_emotion, (0.50, 0.40, 0.50))

            details = {
                "status": "success",
                "confidence": confidence,
                "model_provider": "iemocap",
                "model_version": "1.0.0",
                "checkpoint_path": str(self.checkpoint_dir),
                "device": self.device,
                "native_4_class_probabilities": prob_dict,
                "supported_input": "16kHz mono float32 PCM",
                "duration_sec": round(duration_sec, 3),
                "rms_energy": round(rms_energy, 5)
            }

            # Hindi Language Limitation Flag (Requirement 10)
            if source_language and (source_language.lower().startswith("hi") or source_language.lower() == "hin"):
                details["language_limitation_warning"] = (
                    "IEMOCAP SER was trained on English conversational speech. "
                    "Emotion recognition on Hindi speech has not been validated; "
                    "cross-lingual inference reflects acoustic prosody rather than validated Hindi affective phonology."
                )

            return SERResult(
                top_emotion=top_emotion,
                probabilities=prob_dict,
                valence=v,
                arousal=a,
                dominance=d,
                latency_ms=round(latency_ms, 2),
                model_name=self.model_name,
                is_demo=False,
                details=details
            )

        except Exception as e:
            latency_ms = (time.perf_counter() - t_start) * 1000.0
            logger.error(f"[SER IEMOCAP] Inference execution error: {e}")
            return SERResult(
                top_emotion="unavailable",
                probabilities={},
                valence=0.5,
                arousal=0.5,
                dominance=0.5,
                latency_ms=round(latency_ms, 2),
                model_name=self.model_name,
                is_demo=False,
                details={
                    "status": "failed",
                    "error": str(e),
                    "model_provider": "iemocap"
                }
            )

    def get_health(self) -> ModelHealthStatus:
        """Reports health and initialization state for API health check."""
        status_msg = (
            "IEMOCAP 4-class SER adapter loaded and ready."
            if self._is_loaded else (
                f"IEMOCAP SER failed to load: {self._load_error}"
                if self._load_error else
                "IEMOCAP SER adapter available; weights load on first inference."
            )
        )
        return ModelHealthStatus(
            module="ser",
            model_name=self.model_name,
            is_loaded=self._is_loaded,
            device=self.device,
            status_message=status_msg,
            details={
                "model_provider": "iemocap",
                "checkpoint_dir": str(self.checkpoint_dir),
                "supported_classes": self.NATIVE_LABELS,
                "failure_reason": self._load_error
            }
        )

    def get_model_info(self) -> Dict[str, Any]:
        """Provides transparency metadata for frontend and research dashboard."""
        return {
            "model_name": self.model_name,
            "provenance": "IEMOCAP database (Interactive Emotional Dyadic Motion Capture) / Wav2Vec2-XLSR fine-tuned",
            "classes": self.NATIVE_LABELS,
            "architecture": "Wav2Vec2-XLSR-53 + Temporal Mean Pooling + Linear Projection Head (1024->256->4)",
            "supported_input_sample_rate": 16000,
            "supported_input_format": "16kHz mono float32 PCM",
            "training_split": "Speaker-Independent (Train: Ses1-3, Val: Ses4, Test: Ses5)",
            "model_provider": "iemocap",
            "device": self.device,
            "is_loaded": self._is_loaded,
            "checkpoint_path": str(self.checkpoint_dir),
            "multilingual_limits": "Trained on English IEMOCAP conversational speech. Hindi emotion recognition reflects acoustic prosody only."
        }
