import time
import threading
import json
from pathlib import Path
from typing import Dict, List, Optional, Any
import numpy as np
import torch
import torch.nn as nn
from transformers import Wav2Vec2Model, AutoConfig

from backend.app.config import DEVICE, SER_MODEL_ID
from backend.app.models.base import BaseSERModel, SERResult, ModelHealthStatus
from backend.app.utils.logger import logger

class Wav2Vec2ClassificationHead(nn.Module):
    """
    Classification head matching the exact tensor architecture saved in
    ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition:
      - dense: Linear(1024, 1024)
      - activation: Tanh
      - output: Linear(1024, 8)
      - pooling: Mean pooling across time frames
    """
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(getattr(config, "final_dropout", 0.1))
        self.output = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        # features shape: (batch_size, sequence_length, hidden_size)
        x = torch.mean(features, dim=1)  # Mean pool across time dimension
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.output(x)
        return x

class SpeechEmotionRecognizer(BaseSERModel):
    """
    Dedicated Pretrained Speech Emotion Recognition Adapter.
    
    Model Provenance & Specifications:
      - Identifier: ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition
      - Base Architecture: Facebook Wav2Vec2 Large XLSR-53 (317M parameters)
      - License: Apache 2.0 (Permissive, open source & research friendly)
      - Training Dataset: RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song),
        comprising 24 professional North American English actors.
      - Exact Native Labels (8 classes):
        0: 'angry', 1: 'calm', 2: 'disgust', 3: 'fearful',
        4: 'happy', 5: 'neutral', 6: 'sad', 7: 'surprised'
      - Confidence Meaning: Softmax posterior probabilities over the 8 discrete classes:
        P(c | X) = exp(z_c) / sum_{j=1}^8 exp(z_j)
      - Supported Input: 16kHz, single-channel (mono) float32 PCM waveform.
      - Hardware Compatibility: Apple Silicon Metal Performance Shaders (MPS) or CPU.
      
    Multilingual & Hindi-English Limitations:
      The underlying checkpoint was fine-tuned exclusively on English speech. While
      Wav2Vec2 XLSR multilingual pretraining provides robust acoustic-prosodic feature
      representations across languages, cross-lingual transfer to Hindi relies strictly on
      acoustic prosody (pitch, energy, speaking rate) rather than validated Devanagari affective
      phonology. The adapter explicitly flags this limitation for Hindi speech inputs.
    """

    NATIVE_LABELS = ["angry", "calm", "disgust", "fearful", "happy", "neutral", "sad", "surprised"]
    CANONICAL_6_LABELS = ["neutral", "happy", "sad", "angry", "fearful", "surprised"]

    # Russell's Circumplex Model Affect Coordinates: (Valence, Arousal, Dominance)
    AFFECT_COORDINATES = {
        "neutral":   (0.50, 0.40, 0.50),
        "calm":      (0.60, 0.25, 0.45),
        "happy":     (0.85, 0.70, 0.65),
        "sad":       (0.20, 0.25, 0.30),
        "angry":     (0.20, 0.85, 0.80),
        "fearful":   (0.25, 0.80, 0.25),
        "disgust":   (0.25, 0.60, 0.60),
        "surprised": (0.65, 0.80, 0.50)
    }

    def __init__(self, model_id: str = SER_MODEL_ID, device: Optional[str] = None):
        self.model_id = model_id
        self.device = device or DEVICE
        self._lock = threading.Lock()
        
        self.config = None
        self.wav2vec2 = None
        self.classifier = None
        self._is_loaded = False
        self._is_loading = False
        self._load_error: Optional[str] = None
        self.id2label: Dict[int, str] = {}

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

    def _load_classifier_weights(self, checkpoint_dir: Optional[Path]) -> None:
        """Load every trained classifier tensor, failing closed on missing or mismatched weights."""
        if checkpoint_dir is None:
            raise FileNotFoundError(
                f"No complete local checkpoint found for '{self.model_id}'. "
                "Download the full SER checkpoint before inference."
            )

        expected_shapes = {
            "classifier.dense.weight": tuple(self.classifier.dense.weight.shape),
            "classifier.dense.bias": tuple(self.classifier.dense.bias.shape),
            "classifier.output.weight": tuple(self.classifier.output.weight.shape),
            "classifier.output.bias": tuple(self.classifier.output.bias.shape),
        }
        loaded = {}

        safe_index = checkpoint_dir / "model.safetensors.index.json"
        bin_index = checkpoint_dir / "pytorch_model.bin.index.json"
        safe_file = checkpoint_dir / "model.safetensors"
        bin_file = checkpoint_dir / "pytorch_model.bin"

        if safe_index.is_file():
            index_data = json.loads(safe_index.read_text(encoding="utf-8"))
            weight_map = index_data.get("weight_map", {})
            shard_names = {
                weight_map[key] for key in expected_shapes if key in weight_map
            }
            if len(shard_names) == 0:
                raise RuntimeError("SER safetensors index does not contain the expected classifier tensors.")
            from safetensors import safe_open
            for shard_name in shard_names:
                shard_path = checkpoint_dir / shard_name
                with safe_open(str(shard_path), framework="pt", device="cpu") as handle:
                    for key in expected_shapes:
                        if key in handle.keys():
                            loaded[key] = handle.get_tensor(key)
        elif safe_file.is_file():
            from safetensors import safe_open
            with safe_open(str(safe_file), framework="pt", device="cpu") as handle:
                for key in expected_shapes:
                    if key in handle.keys():
                        loaded[key] = handle.get_tensor(key)
        elif bin_index.is_file():
            index_data = json.loads(bin_index.read_text(encoding="utf-8"))
            weight_map = index_data.get("weight_map", {})
            shard_names = {
                weight_map[key] for key in expected_shapes if key in weight_map
            }
            if len(shard_names) == 0:
                raise RuntimeError("SER PyTorch index does not contain the expected classifier tensors.")
            for shard_name in shard_names:
                state = torch.load(checkpoint_dir / shard_name, map_location="cpu", weights_only=True)
                loaded.update({key: state[key] for key in expected_shapes if key in state})
        elif bin_file.is_file():
            state = torch.load(bin_file, map_location="cpu", weights_only=True)
            loaded = {key: state[key] for key in expected_shapes if key in state}

        missing = sorted(set(expected_shapes) - set(loaded))
        if missing:
            raise RuntimeError(
                "SER checkpoint is missing trained classifier tensors: " + ", ".join(missing)
            )

        mismatched = {
            key: {"expected": expected_shapes[key], "found": tuple(loaded[key].shape)}
            for key in expected_shapes
            if tuple(loaded[key].shape) != expected_shapes[key]
        }
        if mismatched:
            raise RuntimeError(f"SER classifier checkpoint tensor shape mismatch: {mismatched}")

        classifier_state = {
            key.removeprefix("classifier."): tensor
            for key, tensor in loaded.items()
        }
        self.classifier.load_state_dict(classifier_state, strict=True)

    def _ensure_loaded(self) -> None:
        """Load the Wav2Vec2 backbone and its validated trained classifier head."""
        if self._is_loaded:
            return

        with self._lock:
            if self._is_loaded:
                return

            self._is_loading = True
            try:
                logger.info(f"Loading SER model: {self.model_id} on device: {self.device}")
                snapshot_dir = self._locate_checkpoint_dir()
                model_source = str(snapshot_dir) if snapshot_dir else self.model_id

                self.config = AutoConfig.from_pretrained(model_source)
                raw_id2label = getattr(
                    self.config,
                    "id2label",
                    {i: label for i, label in enumerate(self.NATIVE_LABELS)},
                )
                self.id2label = {
                    int(index): str(label).lower()
                    for index, label in raw_id2label.items()
                }

                self.wav2vec2 = Wav2Vec2Model.from_pretrained(model_source)
                # from_pretrained may have downloaded the remote snapshot just now.
                if snapshot_dir is None:
                    snapshot_dir = self._locate_checkpoint_dir()

                if int(getattr(self.config, "num_labels", 0)) != len(self.NATIVE_LABELS):
                    raise RuntimeError(
                        f"SER checkpoint declares {getattr(self.config, 'num_labels', 'unknown')} labels; "
                        f"expected {len(self.NATIVE_LABELS)}."
                    )

                self.classifier = Wav2Vec2ClassificationHead(self.config)
                self._load_classifier_weights(snapshot_dir)

                target_device = torch.device(
                    self.device if self.device in ("cuda", "mps") else "cpu"
                )
                self.wav2vec2.to(target_device)
                self.classifier.to(target_device)
                self.wav2vec2.eval()
                self.classifier.eval()

                self._is_loaded = True
                self._load_error = None
                logger.info("Speech Emotion Recognition model loaded successfully.")
            except Exception as e:
                self._load_error = str(e)
                self._is_loaded = False
                logger.error(f"Failed to load Speech Emotion Recognition model: {e}")
                raise RuntimeError(f"SER model loading failed: {e}")
            finally:
                self._is_loading = False

    def get_model_info(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_id,
            "provenance": "HuggingFace open model hub / Facebook Wav2Vec2 Large XLSR-53 fine-tuned on RAVDESS",
            "license": "Apache 2.0",
            "architecture": "Wav2Vec2LargeXLSR + Dense Classification Head",
            "supported_input_sample_rate": 16000,
            "supported_input_format": "16kHz mono float32 PCM",
            "confidence_metric": "Softmax posterior probability P(c | X)",
            "native_8_classes": self.NATIVE_LABELS,
            "canonical_6_classes": self.CANONICAL_6_LABELS,
            "training_corpus": "RAVDESS (24 English professional actors, 8 emotion classes)",
            "multilingual_limits": "Trained on English RAVDESS. Does not contain Hindi affective lexicon; cross-lingual emotion recognition relies strictly on acoustic prosody."
        }

    def predict_emotion(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
        source_language: Optional[str] = None
    ) -> SERResult:
        """
        Executes genuine neural acoustic Speech Emotion Recognition on input speech.
        Returns actual predicted labels and softmax posterior probabilities without fabrication.
        """
        t_start = time.perf_counter()

        # 1. Multi-channel to mono & float32 normalization
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)

        # 2. Silence & minimal duration gating
        duration_sec = len(audio) / float(sample_rate) if sample_rate > 0 else 0.0
        rms_energy = float(np.sqrt(np.mean(audio**2))) if len(audio) > 0 else 0.0

        if len(audio) < 1600 or duration_sec < 0.1 or rms_energy < 0.005:
            latency_ms = (time.perf_counter() - t_start) * 1000.0
            return SERResult(
                top_emotion="neutral",
                probabilities={e: (1.0 if e == "neutral" else 0.0) for e in self.NATIVE_LABELS},
                valence=0.50,
                arousal=0.40,
                dominance=0.50,
                latency_ms=round(latency_ms, 2),
                model_name=self.model_id,
                is_demo=False,
                details={
                    "status": "silence_or_below_vad_threshold",
                    "silence_fallback": True,
                    "confidence": 0.0,
                    "rms_energy": round(rms_energy, 5),
                    "duration_sec": round(duration_sec, 3),
                    "training_corpus": "RAVDESS (English)",
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
                model_name=self.model_id,
                is_demo=False,
                details={"status": "model_failed_to_load", "error": str(e)}
            )

        # 4. Standardize input audio via zero-mean unit-variance
        audio_norm = (audio - np.mean(audio)) / (np.std(audio) + 1e-7)

        # 5. Neural Inference
        try:
            target_device = torch.device(self.device if self.device in ("cuda", "mps") else "cpu")
            tensor_input = torch.from_numpy(audio_norm).unsqueeze(0).to(target_device)

            with torch.no_grad():
                encoder_outputs = self.wav2vec2(tensor_input)
                hidden_states = encoder_outputs[0]
                logits = self.classifier(hidden_states)
                probs_tensor = torch.softmax(logits, dim=-1)[0].cpu().numpy()

            # Map exact native probabilities
            native_probs = {}
            for i, p_val in enumerate(probs_tensor):
                label_name = str(self.id2label.get(i, self.NATIVE_LABELS[i])).lower()
                if label_name not in self.NATIVE_LABELS:
                    raise RuntimeError(f"Unexpected SER label at index {i}: {label_name}")
                native_probs[label_name] = round(float(p_val), 4)

            top_idx = int(np.argmax(probs_tensor))
            top_native_emotion = str(self.id2label.get(top_idx, self.NATIVE_LABELS[top_idx])).lower()
            confidence = float(native_probs[top_native_emotion])

            # Map to canonical 6-class taxonomy for downstream compatibility:
            # calm -> neutral, disgust -> angry
            canonical_probs = {c: 0.0 for c in self.CANONICAL_6_LABELS}
            for k, v in native_probs.items():
                if k == "calm":
                    canonical_probs["neutral"] += v
                elif k == "disgust":
                    canonical_probs["angry"] += v
                elif k in canonical_probs:
                    canonical_probs[k] += v

            # Re-normalize canonical probabilities to 1.0
            total_can = sum(canonical_probs.values())
            if total_can > 0:
                canonical_probs = {k: round(v / total_can, 4) for k, v in canonical_probs.items()}

            canonical_top = max(canonical_probs.items(), key=lambda it: it[1])[0]

            # Compute continuous VAD paralinguistic vectors via Circumplex Affect mapping
            v_val = sum(native_probs.get(k, 0.0) * self.AFFECT_COORDINATES.get(k, (0.5, 0.5, 0.5))[0] for k in native_probs)
            a_val = sum(native_probs.get(k, 0.0) * self.AFFECT_COORDINATES.get(k, (0.5, 0.5, 0.5))[1] for k in native_probs)
            d_val = sum(native_probs.get(k, 0.0) * self.AFFECT_COORDINATES.get(k, (0.5, 0.5, 0.5))[2] for k in native_probs)

            # Multilingual limitation warning
            is_hindi = source_language and (source_language.lower().startswith("hi") or source_language.lower() == "hin")
            limitation_warning = None
            if is_hindi:
                limitation_warning = (
                    "Cross-lingual limitation: Wav2Vec2 XLSR SER was trained on the English RAVDESS corpus. "
                    "Inference on Hindi speech reflects acoustic prosody rather than validated Hindi affective phonology."
                )

            latency_ms = (time.perf_counter() - t_start) * 1000.0

            return SERResult(
                top_emotion=top_native_emotion,
                probabilities=canonical_probs,
                valence=round(float(v_val), 3),
                arousal=round(float(a_val), 3),
                dominance=round(float(d_val), 3),
                latency_ms=round(latency_ms, 2),
                model_name=self.model_id,
                is_demo=False,
                details={
                    "confidence": round(confidence, 4),
                    "confidence_metric": "Softmax posterior probability P(c | X)",
                    "native_8_class_probabilities": native_probs,
                    "canonical_6_emotion": canonical_top,
                    "training_corpus": "RAVDESS (24 English speakers, 8 classes)",
                    "device": self.device,
                    "language_limitation_warning": limitation_warning
                }
            )

        except Exception as e:
            latency_ms = (time.perf_counter() - t_start) * 1000.0
            logger.error(f"SER inference failed: {e}")
            return SERResult(
                top_emotion="unavailable",
                probabilities={},
                valence=0.5,
                arousal=0.5,
                dominance=0.5,
                latency_ms=round(latency_ms, 2),
                model_name=self.model_id,
                is_demo=False,
                details={"error": str(e), "status": "inference_failure"}
            )

    def get_health(self) -> ModelHealthStatus:
        return ModelHealthStatus(
            module="SER (Speech Emotion Recognition)",
            model_name=self.model_id,
            is_loaded=self._is_loaded,
            device=self.device,
            status_message="Wav2Vec2 XLSR Speech Emotion Recognition Loaded & Ready" if self._is_loaded else (f"Error: {self._load_error}" if self._load_error else "Ready (Lazy load on demand)"),
            details={
                "model_identifier": self.model_id,
                "supported_languages": ["en (native)", "hi (prosodic transfer with documented limits)"],
                "native_emotion_classes": self.NATIVE_LABELS,
                "training_corpus": "RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song)",
                "confidence_metric": "Softmax posterior probability",
                "input_requirements": "16kHz mono float32 PCM"
            }
        )
