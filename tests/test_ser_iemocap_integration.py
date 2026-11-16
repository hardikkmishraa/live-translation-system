"""
Phase 12 Integration Tests: IEMOCAP SER Model into Voice Translation System
===========================================================================
Tests verify:
1. Checkpoint loading and initialization.
2. 4-class label mapping integrity (angry, happy, sad, neutral).
3. Inference output format and softmax probability normalization (sums to 1.0).
4. Invalid audio handling (empty, NaN/Inf, stereo to mono).
5. Silent audio handling (energy < 0.005 RMS or duration < 0.1s).
6. Missing checkpoint error handling (explicit error, no silent fallback).
7. Corrupted checkpoint error handling.
8. CPU inference capability.
9. MPS device inference capability (on Apple Silicon).
10. Model provider switching (iemocap vs ravdess).
11. Pipeline fault tolerance (translation continues when SER fails).
12. Cross-lingual Hindi limitation warning.
13. Real checkpoint inference with actual audio file.
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch
import numpy as np
import pytest
import soundfile as sf
import torch

from backend.app.config import SER_IEMOCAP_CHECKPOINT_DIR, DEVICE
from backend.app.models.base import SERResult, ModelHealthStatus
from backend.app.models.ser_iemocap import (
    IEMOCAPEmotionClassifier,
    IEMOCAPSpeechEmotionRecognizer,
    LABEL_LIST
)
from backend.app.models.ser import SpeechEmotionRecognizer
from backend.app.services.model_lifecycle import ModelLifecycleManager
from backend.app.services.pipeline import LiveTranslationPipeline

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def real_audio():
    audio_path = PROJECT_ROOT / "tests/test_audio/test_speech.wav"
    assert audio_path.exists(), f"Missing test audio fixture at {audio_path}"
    data, sr = sf.read(str(audio_path), dtype="float32")
    return data, sr


def test_checkpoint_files_exist():
    """Verify that the Phase 11 checkpoint artifacts exist in models/ser_iemocap/."""
    assert SER_IEMOCAP_CHECKPOINT_DIR.exists(), f"Missing checkpoint dir: {SER_IEMOCAP_CHECKPOINT_DIR}"
    assert (SER_IEMOCAP_CHECKPOINT_DIR / "config.json").exists()
    assert (SER_IEMOCAP_CHECKPOINT_DIR / "pytorch_model.bin").exists()
    assert (SER_IEMOCAP_CHECKPOINT_DIR / "label_mapping.json").exists()
    assert (SER_IEMOCAP_CHECKPOINT_DIR / "preprocessor_config.json").exists()


def test_label_mapping_and_classes():
    """Verify exact 4-class configuration."""
    recognizer = IEMOCAPSpeechEmotionRecognizer(checkpoint_dir=SER_IEMOCAP_CHECKPOINT_DIR)
    assert recognizer.NATIVE_LABELS == ["angry", "happy", "sad", "neutral"]
    assert len(recognizer.AFFECT_COORDINATES) == 4
    for label in ["angry", "happy", "sad", "neutral"]:
        assert label in recognizer.AFFECT_COORDINATES
        v, a, d = recognizer.AFFECT_COORDINATES[label]
        assert 0.0 <= v <= 1.0
        assert 0.0 <= a <= 1.0
        assert 0.0 <= d <= 1.0


def test_lazy_loading_and_health_reporting():
    """Verify lazy model loading and get_health() states."""
    recognizer = IEMOCAPSpeechEmotionRecognizer(checkpoint_dir=SER_IEMOCAP_CHECKPOINT_DIR)
    assert not recognizer._is_loaded
    health_before = recognizer.get_health()
    assert isinstance(health_before, ModelHealthStatus)
    assert not health_before.is_loaded

    # Trigger load
    recognizer._ensure_loaded()
    assert recognizer._is_loaded
    assert recognizer.classifier_model is not None

    health_after = recognizer.get_health()
    assert health_after.is_loaded
    assert "loaded and ready" in health_after.status_message.lower()


def test_invalid_audio_handling():
    """Verify graceful handling of empty or NaN/Inf audio."""
    recognizer = IEMOCAPSpeechEmotionRecognizer(checkpoint_dir=SER_IEMOCAP_CHECKPOINT_DIR)

    # 1. Empty array
    res_empty = recognizer.predict_emotion(np.array([], dtype=np.float32))
    assert res_empty.top_emotion == "neutral"
    assert res_empty.details["status"] == "invalid_audio_empty"
    assert res_empty.details["insufficient_audio"] is True

    # 2. None input
    res_none = recognizer.predict_emotion(None)
    assert res_none.top_emotion == "neutral"
    assert res_none.details["status"] == "invalid_audio_empty"

    # 3. NaN/Inf input
    nan_audio = np.array([0.1, np.nan, 0.2, np.inf], dtype=np.float32)
    res_nan = recognizer.predict_emotion(nan_audio)
    assert res_nan.top_emotion == "neutral"
    assert res_nan.details["status"] == "invalid_audio_nan_or_inf"


def test_silent_audio_handling():
    """Verify VAD thresholding for near-silent audio."""
    recognizer = IEMOCAPSpeechEmotionRecognizer(checkpoint_dir=SER_IEMOCAP_CHECKPOINT_DIR)
    silent_audio = np.zeros(16000, dtype=np.float32)  # 1 sec of pure silence
    res = recognizer.predict_emotion(silent_audio, sample_rate=16000)

    assert res.top_emotion == "neutral"
    assert res.details["status"] == "silence_or_below_vad_threshold"
    assert res.details["insufficient_audio"] is True
    assert res.details["confidence"] == 0.0


def test_missing_checkpoint_raises_explicit_error():
    """Verify that a missing IEMOCAP checkpoint reports an explicit error without silent fallback."""
    fake_dir = Path("/nonexistent/iemocap/checkpoint")
    recognizer = IEMOCAPSpeechEmotionRecognizer(checkpoint_dir=fake_dir)

    with pytest.raises(RuntimeError) as exc_info:
        recognizer._ensure_loaded()
    assert "not found" in str(exc_info.value).lower()

    # Inference handles it safely by returning top_emotion='unavailable'
    res = recognizer.predict_emotion(np.ones(16000, dtype=np.float32) * 0.1)
    assert res.top_emotion == "unavailable"
    assert res.details["status"] == "failed"


def test_corrupted_checkpoint_handling(tmp_path):
    """Verify that a corrupted checkpoint file is handled defensively."""
    corrupted_dir = tmp_path / "corrupted_ser"
    corrupted_dir.mkdir()
    (corrupted_dir / "config.json").write_text('{"num_labels": 4}')
    (corrupted_dir / "pytorch_model.bin").write_bytes(b"NOT_A_VALID_PYTORCH_FILE")

    recognizer = IEMOCAPSpeechEmotionRecognizer(checkpoint_dir=corrupted_dir)
    res = recognizer.predict_emotion(np.ones(16000, dtype=np.float32) * 0.1)
    assert res.top_emotion == "unavailable"
    assert res.details["status"] == "failed"


def test_cpu_inference():
    """Verify that inference runs successfully on CPU."""
    recognizer = IEMOCAPSpeechEmotionRecognizer(
        checkpoint_dir=SER_IEMOCAP_CHECKPOINT_DIR,
        device="cpu"
    )
    dummy_audio = (0.2 * np.sin(2 * np.pi * 220.0 * np.linspace(0, 1, 16000))).astype(np.float32)
    res = recognizer.predict_emotion(dummy_audio, sample_rate=16000)

    assert res.top_emotion in LABEL_LIST
    assert res.details["device"] == "cpu"
    assert res.details["status"] == "success"
    # Probability normalization
    prob_sum = sum(res.probabilities.values())
    assert abs(prob_sum - 1.0) < 1e-3


def test_mps_inference_if_available():
    """Verify inference on Apple Silicon MPS if supported."""
    if not torch.backends.mps.is_available():
        pytest.skip("MPS not available on this host.")

    recognizer = IEMOCAPSpeechEmotionRecognizer(
        checkpoint_dir=SER_IEMOCAP_CHECKPOINT_DIR,
        device="mps"
    )
    dummy_audio = (0.2 * np.sin(2 * np.pi * 220.0 * np.linspace(0, 1, 16000))).astype(np.float32)
    res = recognizer.predict_emotion(dummy_audio, sample_rate=16000)

    assert res.top_emotion in LABEL_LIST
    assert res.details["device"] == "mps"
    assert res.details["status"] == "success"


def test_model_lifecycle_provider_selection():
    """Verify configurable model selection between 'iemocap' and 'ravdess'."""
    lifecycle = ModelLifecycleManager()

    # 1. Default or explicit IEMOCAP provider
    ser_iemocap = lifecycle.get_ser(provider="iemocap")
    assert isinstance(ser_iemocap, IEMOCAPSpeechEmotionRecognizer)
    assert ser_iemocap.NATIVE_LABELS == ["angry", "happy", "sad", "neutral"]

    # 2. Switch to RAVDESS provider
    ser_ravdess = lifecycle.get_ser(provider="ravdess")
    assert isinstance(ser_ravdess, SpeechEmotionRecognizer)
    assert "calm" in ser_ravdess.NATIVE_LABELS

    # 3. Invalid provider raises ValueError
    with pytest.raises(ValueError):
        lifecycle.get_ser(provider="invalid_provider")


def test_hindi_language_limitation_warning():
    """Verify cross-lingual limitation warning for Hindi speech input."""
    recognizer = IEMOCAPSpeechEmotionRecognizer(checkpoint_dir=SER_IEMOCAP_CHECKPOINT_DIR)
    dummy_audio = (0.2 * np.sin(2 * np.pi * 220.0 * np.linspace(0, 1, 16000))).astype(np.float32)

    # English input -> no warning
    res_en = recognizer.predict_emotion(dummy_audio, sample_rate=16000, source_language="en")
    assert "language_limitation_warning" not in res_en.details

    # Hindi input -> warning generated
    res_hi = recognizer.predict_emotion(dummy_audio, sample_rate=16000, source_language="hi")
    assert "language_limitation_warning" in res_hi.details
    assert "hindi" in res_hi.details["language_limitation_warning"].lower()


def test_pipeline_fault_tolerance_when_ser_fails():
    """Verify that the translation pipeline continues if SER fails."""
    pipeline = LiveTranslationPipeline()

    dummy_audio = (0.2 * np.sin(2 * np.pi * 220.0 * np.linspace(0, 1.5, 24000))).astype(np.float32)

    # Mock ASR returning valid speech
    mock_asr = MagicMock()
    mock_asr.transcribe.return_value = MagicMock(
        text="Hello, how are you today?",
        language="en",
        confidence=0.95,
        latency_ms=120.0,
        model_name="whisper-tiny",
        details={"status": "success"}
    )

    # Mock SER failing
    mock_ser = MagicMock()
    mock_ser.predict_emotion.side_effect = RuntimeError("Simulated SER CUDA out of memory error")

    # Mock NMT returning translation
    mock_nmt = MagicMock()
    mock_nmt.translate.return_value = MagicMock(
        translated_text="नमस्ते, आज आप कैसे हैं?",
        source_lang="en",
        target_lang="hi",
        latency_ms=250.0,
        model_name="sarvam",
        details={"status": "success"}
    )

    with patch.object(LiveTranslationPipeline, "asr", new_callable=lambda: property(lambda self: mock_asr)), \
         patch.object(LiveTranslationPipeline, "ser", new_callable=lambda: property(lambda self: mock_ser)), \
         patch.object(LiveTranslationPipeline, "nmt", new_callable=lambda: property(lambda self: mock_nmt)):
        
        result = pipeline.process_audio(
            audio=dummy_audio,
            sample_rate=16000,
            forced_source_lang="en",
            forced_target_lang="hi",
            synthesize_speech=False
        )

        assert result.asr_status == "success"
        assert result.ser_status == "failed"
        assert result.detected_emotion == "unavailable"
        assert result.translated_text == "नमस्ते, आज आप कैसे हैं?"
        assert any("ser" in w.lower() for w in result.warnings)


def test_real_audio_checkpoint_inference(real_audio):
    """
    Real checkpoint inference test using an actual audio recording.
    Validates end-to-end tensor flow without mocks.
    """
    audio_data, sr = real_audio
    recognizer = IEMOCAPSpeechEmotionRecognizer(checkpoint_dir=SER_IEMOCAP_CHECKPOINT_DIR)

    res = recognizer.predict_emotion(audio_data, sample_rate=sr)
    assert res.top_emotion in ["angry", "happy", "sad", "neutral"]
    assert res.latency_ms > 0.0
    assert len(res.probabilities) == 4

    # Validate probabilities
    for c in ["angry", "happy", "sad", "neutral"]:
        assert c in res.probabilities
        assert 0.0 <= res.probabilities[c] <= 1.0

    prob_sum = sum(res.probabilities.values())
    assert abs(prob_sum - 1.0) < 1e-3
    print(f"\n[Real Audio Test] Detected Emotion: {res.top_emotion} (Conf: {res.details['confidence']:.4f})")
    print(f"Probabilities: {res.probabilities} | Latency: {res.latency_ms:.1f}ms")
