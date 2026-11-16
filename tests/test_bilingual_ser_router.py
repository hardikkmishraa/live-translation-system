"""
Unit and Integration Tests for Phase 18 Bilingual SER Router & Live Integration.
================================================================================
Tests:
  1. Language normalization helper
  2. Router dispatch (English -> IEMOCAP 4-class, Hindi -> Vishal B. 8-class)
  3. Router fallback for unsupported/unknown languages with documented warning
  4. Singleton model lifecycle and memory persistence (no reload per request)
  5. Latent-depth emotion preservation and confidence gating across all 8 emotions
  6. Sarvam TTS persona mapping across all 8 emotions
  7. End-to-end pipeline execution with routed SER
"""

import os
import pytest
import numpy as np
from pathlib import Path
from unittest.mock import MagicMock, patch

from backend.app.config import BASE_DIR, DEVICE
from backend.app.models.base import SERResult
from backend.app.models.ser_router import BilingualSERRouter, normalize_language_code
from backend.app.models.ser_hindi import HindiSpeechEmotionRecognizer, HINDI_LABEL_LIST
from backend.app.models.ser_iemocap import IEMOCAPSpeechEmotionRecognizer
from backend.app.models.emotion_preservation import LatentDepthEmotionPreserver
from backend.app.models.tts_sarvam import SarvamTTSProvider
from backend.app.services.pipeline import LiveTranslationPipeline


@pytest.fixture(scope="module")
def router():
    """Initializes router once for all router tests."""
    return BilingualSERRouter(device="cpu", preload_at_startup=True)


@pytest.fixture
def dummy_audio():
    """Generates 1 second of dummy speech-like audio (16 kHz)."""
    np.random.seed(42)
    t = np.linspace(0, 1.0, 16000, endpoint=False)
    # 440 Hz tone with noise
    audio = 0.3 * np.sin(2 * np.pi * 440 * t) + 0.05 * np.random.randn(16000)
    return audio.astype(np.float32)


# ============================================================================
# 1. Language Normalization Tests
# ============================================================================

def test_language_normalization():
    assert normalize_language_code("hi") == "hi"
    assert normalize_language_code("hin") == "hi"
    assert normalize_language_code("Hindi") == "hi"
    assert normalize_language_code("hi-IN") == "hi"
    assert normalize_language_code("hi_IN") == "hi"

    assert normalize_language_code("en") == "en"
    assert normalize_language_code("eng") == "en"
    assert normalize_language_code("English") == "en"
    assert normalize_language_code("en-US") == "en"
    assert normalize_language_code("en_IN") == "en"

    assert normalize_language_code("fr") == "fr"
    assert normalize_language_code("es") == "es"
    assert normalize_language_code("") == "unknown"
    assert normalize_language_code(None) == "unknown"


# ============================================================================
# 2. Router Model Selection & Dispatch Tests
# ============================================================================

def test_route_model_english(router):
    model, lang, is_fallback, fallback_reason = router.route_model("en")
    assert isinstance(model, IEMOCAPSpeechEmotionRecognizer)
    assert lang == "en"
    assert is_fallback is False
    assert fallback_reason is None


def test_route_model_hindi(router):
    model, lang, is_fallback, fallback_reason = router.route_model("hi")
    assert isinstance(model, HindiSpeechEmotionRecognizer)
    assert lang == "hi"
    assert is_fallback is False
    assert fallback_reason is None


def test_route_model_unsupported_fallback(router):
    model, lang, is_fallback, fallback_reason = router.route_model("fr")
    assert isinstance(model, IEMOCAPSpeechEmotionRecognizer)
    assert lang == "fr"
    assert is_fallback is True
    assert "unsupported or uncertain" in fallback_reason
    assert "English IEMOCAP" in fallback_reason


def test_route_model_none_fallback(router):
    model, lang, is_fallback, fallback_reason = router.route_model(None)
    assert isinstance(model, IEMOCAPSpeechEmotionRecognizer)
    assert is_fallback is True


# ============================================================================
# 3. Router Inference & Label Space Preservation Tests
# ============================================================================

def test_predict_emotion_hindi_label_space(router, dummy_audio):
    """Verifies Hindi speech routes to Hindi SER with 8 distinct emotion classes."""
    res = router.predict_emotion(dummy_audio, 16000, source_language="hi")
    assert isinstance(res, SERResult)
    assert res.top_emotion in HINDI_LABEL_LIST
    assert len(res.probabilities) == 8
    assert set(res.probabilities.keys()) == set(HINDI_LABEL_LIST)
    # Check telemetry
    assert res.details.get("routed_by") == "bilingual_ser_router"
    assert res.details.get("routing_canonical_language") == "hi"
    assert res.details.get("is_routing_fallback") is False
    assert res.details.get("native_label_count") == 8
    assert "bilingual-ser-router" in res.model_name
    assert "hindi-wav2vec2-xlsr-8class" in res.model_name


def test_predict_emotion_english_label_space(router, dummy_audio):
    """Verifies English speech routes to IEMOCAP SER with 4 distinct emotion classes."""
    res = router.predict_emotion(dummy_audio, 16000, source_language="en")
    assert isinstance(res, SERResult)
    iemocap_classes = ["angry", "happy", "sad", "neutral"]
    assert res.top_emotion in iemocap_classes
    assert len(res.probabilities) == 4
    assert set(res.probabilities.keys()) == set(iemocap_classes)
    # Check telemetry
    assert res.details.get("routed_by") == "bilingual_ser_router"
    assert res.details.get("routing_canonical_language") == "en"
    assert res.details.get("is_routing_fallback") is False
    assert res.details.get("native_label_count") == 4
    assert "iemocap" in res.model_name


def test_predict_emotion_fallback_telemetry(router, dummy_audio):
    """Verifies fallback language produces documented warning in details."""
    res = router.predict_emotion(dummy_audio, 16000, source_language="de")
    assert res.details.get("is_routing_fallback") is True
    assert "unsupported or uncertain" in res.details.get("routing_fallback_reason")
    assert len(res.probabilities) == 4  # Fell back to 4-class IEMOCAP


# ============================================================================
# 4. Lifecycle & Health Check Tests
# ============================================================================

def test_router_health(router):
    health = router.get_health()
    assert health.module == "ser_router"
    assert health.is_loaded is True
    assert "hindi_submodel" in health.details
    assert "english_submodel" in health.details
    assert health.details["hindi_submodel"]["is_loaded"] is True
    assert health.details["english_submodel"]["is_loaded"] is True


def test_models_persisted_no_reload(router, dummy_audio):
    """Verifies internal model references remain identical across calls."""
    hi_model_ref1 = router.hindi_ser
    en_model_ref1 = router.english_ser

    router.predict_emotion(dummy_audio, 16000, source_language="hi")
    router.predict_emotion(dummy_audio, 16000, source_language="en")

    assert router.hindi_ser is hi_model_ref1
    assert router.english_ser is en_model_ref1


# ============================================================================
# 5. Emotion Preservation & Gating with 8 Classes
# ============================================================================

def test_emotion_preservation_gating():
    preserver = LatentDepthEmotionPreserver()

    # Low confidence (< 0.35) -> neutral fallback
    ser_low = SERResult(
        top_emotion="angry",
        probabilities={"angry": 0.25, "neutral": 0.20},
        valence=0.2, arousal=0.8, dominance=0.7,
        latency_ms=10.0, model_name="test", is_demo=False,
        details={"confidence": 0.25}
    )
    plan_low = preserver.compute_preservation_plan(ser_low, target_lang="en")
    assert plan_low.emotion == "neutral"
    assert plan_low.pitch_multiplier == 1.0
    assert plan_low.rate_multiplier == 1.0
    assert plan_low.volume_multiplier == 1.0
    assert "Gated: SER confidence" in plan_low.prosody_notes

    # High confidence (>= 0.50) -> full expressive prosody
    ser_disgust = SERResult(
        top_emotion="disgust",
        probabilities={"disgust": 0.85},
        valence=0.3, arousal=0.4, dominance=0.6,
        latency_ms=10.0, model_name="test", is_demo=False,
        details={"confidence": 0.85}
    )
    plan_disgust = preserver.compute_preservation_plan(ser_disgust, target_lang="en")
    assert plan_disgust.emotion == "disgust"
    assert plan_disgust.rate_multiplier <= 0.95
    assert "Disgust" in plan_disgust.prosody_notes

    ser_sarcastic = SERResult(
        top_emotion="sarcastic",
        probabilities={"sarcastic": 0.78},
        valence=0.45, arousal=0.55, dominance=0.6,
        latency_ms=10.0, model_name="test", is_demo=False,
        details={"confidence": 0.78}
    )
    plan_sarcastic = preserver.compute_preservation_plan(ser_sarcastic, target_lang="en")
    assert plan_sarcastic.emotion == "sarcastic"
    assert plan_sarcastic.pitch_multiplier >= 1.0
    assert "Sarcastic" in plan_sarcastic.prosody_notes


# ============================================================================
# 6. Sarvam TTS Persona Mapping Tests
# ============================================================================

def test_sarvam_tts_persona_mapping_all_emotions():
    provider = SarvamTTSProvider()

    for emotion in HINDI_LABEL_LIST:
        voice_hi_f = provider._select_speaker("hi", "female", emotion)
        voice_hi_m = provider._select_speaker("hi", "male", emotion)
        voice_en_f = provider._select_speaker("en", "female", emotion)
        voice_en_m = provider._select_speaker("en", "male", emotion)

        assert voice_hi_f, f"Missing Hindi female voice for {emotion}"
        assert voice_hi_m, f"Missing Hindi male voice for {emotion}"
        assert voice_en_f, f"Missing English female voice for {emotion}"
        assert voice_en_m, f"Missing English male voice for {emotion}"


# ============================================================================
# 7. Live Translation Pipeline Integration
# ============================================================================

def test_pipeline_bilingual_routing_integration(dummy_audio):
    """Verifies the unified pipeline routes SER dynamically based on source language."""
    pipeline = LiveTranslationPipeline()

    # Mock ASR and NMT to avoid network dependencies
    with patch.object(pipeline.asr, "transcribe") as mock_asr, \
         patch.object(pipeline.nmt, "translate") as mock_nmt:

        # Case 1: Hindi input
        mock_asr.return_value = MagicMock(
            text="नमस्ते, आप कैसे हैं?",
            language="hi",
            confidence=0.95,
            latency_ms=45.0
        )
        mock_nmt.return_value = MagicMock(
            translated_text="Hello, how are you?",
            source_language="hi",
            target_language="en",
            latency_ms=60.0,
            details={"provider": "sarvam", "model": "mayura:v1"}
        )

        result_hi = pipeline.process_audio(
            audio=dummy_audio,
            sample_rate=16000,
            forced_source_lang="hi",
            forced_target_lang="en",
            synthesize_speech=False
        )

        assert result_hi.source_language == "hi"
        assert result_hi.ser_status == "success"
        assert result_hi.detected_emotion in HINDI_LABEL_LIST
        assert len(result_hi.emotion_scores) == 8
        assert "hindi-wav2vec2-xlsr-8class" in result_hi.ser_model_id

        # Case 2: English input
        mock_asr.return_value = MagicMock(
            text="Hello, nice to meet you.",
            language="en",
            confidence=0.96,
            latency_ms=45.0
        )
        mock_nmt.return_value = MagicMock(
            translated_text="नमस्ते, आपसे मिलकर अच्छा लगा।",
            source_language="en",
            target_language="hi",
            latency_ms=60.0,
            details={"provider": "sarvam", "model": "mayura:v1"}
        )

        result_en = pipeline.process_audio(
            audio=dummy_audio,
            sample_rate=16000,
            forced_source_lang="en",
            forced_target_lang="hi",
            synthesize_speech=False
        )

        assert result_en.source_language == "en"
        assert result_en.ser_status == "success"
        assert result_en.detected_emotion in ["angry", "happy", "sad", "neutral"]
        assert len(result_en.emotion_scores) == 4
        assert "iemocap" in result_en.ser_model_id
