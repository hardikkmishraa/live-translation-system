import time
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.models.ser import SpeechEmotionRecognizer, Wav2Vec2ClassificationHead
from backend.app.models.emotion_preservation import LatentDepthEmotionPreserver
from backend.app.services.evaluation import EvaluationService
from backend.app.services.model_lifecycle import model_lifecycle
from backend.app.models.base import SERResult
from safetensors.torch import save_file

client = TestClient(app)


@pytest.fixture(autouse=True)
def phase5_env_setup(monkeypatch):
    import backend.app.config as cfg
    monkeypatch.setattr(cfg, "TTS_ENGINE", "macos_native")
    monkeypatch.setattr(cfg, "SER_MODEL_PROVIDER", "ravdess")
    monkeypatch.setattr(model_lifecycle, "_ser_provider", "ravdess")
    saved_ser = model_lifecycle._ser_instance
    model_lifecycle._ser_instance = None
    yield
    model_lifecycle._ser_instance = saved_ser
    model_lifecycle._ser_provider = "iemocap"


def test_lifecycle_reports_lazy_weights_and_tts_capability_truthfully():
    from backend.app.services.model_lifecycle import ModelLifecycleManager
    manager = ModelLifecycleManager()

    initial = manager.get_detailed_status()
    assert initial.modules["asr"].is_loaded is False
    assert initial.modules["ser"].is_loaded is False
    assert initial.all_ready is False
    assert "first inference" in initial.modules["asr"].status_message.lower()

    manager.get_asr()
    manager.get_ser()
    refreshed = manager.get_detailed_status()
    assert refreshed.modules["asr"].is_loaded is False
    assert refreshed.modules["ser"].is_loaded is False
    assert refreshed.modules["tts"].status.value in {
        "available_and_ready", "unsupported_for_feature"
    }


def test_tts_unavailable_returns_no_fabricated_audio(monkeypatch):
    from backend.app.models.tts import EmotionPreservingTTS
    tts = EmotionPreservingTTS()
    monkeypatch.setattr(tts, "_has_native", False)

    result = tts.synthesize("This must be spoken audio.", target_lang="en")
    assert result.audio_bytes == b""
    assert result.model_name == "tts_error"
    assert "No speech audio was generated" in result.details["error"]


# =========================================================================
# A. Speech Emotion Recognition (SER) Adapter Tests
# =========================================================================

def test_ser_provenance_and_metadata():
    ser = model_lifecycle.get_ser()
    info = ser.get_model_info()

    assert info["model_name"] == "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"
    assert info["license"] == "Apache 2.0"
    assert info["architecture"] == "Wav2Vec2LargeXLSR + Dense Classification Head"
    assert info["supported_input_sample_rate"] == 16000
    assert info["confidence_metric"] == "Softmax posterior probability P(c | X)"
    assert len(info["native_8_classes"]) == 8
    assert "calm" in info["native_8_classes"]
    assert "angry" in info["native_8_classes"]
    assert "happy" in info["native_8_classes"]



def test_ser_checkpoint_missing_classifier_weights_fails_closed(tmp_path):
    ser = SpeechEmotionRecognizer()
    config = SimpleNamespace(hidden_size=4, num_labels=8, final_dropout=0.1)
    ser.classifier = Wav2Vec2ClassificationHead(config)

    with pytest.raises(RuntimeError, match="missing trained classifier tensors"):
        ser._load_classifier_weights(tmp_path)

def test_ser_loads_all_trained_classifier_tensors(tmp_path):
    ser = SpeechEmotionRecognizer()
    config = SimpleNamespace(hidden_size=4, num_labels=8, final_dropout=0.1)
    ser.classifier = Wav2Vec2ClassificationHead(config)
    expected = {
        key: tensor.detach().cpu().contiguous()
        for key, tensor in ser.classifier.state_dict().items()
    }
    save_file(
        {f"classifier.{key}": tensor for key, tensor in expected.items()},
        str(tmp_path / "model.safetensors"),
    )

    with torch.no_grad():
        for parameter in ser.classifier.parameters():
            parameter.zero_()
    ser._load_classifier_weights(tmp_path)

    for key, tensor in expected.items():
        assert torch.equal(ser.classifier.state_dict()[key], tensor)


def test_ser_native_labels_and_canonical_mapping():
    ser = model_lifecycle.get_ser()
    assert ser.NATIVE_LABELS == [
        "angry", "calm", "disgust", "fearful", "happy", "neutral", "sad", "surprised"
    ]
    assert ser.CANONICAL_6_LABELS == [
        "neutral", "happy", "sad", "angry", "fearful", "surprised"
    ]

def test_ser_silence_handling_no_hallucination():
    """Verify that pure silence yields neutral and does not invent artificial confidence."""
    ser = model_lifecycle.get_ser()
    silence = np.zeros(16000 * 2, dtype=np.float32)

    res = ser.predict_emotion(silence, sample_rate=16000)
    assert isinstance(res, SERResult)
    assert res.top_emotion == "neutral"
    assert res.details.get("silence_fallback") is True
    assert res.details.get("confidence") == 0.0
    assert np.isclose(sum(res.probabilities.values()), 1.0, atol=1e-3)

def test_ser_hindi_language_limitation_warning():
    """Verify explicit warning when non-English/Hindi audio is evaluated on English-trained SER."""
    ser = model_lifecycle.get_ser()
    # Generate 1s synthetic sine wave
    t = np.linspace(0, 1.0, 16000, endpoint=False)
    synthetic_audio = (0.3 * np.sin(2 * np.pi * 300 * t)).astype(np.float32)

    # Calling with source_language="hi" MUST trigger cross-lingual warning
    res_hi = ser.predict_emotion(synthetic_audio, sample_rate=16000, source_language="hi")
    assert res_hi.details.get("language_limitation_warning") is not None
    assert "RAVDESS" in res_hi.details["language_limitation_warning"]
    assert "Hindi" in res_hi.details["language_limitation_warning"]

    # Calling with source_language="en" should NOT have warning
    res_en = ser.predict_emotion(synthetic_audio, sample_rate=16000, source_language="en")
    assert res_en.details.get("language_limitation_warning") is None

def test_ser_real_model_smoke_test():
    """Real model smoke test executing genuine Wav2Vec2 forward pass on device."""
    ser = model_lifecycle.get_ser()
    t = np.linspace(0, 1.5, 24000, endpoint=False)
    # Synthetic speech-like harmonic signal
    test_waveform = (
        0.4 * np.sin(2 * np.pi * 220 * t) +
        0.2 * np.sin(2 * np.pi * 440 * t) +
        0.1 * np.sin(2 * np.pi * 880 * t)
    ).astype(np.float32)

    start_t = time.time()
    res = ser.predict_emotion(test_waveform, sample_rate=16000)
    elapsed = time.time() - start_t

    assert res.top_emotion in ser.NATIVE_LABELS
    assert res.details.get("canonical_6_emotion") in ser.CANONICAL_6_LABELS
    assert 0.0 <= res.details["confidence"] <= 1.0
    # Probabilities sum to 1.0
    prob_sum = sum(res.probabilities.values())
    assert np.isclose(prob_sum, 1.0, atol=1e-2)
    # Native 8-class probabilities also sum to ~1.0
    native_sum = sum(res.details["native_8_class_probabilities"].values())
    assert np.isclose(native_sum, 1.0, atol=1e-2)
    # Latency recorded
    assert res.latency_ms > 0
    # Coordinates in valid [0, 1] bounds
    assert 0.0 <= res.valence <= 1.0
    assert 0.0 <= res.arousal <= 1.0
    assert 0.0 <= res.dominance <= 1.0

# =========================================================================
# B. Emotion Preservation Service & Fidelity Reporting Tests
# =========================================================================

def test_emotion_preservation_honest_capability_reporting():
    preserver = LatentDepthEmotionPreserver()
    caps = preserver.get_synthesis_capabilities()

    # System speech does NOT support neural emotion or speaker conditioning
    assert caps["expressive_synthesis_supported"] is False
    assert caps["speaker_conditioning_supported"] is False
    assert "Heuristic prosody" in caps["emotional_fidelity_guarantee"]

def test_emotion_preservation_plan_mappings():
    preserver = LatentDepthEmotionPreserver()

    # Test Happy
    res_happy = SERResult(
        top_emotion="happy",
        probabilities={"happy": 0.85, "neutral": 0.15},
        valence=0.9, arousal=0.8, dominance=0.7,
        latency_ms=10.0,
        model_name="ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition",
        details={"confidence": 0.85}
    )
    plan_happy = preserver.create_preservation_plan(res_happy, target_language="en")
    assert plan_happy.pitch_multiplier > 1.0  # Elevated pitch for happy
    assert plan_happy.rate_multiplier > 1.0   # Accelerated tempo for happy
    assert plan_happy.expressive_synthesis_supported is False
    assert "emotional fidelity is not claimed" in plan_happy.emotional_fidelity_status
    assert plan_happy.speaker_similarity_retained is False
    assert "Samantha" in plan_happy.synthesis_voice_used

    # Test Sad
    res_sad = SERResult(
        top_emotion="sad",
        probabilities={"sad": 0.80, "neutral": 0.20},
        valence=0.15, arousal=0.25, dominance=0.3,
        latency_ms=10.0,
        model_name="ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition",
        details={"confidence": 0.80}
    )
    plan_sad = preserver.create_preservation_plan(res_sad, target_language="hi")
    assert plan_sad.pitch_multiplier < 1.0  # Lowered pitch for sad
    assert plan_sad.rate_multiplier < 1.0   # Decelerated tempo for sad
    assert "Lekha" in plan_sad.synthesis_voice_used

def test_emotion_preservation_speaker_consent_gating():
    preserver = LatentDepthEmotionPreserver()
    res = SERResult(
        top_emotion="neutral",
        probabilities={"neutral": 1.0},
        valence=0.5, arousal=0.5, dominance=0.5,
        latency_ms=5.0,
        model_name="ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"
    )

    # Without consent (default)
    plan_no_consent = preserver.create_preservation_plan(res, user_consent_voice_cloning=False)
    assert plan_no_consent.speaker_similarity_retained is False
    assert plan_no_consent.user_consent_granted is False

    # With consent: still False because baseline TTS cannot clone timbre, but consent flag is recorded
    plan_consent = preserver.create_preservation_plan(res, user_consent_voice_cloning=True)
    assert plan_consent.speaker_similarity_retained is False
    assert plan_consent.user_consent_granted is True
    assert "tts_voice" in plan_consent.details
    assert plan_consent.details["tts_voice"] in ("Samantha", "Lekha")


# =========================================================================
# C. Paper-Specific Methods Reproducibility Audit Tests
# =========================================================================

def test_paper_methods_audit():
    preserver = LatentDepthEmotionPreserver()
    audit = preserver.audit_research_reproducibility()

    # CAMuLeNet
    assert audit["camulenet"]["reproducibility_status"] == "UNIMPLEMENTED / UNREPRODUCIBLE"
    assert "code" in audit["camulenet"]["reason"].lower()

    # Latent-Depth Gating
    assert audit["latent_depth_gating"]["reproducibility_status"] == "THEORETICAL ADAPTATION"

    # YourTTS
    assert audit["yourtts"]["reproducibility_status"] == "INCOMPATIBLE WITH HINDI"

# =========================================================================
# D. Evaluation Service Tests
# =========================================================================

def test_evaluation_service_emotion_consistency():
    eval_service = EvaluationService()

    t = np.linspace(0, 1.0, 16000, endpoint=False)
    src_audio = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    gen_audio = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)

    consistency = eval_service.evaluate_emotion_consistency(src_audio, gen_audio, sample_rate=16000)
    assert hasattr(consistency, "source_emotion")
    assert hasattr(consistency, "generated_emotion")
    assert hasattr(consistency, "distribution_cosine_similarity")
    assert hasattr(consistency, "retention_verdict")
    assert consistency.distribution_cosine_similarity >= 0.0

def test_evaluation_service_speech_naturalness():
    eval_service = EvaluationService()

    t = np.linspace(0, 1.0, 16000, endpoint=False)
    tone = (0.4 * np.sin(2 * np.pi * 200 * t)).astype(np.float32)

    naturalness = eval_service.evaluate_speech_naturalness(tone, sample_rate=16000)
    assert naturalness.estimated_snr_db > 0
    assert 1.0 <= naturalness.estimated_mos_proxy <= 5.0
    assert naturalness.acoustic_assessment in ["Clean speech signal", "Degraded or noisy speech signal"]

def test_evaluation_service_speaker_similarity_consent_enforcement():
    eval_service = EvaluationService()
    dummy = np.zeros(1000, dtype=np.float32)

    # Consent not granted
    sim_no_consent = eval_service.evaluate_speaker_similarity(dummy, dummy, user_consent_granted=False)
    assert sim_no_consent.speaker_similarity_retained is False
    assert sim_no_consent.similarity_score == 0.0
    assert sim_no_consent.consent_requirement_met is False

    # Consent granted
    sim_consent = eval_service.evaluate_speaker_similarity(dummy, dummy, user_consent_granted=True)
    assert sim_consent.speaker_similarity_retained is False
    assert sim_consent.consent_requirement_met is True

# =========================================================================
# E. API Endpoint Tests
# =========================================================================

def test_api_evaluation_report_endpoint():
    res = client.get("/api/v1/evaluation/report")
    assert res.status_code == 200
    data = res.json()
    assert data["phase"] == "Phase 5 - Emotion-Aware Pipeline & Scientific Evaluation"
    assert "speech_emotion_recognition" in data
    assert "speech_synthesis_capabilities" in data
    assert "paper_methodology_reproducibility_audit" in data
    assert data["speech_synthesis_capabilities"]["emotion_conditioning"] is False

def test_api_text_translation_with_emotion_preservation_fields():
    res = client.post("/api/v1/text-translate", json={
        "text": "This is a wonderful breakthrough.",
        "source_lang": "en",
        "target_lang": "hi",
        "emotion": "happy",
        "synthesize_speech": True,
        "user_consent_voice_cloning": False
    })
    assert res.status_code == 200
    data = res.json()
    assert data["success"] is True
    assert data["emotion"] == "happy"
    assert data["expressive_synthesis_supported"] is False
    assert data["speaker_similarity_retained"] is False
    assert "pitch_multiplier" in data["preservation_plan"]
    assert data["preservation_plan"]["pitch_multiplier"] > 1.0
