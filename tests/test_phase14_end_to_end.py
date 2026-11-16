"""
Phase 14 End-to-End Integration, Real-World Validation & Resilience Test Suite.
=============================================================================
Distinguishes between:
  1. Hermetic Mocked Tests (isolated network, deterministic simulation of timeouts, 429, 503).
  2. Live Service Verification (genuine Sarvam AI, Edge-TTS, and local audio pipeline).

Covers:
  - Complete pipeline: Audio In -> Preprocessing -> Whisper ASR -> IEMOCAP SER -> Sarvam NMT -> Emotion-Aware TTS
  - Bidirectional language pairs: English -> Hindi and Hindi -> English
  - 4 IEMOCAP Emotion Classes (Angry, Happy, Sad, Neutral) with confidence gating
  - Multi-provider fallback cascade under simulated failures (Timeout, HTTP 429, HTTP 503, Network drops)
  - Audio playback and download readiness
"""

import io
import json
import time
import uuid
from pathlib import Path
from unittest.mock import patch, MagicMock
import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.config import TEMP_STORAGE_DIR
from backend.app.models.base import ASRResult, SERResult, NMTResult, EmotionPreservationPlan, TTSResult
from backend.app.services.pipeline import LiveTranslationPipeline
from backend.app.models.tts import EmotionPreservingTTS

client = TestClient(app)


def generate_test_wav_bytes(duration_s: float = 1.0, sr: int = 16000, freq: float = 220.0) -> bytes:
    """Creates in-memory 16kHz mono WAV bytes for test upload."""
    t = np.linspace(0, duration_s, int(sr * duration_s), endpoint=False)
    audio = (0.6 * np.sin(2 * np.pi * freq * t) + 0.3 * np.sin(2 * np.pi * 2 * freq * t)).astype(np.float32)
    buf = io.BytesIO()
    sf.write(buf, audio, sr, format="WAV", subtype="PCM_16")
    buf.seek(0)
    return buf.read()


# ===========================================================================
# 1. HERMETIC MOCKED TESTS (Fault-Tolerance, Timeouts, HTTP 429, HTTP 503)
# ===========================================================================

class TestHermeticFaultToleranceAndFallback:
    """Isolated tests verifying provider resilience without external network dependencies."""

    @pytest.fixture
    def mock_plan(self):
        return EmotionPreservationPlan(
            emotion="happy",
            pitch_multiplier=1.15,
            rate_multiplier=1.12,
            volume_multiplier=1.10,
            gating_weight_z=0.6,
            applied_to_tts=True,
            prosody_notes="Happy expressive plan",
            synthesis_voice_used="simran_hi_social_excited"
        )

    def test_sarvam_http_503_cascades_to_edge_tts(self, mock_plan):
        """Simulates Sarvam AI HTTP 503; verifies automatic fallback to Edge-TTS."""
        tts = EmotionPreservingTTS(primary_engine="sarvam")
        tts.sarvam_provider.synthesize = MagicMock(side_effect=RuntimeError("Sarvam HTTP 503: Model Unavailable"))

        dummy_wav = b"RIFF" + b"\x00" * 36 + b"data" + b"\x00" * 3200
        mock_edge_res = TTSResult(
            audio_bytes=dummy_wav,
            audio_format="wav",
            sample_rate=16000,
            duration_seconds=1.0,
            voice_used="hi-IN-SwaraNeural",
            latency_ms=250.0,
            model_name="edge_tts_hi-IN-SwaraNeural",
            emotion_applied="happy",
            details={"provider": "edge_tts"}
        )
        tts.edge_provider.synthesize = MagicMock(return_value=mock_edge_res)

        res = tts.synthesize("परीक्षण पाठ", target_lang="hi", preservation_plan=mock_plan)
        assert res.audio_bytes == dummy_wav
        assert res.details.get("fallback_used") is True
        assert res.details.get("primary_provider_failed") == "SarvamTTSProvider"
        assert res.voice_used == "hi-IN-SwaraNeural"

    def test_sarvam_http_429_rate_limit_cascades(self, mock_plan):
        """Simulates Sarvam AI HTTP 429 Rate Limit; verifies automatic fallback."""
        tts = EmotionPreservingTTS(primary_engine="sarvam")
        tts.sarvam_provider.synthesize = MagicMock(side_effect=RuntimeError("Sarvam HTTP 429: Rate limit exceeded"))

        dummy_wav = b"RIFF" + b"\x00" * 36 + b"data" + b"\x00" * 3200
        tts.edge_provider.synthesize = MagicMock(return_value=TTSResult(
            audio_bytes=dummy_wav, audio_format="wav", sample_rate=16000,
            duration_seconds=1.0, voice_used="hi-IN-SwaraNeural", latency_ms=300.0,
            model_name="edge_tts", emotion_applied="happy", details={"provider": "edge_tts"}
        ))

        res = tts.synthesize("नमस्ते", target_lang="hi", preservation_plan=mock_plan)
        assert len(res.audio_bytes) > 0
        assert res.details.get("fallback_used") is True

    def test_network_timeout_cascades_to_macos_say(self, mock_plan):
        """Simulates complete cloud disconnect (Sarvam and Edge fail); verifies macOS say fallback."""
        tts = EmotionPreservingTTS(primary_engine="sarvam")
        tts.sarvam_provider.synthesize = MagicMock(side_effect=TimeoutError("Sarvam socket timed out"))
        tts.edge_provider.synthesize = MagicMock(side_effect=TimeoutError("Edge-TTS socket timed out"))

        dummy_wav = b"RIFF" + b"\x00" * 36 + b"data" + b"\x00" * 1600
        mock_macos_res = TTSResult(
            audio_bytes=dummy_wav, audio_format="wav", sample_rate=16000,
            duration_seconds=0.5, voice_used="Lekha", latency_ms=450.0,
            model_name="macos_say_Lekha", emotion_applied="happy", details={"provider": "macos_native"}
        )
        tts.macos_provider.synthesize = MagicMock(return_value=mock_macos_res)

        res = tts.synthesize("नमस्ते", target_lang="hi", preservation_plan=mock_plan)
        assert res.audio_bytes == dummy_wav
        assert res.details.get("fallback_used") is True
        assert res.model_name == "macos_say_Lekha"

    def test_complete_tts_failure_never_fabricates_audio(self, mock_plan):
        """When all providers fail, returns explicit error without fabricated audio bytes."""
        tts = EmotionPreservingTTS(primary_engine="sarvam")
        tts.sarvam_provider.synthesize = MagicMock(side_effect=RuntimeError("Sarvam Outage"))
        tts.edge_provider.synthesize = MagicMock(side_effect=RuntimeError("Edge Outage"))
        tts.macos_provider.synthesize = MagicMock(side_effect=RuntimeError("macOS 'say' Missing"))

        res = tts.synthesize("नमस्ते", target_lang="hi", preservation_plan=mock_plan)
        assert res.audio_bytes == b""
        assert res.model_name == "tts_error"
        assert "error" in res.details
        assert "All TTS providers failed" in res.details["error"]


# ===========================================================================
# 2. EMOTION PRESERVATION & CONFIDENCE GATING TESTS (4 IEMOCAP CLASSES)
# ===========================================================================

class TestEmotionPreservationAndGating:
    """Verifies that 4 IEMOCAP emotions correctly condition speech without distortion."""

    def test_angry_high_confidence_elevates_prosody(self):
        pipeline = LiveTranslationPipeline()
        ser_res = SERResult(
            top_emotion="angry", probabilities={"angry": 0.88, "neutral": 0.12},
            valence=0.2, arousal=0.85, dominance=0.8, latency_ms=10.0,
            model_name="iemocap-test", details={"confidence": 0.88}
        )
        plan = pipeline.emotion_preserver.compute_preservation_plan(ser_res, target_lang="hi", speaker_gender="male")
        assert plan.emotion == "angry"
        assert plan.pitch_multiplier >= 1.15
        assert plan.rate_multiplier >= 1.15
        assert plan.volume_multiplier >= 1.15

    def test_sad_high_confidence_lowers_prosody(self):
        pipeline = LiveTranslationPipeline()
        ser_res = SERResult(
            top_emotion="sad", probabilities={"sad": 0.92, "neutral": 0.08},
            valence=0.2, arousal=0.25, dominance=0.3, latency_ms=10.0,
            model_name="iemocap-test", details={"confidence": 0.92}
        )
        plan = pipeline.emotion_preserver.compute_preservation_plan(ser_res, target_lang="en", speaker_gender="female")
        assert plan.emotion == "sad"
        assert plan.pitch_multiplier <= 0.90
        assert plan.rate_multiplier <= 0.88
        assert plan.volume_multiplier <= 0.88

    def test_uncertain_confidence_gates_to_neutral(self):
        """Confidence below 0.35 gates to neutral baseline (1.0x)."""
        pipeline = LiveTranslationPipeline()
        ser_res = SERResult(
            top_emotion="angry", probabilities={"angry": 0.28, "neutral": 0.25, "sad": 0.25, "happy": 0.22},
            valence=0.3, arousal=0.7, dominance=0.6, latency_ms=10.0,
            model_name="iemocap-test", details={"confidence": 0.28}
        )
        plan = pipeline.emotion_preserver.compute_preservation_plan(ser_res, target_lang="hi", speaker_gender="female")
        assert plan.emotion == "neutral"
        assert plan.pitch_multiplier == 1.0
        assert plan.rate_multiplier == 1.0
        assert plan.volume_multiplier == 1.0
        assert "Gated" in plan.prosody_notes

    def test_soft_blending_interval(self):
        """Confidence between 0.35 and 0.50 produces smoothly scaled multipliers."""
        pipeline = LiveTranslationPipeline()
        ser_res = SERResult(
            top_emotion="happy", probabilities={"happy": 0.42, "neutral": 0.30},
            valence=0.8, arousal=0.7, dominance=0.6, latency_ms=10.0,
            model_name="iemocap-test", details={"confidence": 0.42}
        )
        plan = pipeline.emotion_preserver.compute_preservation_plan(ser_res, target_lang="hi", speaker_gender="female")
        assert plan.emotion == "happy"
        assert 1.0 < plan.pitch_multiplier < 1.15
        assert 1.0 < plan.rate_multiplier < 1.12
        assert "Soft blended" in plan.prosody_notes


# ===========================================================================
# 3. LIVE PIPELINE & API VALIDATION (Real Audio & Endpoints)
# ===========================================================================

class TestLiveAPIAndPipelineIntegration:
    """Validates real audio processing through API endpoints and pipeline."""

    def test_api_audio_upload_bilingual_hindi_to_english(self):
        """Uploads synthetic audio with forced hi -> en direction."""
        wav_bytes = generate_test_wav_bytes(duration_s=1.0, freq=220.0)
        files = {"file": ("test_hindi_input.wav", wav_bytes, "audio/wav")}
        data = {
            "source_lang": "hi",
            "target_lang": "en",
            "synthesize_speech": "true"
        }

        resp = client.post("/api/v1/audio/upload", files=files, data=data)
        assert resp.status_code == 200
        json_data = resp.json()

        assert json_data["success"] is True
        assert json_data["target_language"] == "en"
        assert "latency" in json_data
        assert json_data["latency"]["total_pipeline_ms"] > 0
        assert json_data["detected_emotion"] in ("neutral", "happy", "sad", "angry")
        assert json_data["detected_speaker_gender"] in ("female", "male")

    def test_api_text_translate_with_all_emotions(self):
        """Tests /api/v1/text-translate across all 4 emotions."""
        for emo in ["happy", "sad", "angry", "neutral"]:
            resp = client.post("/api/v1/text-translate", json={
                "text": "Hello, how are you today?",
                "source_lang": "en",
                "target_lang": "hi",
                "emotion": emo,
                "speaker_gender": "female",
                "synthesize_speech": True
            })
            assert resp.status_code == 200
            data = resp.json()
            assert data["success"] is True
            assert data["emotion"] == emo
            assert data["preservation_plan"] is not None
            assert data["preservation_plan"]["emotion"] == emo
            assert data["tts_status"] in ("success", "skipped")

    def test_audio_playback_streaming_endpoint(self):
        """Verifies synthesized audio is streamable from /api/v1/audio/{filename}."""
        # Create a small valid test file in TEMP_STORAGE_DIR
        test_fn = f"phase14_verify_{uuid.uuid4().hex[:8]}.wav"
        target_path = TEMP_STORAGE_DIR / test_fn
        sf.write(str(target_path), np.zeros(8000, dtype=np.float32), 16000, format="WAV", subtype="PCM_16")

        try:
            resp = client.get(f"/api/v1/audio/{test_fn}")
            assert resp.status_code == 200
            assert resp.headers["content-type"].startswith("audio/")
            assert len(resp.content) > 44
        finally:
            target_path.unlink(missing_ok=True)
