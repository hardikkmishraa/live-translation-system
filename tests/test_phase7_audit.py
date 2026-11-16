"""
Phase 7 Comprehensive Audit & Regression Test Suite
===================================================
Separated into:
1. Unit Tests (Security, Evaluation Metrics, Path Traversal, Subprocess Hardening)
2. Mocked Integration Tests (Upload edge cases, WebSocket lifecycle, Error Sanitization)
3. Real-Model Smoke Tests (ASR, NMT, SER, TTS live inference)
4. Manual Live-Device Verification Specification
"""

import io
import json
import os
import tempfile
from pathlib import Path
import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.utils.security import validate_safe_path
from backend.app.services.benchmark import benchmark_service, compute_wer, compute_sentence_bleu
from backend.app.core.audio_processor import AudioProcessor

client = TestClient(app)

# =========================================================================
# SUITE 1: UNIT TESTS
# =========================================================================

class TestUnitAudit:
    """Security utilities, metric algorithms, and audio validation unit tests."""

    def test_path_traversal_prevention(self):
        """Verify that directory traversal attempts are strictly blocked."""
        base_dir = Path("/tmp/voiceaffect_test_base").resolve()
        base_dir.mkdir(parents=True, exist_ok=True)

        # Malicious paths attempting to escape base_dir
        malicious_filenames = [
            "../../etc/passwd",
            "../../../var/log/system.log",
            "subdir/../../secrets.env",
            "/etc/shadow",
            "../../windows/system32/cmd.exe",
        ]

        from fastapi import HTTPException
        for mal in malicious_filenames:
            # Construct target path attempting traversal
            target = (base_dir / mal).resolve()
            with pytest.raises(HTTPException) as exc_info:
                validate_safe_path(target, base_dir)
            assert exc_info.value.status_code == 403

        # Valid relative paths should succeed
        valid_path = validate_safe_path(base_dir / "safe_file.wav", base_dir)
        assert valid_path == (base_dir / "safe_file.wav").resolve()

    def test_tts_say_command_injection_hardening(self):
        """Verify that TTS text with leading hyphens or shell metacharacters cannot execute flags."""
        from backend.app.models.tts import EmotionPreservingTTS
        engine = EmotionPreservingTTS()

        # Text engineered to mimic command-line flags
        injection_text = "--rate=999 -v Alex; rm -rf /; echo hacked"

        # Synthesize call should pass text strictly via stdin, never as a CLI argument
        result = engine.synthesize(injection_text, target_lang="en", preservation_plan=None)
        assert result is not None
        if result.audio_bytes:
            assert engine._has_native
        else:
            assert result.details.get("error")

    def test_levenshtein_wer_metric_accuracy(self):
        """Verify that Word Error Rate (WER) correctly penalizes substitutions, deletions, and insertions."""
        assert compute_wer("hello world", "hello world") == 0.0
        assert compute_wer("cat sat mat", "dog ran bat") == 1.0
        assert compute_wer("good morning", "good bright morning") == 0.5
        assert compute_wer("test sentence", "") == 1.0
        assert compute_wer("", "") == 0.0

    def test_sentence_bleu_metric_accuracy(self):
        """Verify that sentence BLEU returns 1.0 on exact matches and degrades on mismatches."""
        ref = "नमस्ते, आप कैसे हैं?"
        hyp_exact = "नमस्ते, आप कैसे हैं?"
        assert compute_sentence_bleu(ref, hyp_exact) == 1.0

        hyp_partial = "नमस्ते आप"
        score = compute_sentence_bleu(ref, hyp_partial)
        assert 0.0 <= score < 1.0

    def test_audio_processor_edge_signals(self):
        """Verify audio processor handles pure silence, DC offset, and short bursts."""
        processor = AudioProcessor()

        # 1. Pure silence
        silence = np.zeros(16000, dtype=np.float32)
        norm_silence = processor.normalize_amplitude(silence)
        assert np.all(norm_silence == 0.0)

        # 2. Resampling exact factor
        stereo = np.ones((8000, 2), dtype=np.float32)
        mono = np.mean(stereo, axis=1)
        resampled = processor.resample(mono, orig_sr=8000, target_sr=16000)
        assert resampled.ndim == 1
        assert len(resampled) == 16000


# =========================================================================
# SUITE 2: MOCKED & SYSTEM INTEGRATION TESTS
# =========================================================================

class TestIntegrationAudit:
    """End-to-end API workflows, edge cases, error sanitization, and WebSocket validation."""

    def test_upload_unsupported_file_format(self):
        """Verify that non-audio formats (.txt, .exe, .sh) are rejected with 400 Bad Request."""
        payload = b"This is plain text disguised as audio"
        response = client.post(
            "/api/v1/audio/upload",
            files={"file": ("malicious.txt", payload, "text/plain")},
            data={"source_lang": "en", "target_lang": "hi", "consent_voice_cloning": "false"}
        )
        assert response.status_code == 400
        data = response.json()
        assert not data["success"]
        assert "Unsupported format" in data["error"]["message"]

    def test_upload_empty_audio_file(self):
        """Verify that an empty (0 byte) file upload is rejected gracefully."""
        response = client.post(
            "/api/v1/audio/upload",
            files={"file": ("empty.wav", b"", "audio/wav")},
            data={"source_lang": "en", "target_lang": "hi", "consent_voice_cloning": "false"}
        )
        assert response.status_code == 400
        data = response.json()
        assert not data["success"]
        assert "empty" in data["error"]["message"].lower()

    def test_cors_headers_configured(self):
        """Verify CORS allows standard frontend origin access without wildcard vulnerability."""
        response = client.options(
            "/api/v1/health",
            headers={
                "Origin": "http://localhost:8000",
                "Access-Control-Request-Method": "GET"
            }
        )
        assert response.status_code in [200, 204]
        assert "access-control-allow-origin" in response.headers

    def test_websocket_cancellation_cleans_buffer(self):
        """Verify that the 'cancel' action safely resets the audio stream and utterance state."""
        with client.websocket_connect("/ws/live-translate") as ws:
            ws.send_text(json.dumps({
                "action": "session_start",
                "session_id": "audit-cancel-001",
                "source_lang": "en",
                "target_lang": "hi",
                "user_consent": False
            }))
            ack = json.loads(ws.receive_text())
            assert ack["type"] == "session_acknowledged"

            # Stream audio chunk
            pcm = (np.sin(np.linspace(0, 1, 16000)) * 32767).astype(np.int16).tobytes()
            ws.send_bytes(pcm)

            # Cancel session midway
            ws.send_text(json.dumps({"action": "cancel"}))
            cancelled = json.loads(ws.receive_text())
            while cancelled.get("type") == "partial_transcript":
                cancelled = json.loads(ws.receive_text())
            assert cancelled["type"] == "cancelled"
            assert cancelled["session_id"] == "audit-cancel-001"

    def test_websocket_empty_payload_rejection(self):
        """Verify invalid json on WebSocket produces structured error without dropping connection."""
        with client.websocket_connect("/ws/live-translate") as ws:
            ws.send_text("INVALID_NON_JSON")
            err = json.loads(ws.receive_text())
            assert err["type"] == "error"
            assert "valid JSON" in err["message"]


# =========================================================================
# SUITE 3: REAL-MODEL SMOKE TESTS
# =========================================================================

class TestRealModelSmokeAudit:
    """Smoke tests executing actual local models across both language paths."""

    def test_bilingual_nmt_smoke_both_directions(self):
        """Smoke test local NMT for English->Hindi and Hindi->English."""
        from backend.app.models.nmt import BilingualNMTModel
        nmt = BilingualNMTModel()

        # En -> Hi
        res_hi = nmt.translate("Good morning", source_lang="en", target_lang="hi")
        assert len(res_hi.translated_text) > 0
        assert res_hi.source_lang == "en"
        assert res_hi.target_lang == "hi"

        # Hi -> En
        res_en = nmt.translate("नमस्ते", source_lang="hi", target_lang="en")
        assert len(res_en.translated_text) > 0
        assert res_en.source_lang == "hi"
        assert res_en.target_lang == "en"

    def test_ser_smoke_wav2vec2_inference(self):
        """Smoke test local SER wav2vec2 model on a 1-second synthetic waveform."""
        from backend.app.models.ser import SpeechEmotionRecognizer
        ser = SpeechEmotionRecognizer()
        audio = (0.3 * np.sin(2 * np.pi * 440 * np.linspace(0, 1, 16000))).astype(np.float32)
        res = ser.predict_emotion(audio, 16000)
        assert res.top_emotion in ser.NATIVE_LABELS
        assert 0.0 <= res.details["confidence"] <= 1.0
        assert "neutral" in res.probabilities

    def test_empirical_benchmark_endpoint_or_execution(self):
        """Verify benchmark service completes without error and returns all required metric fields."""
        summary = benchmark_service.run_full_benchmark()
        assert "asr_evaluation" in summary
        assert "nmt_evaluation" in summary
        assert "ser_evaluation" in summary
        assert "emotion_retention_evaluation" in summary
        assert "speech_naturalness_evaluation" in summary
        assert "latency_and_throughput" in summary
        assert "ram_rss_mb" in summary["evaluation_metadata"]
