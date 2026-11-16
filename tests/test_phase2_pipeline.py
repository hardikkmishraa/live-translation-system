import io
import time
import uuid
from unittest.mock import MagicMock, patch
import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.config import SARVAM_API_KEY
from backend.app.services.pipeline import LiveTranslationPipeline
from backend.app.services.model_lifecycle import model_lifecycle
from backend.app.models.base import ASRResult, SERResult, NMTResult, TTSResult
from backend.app.services.sarvam_translation import (
    SarvamAuthError,
    SarvamTimeoutError,
    SarvamResponseError,
    SarvamTranslationService
)

client = TestClient(app)

def create_synthetic_wav(duration_sec: float = 1.0, sr: int = 16000) -> bytes:
    """Generates synthetic audio waveform as PCM WAV bytes."""
    t = np.linspace(0, duration_sec, int(sr * duration_sec), endpoint=False)
    waveform = (0.5 * np.sin(2 * np.pi * 220 * t) + 0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    buf = io.BytesIO()
    sf.write(buf, waveform, sr, format="WAV", subtype="PCM_16")
    buf.seek(0)
    return buf.read()

def create_synthetic_waveform(duration_sec: float = 1.0, sr: int = 16000) -> np.ndarray:
    t = np.linspace(0, duration_sec, int(sr * duration_sec), endpoint=False)
    return (0.5 * np.sin(2 * np.pi * 220 * t) + 0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)


# =========================================================================
# Phase 2 End-to-End Tests
# =========================================================================

def test_successful_hindi_to_english_workflow():
    """Verify full pipeline: Audio In -> VAD -> Whisper ASR -> SER -> Sarvam NMT -> Prosody -> TTS."""
    pipeline = LiveTranslationPipeline()
    audio = create_synthetic_waveform(1.0)

    mock_asr = MagicMock()
    mock_asr.transcribe.return_value = ASRResult(
        text="नमस्ते, आप कैसे हैं?",
        language="hi",
        confidence=None,
        latency_ms=120.0,
        model_name="openai/whisper-tiny"
    )

    mock_ser = MagicMock()
    mock_ser.predict_emotion.return_value = SERResult(
        top_emotion="happy",
        probabilities={"happy": 0.85, "neutral": 0.15},
        valence=0.85,
        arousal=0.70,
        dominance=0.65,
        latency_ms=80.0,
        model_name="ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition",
        details={"confidence": 0.85, "language_limitation_warning": "Cross-lingual prosody only."}
    )

    mock_nmt = MagicMock()
    mock_nmt.translate.return_value = NMTResult(
        source_text="नमस्ते, आप कैसे हैं?",
        translated_text="Hello, how are you?",
        source_lang="hi",
        target_lang="en",
        latency_ms=250.0,
        model_name="mayura:v1",
        details={"provider": "sarvam", "status": "success", "model": "mayura:v1"}
    )

    mock_tts = MagicMock()
    mock_tts.synthesize.return_value = TTSResult(
        audio_bytes=b"RIFF_FAKE_WAV_BYTES_DATA",
        audio_format="wav",
        sample_rate=16000,
        duration_seconds=1.2,
        voice_used="Samantha",
        latency_ms=150.0,
        model_name="macos_say_Samantha",
        emotion_applied="happy",
        details={"audio_filename": "tts_synthetic_test.wav"}
    )

    with patch.object(model_lifecycle, "get_asr", return_value=mock_asr), \
         patch.object(model_lifecycle, "get_ser", return_value=mock_ser), \
         patch.object(model_lifecycle, "get_nmt", return_value=mock_nmt), \
         patch.object(model_lifecycle, "get_tts", return_value=mock_tts):

        result = pipeline.process_audio(
            audio=audio,
            sample_rate=16000,
            forced_source_lang="hi",
            forced_target_lang="en",
            synthesize_speech=True
        )

        assert result.request_id is not None
        assert result.speech_detected is True
        assert result.source_language == "hi"
        assert result.target_language == "en"
        assert result.source_transcript == "नमस्ते, आप कैसे हैं?"
        assert result.translated_text == "Hello, how are you?"
        assert result.asr_status == "success"
        assert result.ser_status == "success"
        assert result.detected_emotion == "happy"
        assert result.translation_provider == "sarvam"
        assert result.translation_status == "success"
        assert result.tts_status == "success"
        assert result.audio_base64 is not None
        assert result.audio_output_reference == "/api/v1/audio/tts_synthetic_test.wav"
        assert result.total_processing_duration > 0
        assert len(result.errors) == 0


def test_successful_english_to_hindi_workflow():
    """Verify reverse workflow English to Hindi with Sarvam AI."""
    pipeline = LiveTranslationPipeline()
    audio = create_synthetic_waveform(1.0)

    mock_asr = MagicMock()
    mock_asr.transcribe.return_value = ASRResult(
        text="Good morning, have a wonderful day.",
        language="en",
        confidence=None,
        latency_ms=110.0,
        model_name="openai/whisper-tiny"
    )

    mock_ser = MagicMock()
    mock_ser.predict_emotion.return_value = SERResult(
        top_emotion="neutral",
        probabilities={"neutral": 0.90, "calm": 0.10},
        valence=0.50,
        arousal=0.40,
        dominance=0.50,
        latency_ms=75.0,
        model_name="ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition",
        details={"confidence": 0.90}
    )

    mock_nmt = MagicMock()
    mock_nmt.translate.return_value = NMTResult(
        source_text="Good morning, have a wonderful day.",
        translated_text="शुभ प्रभात, आपका दिन मंगलमय हो।",
        source_lang="en",
        target_lang="hi",
        latency_ms=260.0,
        model_name="mayura:v1",
        details={"provider": "sarvam", "status": "success", "model": "mayura:v1"}
    )

    mock_tts = MagicMock()
    mock_tts.synthesize.return_value = TTSResult(
        audio_bytes=b"RIFF_FAKE_HI_WAV",
        audio_format="wav",
        sample_rate=16000,
        duration_seconds=1.4,
        voice_used="Lekha",
        latency_ms=160.0,
        model_name="macos_say_Lekha",
        emotion_applied="neutral",
        details={"audio_filename": "tts_hi_test.wav"}
    )

    with patch.object(model_lifecycle, "get_asr", return_value=mock_asr), \
         patch.object(model_lifecycle, "get_ser", return_value=mock_ser), \
         patch.object(model_lifecycle, "get_nmt", return_value=mock_nmt), \
         patch.object(model_lifecycle, "get_tts", return_value=mock_tts):

        result = pipeline.process_audio(
            audio=audio,
            sample_rate=16000,
            forced_source_lang="en",
            forced_target_lang="hi",
            synthesize_speech=True
        )

        assert result.source_language == "en"
        assert result.target_language == "hi"
        assert result.source_transcript == "Good morning, have a wonderful day."
        assert result.translated_text == "शुभ प्रभात, आपका दिन मंगलमय हो।"
        assert result.translation_status == "success"
        assert result.tts_status == "success"
        assert result.synthesis_voice_used == "Lekha"


def test_missing_audio():
    """Verify endpoint rejects empty or missing audio file."""
    res = client.post(
        "/api/v1/audio/upload",
        files={"file": ("empty.wav", b"", "audio/wav")}
    )
    assert res.status_code == 400
    data = res.json()
    err_msg = data.get("error", {}).get("message", "") or data.get("detail", "")
    assert "empty" in err_msg.lower()


def test_silent_audio():
    """Verify that pure silence triggers VAD silence gate and does not run models."""
    pipeline = LiveTranslationPipeline()
    silence = np.zeros(16000, dtype=np.float32)

    result = pipeline.process_audio(silence, sample_rate=16000)

    assert result.speech_detected is False
    assert result.asr_status == "no_speech_detected"
    assert result.ser_status == "skipped_silence"
    assert result.translation_status == "noop"
    assert result.tts_status == "skipped"
    assert result.source_transcript == ""
    assert result.translated_text == ""
    assert any("silence gate" in w.lower() for w in result.warnings)


def test_invalid_audio_corrupted():
    """Verify corrupted non-audio bytes are rejected cleanly with 400 Bad Request."""
    corrupted_data = b"NOT_A_VALID_AUDIO_HEADER_CORRUPTED_STREAM_DATA" * 50
    res = client.post(
        "/api/v1/audio/upload",
        files={"file": ("corrupted.wav", corrupted_data, "audio/wav")}
    )
    assert res.status_code == 400
    data = res.json()
    err_msg = data.get("error", {}).get("message", "") or data.get("detail", "")
    assert "corrupted" in err_msg.lower() or "invalid" in err_msg.lower() or "failed" in err_msg.lower()


def test_sarvam_authentication_failure():
    """Verify that Sarvam auth error preserves transcript, marks translation failed, and does not crash."""
    pipeline = LiveTranslationPipeline()
    audio = create_synthetic_waveform(1.0)

    mock_asr = MagicMock()
    mock_asr.transcribe.return_value = ASRResult(
        text="नमस्ते", language="hi", confidence=None, latency_ms=100.0, model_name="whisper-tiny"
    )

    mock_ser = MagicMock()
    mock_ser.predict_emotion.return_value = SERResult(
        top_emotion="neutral", probabilities={"neutral": 1.0}, valence=0.5, arousal=0.5, dominance=0.5,
        latency_ms=50.0, model_name="wav2vec2", details={"confidence": 1.0}
    )

    mock_nmt = MagicMock()
    mock_nmt.translate.return_value = NMTResult(
        source_text="नमस्ते",
        translated_text="",
        source_lang="hi",
        target_lang="en",
        latency_ms=50.0,
        model_name="sarvam_auth_error",
        details={"status": "translation_failed", "error": "Sarvam API authentication failed (HTTP 401). Invalid key.", "provider": "sarvam"}
    )

    with patch.object(model_lifecycle, "get_asr", return_value=mock_asr), \
         patch.object(model_lifecycle, "get_ser", return_value=mock_ser), \
         patch.object(model_lifecycle, "get_nmt", return_value=mock_nmt):

        result = pipeline.process_audio(audio, forced_source_lang="hi", forced_target_lang="en")

        # Original transcript preserved
        assert result.source_transcript == "नमस्ते"
        # Zero fabricated translation
        assert result.translated_text == ""
        assert result.translation_status == "failed"
        assert result.tts_status == "skipped"
        assert any("authentication failed" in e.lower() for e in result.errors)


def test_sarvam_timeout():
    """Verify timeout preserves transcript and reports error."""
    pipeline = LiveTranslationPipeline()
    audio = create_synthetic_waveform(1.0)

    mock_asr = MagicMock()
    mock_asr.transcribe.return_value = ASRResult(
        text="Hello world", language="en", confidence=None, latency_ms=100.0, model_name="whisper-tiny"
    )
    mock_ser = MagicMock()
    mock_ser.predict_emotion.return_value = SERResult(
        top_emotion="neutral", probabilities={"neutral": 1.0}, valence=0.5, arousal=0.5, dominance=0.5,
        latency_ms=50.0, model_name="wav2vec2", details={"confidence": 1.0}
    )
    mock_nmt = MagicMock()
    mock_nmt.translate.return_value = NMTResult(
        source_text="Hello world",
        translated_text="",
        source_lang="en",
        target_lang="hi",
        latency_ms=5000.0,
        model_name="sarvam_timeout",
        details={"status": "translation_failed", "error": "Sarvam API request timed out after 10.0s.", "provider": "sarvam"}
    )

    with patch.object(model_lifecycle, "get_asr", return_value=mock_asr), \
         patch.object(model_lifecycle, "get_ser", return_value=mock_ser), \
         patch.object(model_lifecycle, "get_nmt", return_value=mock_nmt):

        result = pipeline.process_audio(audio, forced_source_lang="en", forced_target_lang="hi")

        assert result.source_transcript == "Hello world"
        assert result.translated_text == ""
        assert result.translation_status == "failed"
        assert any("timed out" in e.lower() for e in result.errors)


def test_sarvam_invalid_response():
    """Verify invalid or malformed response from Sarvam is handled safely."""
    pipeline = LiveTranslationPipeline()
    audio = create_synthetic_waveform(1.0)

    mock_asr = MagicMock()
    mock_asr.transcribe.return_value = ASRResult(
        text="Hello", language="en", confidence=None, latency_ms=100.0, model_name="whisper-tiny"
    )
    mock_ser = MagicMock()
    mock_ser.predict_emotion.return_value = SERResult(
        top_emotion="neutral", probabilities={"neutral": 1.0}, valence=0.5, arousal=0.5, dominance=0.5,
        latency_ms=50.0, model_name="wav2vec2", details={"confidence": 1.0}
    )
    mock_nmt = MagicMock()
    mock_nmt.translate.return_value = NMTResult(
        source_text="Hello",
        translated_text="",
        source_lang="en",
        target_lang="hi",
        latency_ms=100.0,
        model_name="sarvam_response_error",
        details={"status": "translation_failed", "error": "Sarvam API returned malformed response.", "provider": "sarvam"}
    )

    with patch.object(model_lifecycle, "get_asr", return_value=mock_asr), \
         patch.object(model_lifecycle, "get_ser", return_value=mock_ser), \
         patch.object(model_lifecycle, "get_nmt", return_value=mock_nmt):

        result = pipeline.process_audio(audio, forced_source_lang="en", forced_target_lang="hi")
        assert result.source_transcript == "Hello"
        assert result.translated_text == ""
        assert result.translation_status == "failed"


def test_missing_ser_checkpoint():
    """Verify that when SER checkpoint is missing or fails, translation proceeds without blocking."""
    pipeline = LiveTranslationPipeline()
    audio = create_synthetic_waveform(1.0)

    mock_asr = MagicMock()
    mock_asr.transcribe.return_value = ASRResult(
        text="Good morning", language="en", confidence=None, latency_ms=100.0, model_name="whisper-tiny"
    )
    mock_ser = MagicMock()
    mock_ser.predict_emotion.side_effect = RuntimeError("SER checkpoint weights not found in local cache.")

    mock_nmt = MagicMock()
    mock_nmt.translate.return_value = NMTResult(
        source_text="Good morning",
        translated_text="शुभ प्रभात",
        source_lang="en",
        target_lang="hi",
        latency_ms=200.0,
        model_name="mayura:v1",
        details={"status": "success", "provider": "sarvam"}
    )

    with patch.object(model_lifecycle, "get_asr", return_value=mock_asr), \
         patch.object(model_lifecycle, "get_ser", return_value=mock_ser), \
         patch.object(model_lifecycle, "get_nmt", return_value=mock_nmt):

        result = pipeline.process_audio(audio, forced_source_lang="en", forced_target_lang="hi", synthesize_speech=False)

        # SER failed
        assert result.ser_status == "failed"
        assert result.detected_emotion == "unavailable"
        assert any("ser" in w.lower() for w in result.warnings)

        # Translation succeeded despite SER failure
        assert result.source_transcript == "Good morning"
        assert result.translated_text == "शुभ प्रभात"
        assert result.translation_status == "success"


def test_tts_unavailable():
    """Verify that if TTS is unavailable, translation is still returned."""
    pipeline = LiveTranslationPipeline()
    audio = create_synthetic_waveform(1.0)

    mock_asr = MagicMock()
    mock_asr.transcribe.return_value = ASRResult(
        text="Hello", language="en", confidence=None, latency_ms=100.0, model_name="whisper-tiny"
    )
    mock_ser = MagicMock()
    mock_ser.predict_emotion.return_value = SERResult(
        top_emotion="neutral", probabilities={"neutral": 1.0}, valence=0.5, arousal=0.5, dominance=0.5,
        latency_ms=50.0, model_name="wav2vec2", details={"confidence": 1.0}
    )
    mock_nmt = MagicMock()
    mock_nmt.translate.return_value = NMTResult(
        source_text="Hello",
        translated_text="नमस्ते",
        source_lang="en",
        target_lang="hi",
        latency_ms=200.0,
        model_name="mayura:v1",
        details={"status": "success", "provider": "sarvam"}
    )
    mock_tts = MagicMock()
    mock_tts.synthesize.return_value = TTSResult(
        audio_bytes=b"",
        audio_format="wav",
        sample_rate=16000,
        duration_seconds=0.0,
        voice_used="none",
        latency_ms=10.0,
        model_name="tts_unavailable",
        emotion_applied="neutral",
        details={"error": "macOS say command unavailable"}
    )

    with patch.object(model_lifecycle, "get_asr", return_value=mock_asr), \
         patch.object(model_lifecycle, "get_ser", return_value=mock_ser), \
         patch.object(model_lifecycle, "get_nmt", return_value=mock_nmt), \
         patch.object(model_lifecycle, "get_tts", return_value=mock_tts):

        result = pipeline.process_audio(audio, forced_source_lang="en", forced_target_lang="hi", synthesize_speech=True)

        assert result.translated_text == "नमस्ते"
        assert result.translation_status == "success"
        assert result.tts_status in ("unavailable", "failed")
        assert result.audio_base64 is None
        assert any("tts" in w.lower() for w in result.warnings)


def test_partial_pipeline_asr_failure():
    """Verify that if ASR fails, translation is NOT attempted with fabricated input."""
    pipeline = LiveTranslationPipeline()
    audio = create_synthetic_waveform(1.0)

    mock_asr = MagicMock()
    mock_asr.transcribe.side_effect = RuntimeError("Whisper CUDA/MPS memory allocation failure")

    mock_nmt = MagicMock()

    with patch.object(model_lifecycle, "get_asr", return_value=mock_asr), \
         patch.object(model_lifecycle, "get_nmt", return_value=mock_nmt):

        result = pipeline.process_audio(audio)

        assert result.asr_status == "failed"
        assert result.source_transcript == ""
        assert result.translated_text == ""
        # NMT translate must NOT have been called
        mock_nmt.translate.assert_not_called()
        assert any("asr failed" in e.lower() for e in result.errors)


def test_api_upload_endpoint_full_contract():
    """Verify the upload endpoint returns all structured fields specified in Section 7."""
    wav_bytes = create_synthetic_wav(1.0)

    mock_result = LiveTranslationPipeline().process_audio(
        audio=create_synthetic_waveform(1.0),
        sample_rate=16000
    )

    with patch.object(LiveTranslationPipeline, "process_audio") as mock_proc:
        mock_proc.return_value = mock_result
        mock_result.source_transcript = "Good day"
        mock_result.translated_text = "शुभ दिन"
        mock_result.translation_status = "success"
        mock_result.translation_provider = "sarvam"
        mock_result.translation_model = "mayura:v1"
        mock_result.tts_engine = "macos_native"
        mock_result.tts_voice = "Lekha"
        mock_result.tts_status = "success"

        res = client.post(
            "/api/v1/audio/upload",
            files={"file": ("speech.wav", wav_bytes, "audio/wav")},
            data={"source_lang": "en", "target_lang": "hi"}
        )

        assert res.status_code == 200
        data = res.json()

        # All Section 7 fields present
        assert "request_id" in data
        assert "source_language" in data
        assert "target_language" in data
        assert "source_transcript" in data
        assert "translated_text" in data
        assert "asr_status" in data
        assert "ser_status" in data
        assert "detected_emotion" in data
        assert "emotion_scores" in data
        assert "asr_confidence" in data
        assert "translation_provider" in data
        assert "translation_model" in data
        assert "translation_status" in data
        assert "tts_engine" in data
        assert "tts_voice" in data
        assert "tts_status" in data
        assert "audio_output_reference" in data
        assert "total_processing_duration" in data
        assert "warnings" in data
        assert "errors" in data
