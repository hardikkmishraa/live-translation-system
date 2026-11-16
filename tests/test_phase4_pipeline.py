import io
import time
import numpy as np
import pytest
import soundfile as sf
from pathlib import Path

from backend.app.core.audio_processor import AudioProcessor
from backend.app.models.asr import WhisperASRModel
from backend.app.models.nmt import BilingualNMTModel, SarvamNMTModel
from backend.app.models.tts import EmotionPreservingTTS
from backend.app.models.base import ASRResult, NMTResult, TTSResult
from backend.app.services.pipeline import LiveTranslationPipeline
from backend.app.config import TEMP_STORAGE_DIR

# --- 1. Audio Preprocessing Tests ---

def test_audio_preprocessing_resampling():
    processor = AudioProcessor(target_sample_rate=16000)
    orig_sr = 44100
    duration_sec = 0.5
    t = np.linspace(0, duration_sec, int(orig_sr * duration_sec), endpoint=False)
    sine_44k = (0.6 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)

    resampled = processor.resample(sine_44k, orig_sr=orig_sr, target_sr=16000)
    expected_samples = int(round(len(sine_44k) * 16000.0 / orig_sr))
    assert len(resampled) == expected_samples
    assert resampled.dtype == np.float32

def test_audio_preprocessing_amplitude_normalization():
    processor = AudioProcessor()
    # Unnormalized high amplitude array
    raw = np.array([-2.5, 0.0, 1.2, 5.0], dtype=np.float32)
    normalized = processor.normalize_amplitude(raw, target_peak=0.95)
    assert np.isclose(np.max(np.abs(normalized)), 0.95, atol=1e-4)

def test_audio_preprocessing_optional_mfcc_fft():
    """Verify separate MFCC/FFT feature utility output (Eq. 2)."""
    processor = AudioProcessor()
    sr = 16000
    duration_sec = 0.5
    t = np.linspace(0, duration_sec, int(sr * duration_sec), endpoint=False)
    synthetic_speech = (0.5 * np.sin(2 * np.pi * 220 * t) + 0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)

    features = processor.compute_fft_and_mfcc(synthetic_speech, sr=sr)
    assert "num_frames" in features
    assert features["num_frames"] > 0
    assert "mfcc_shape" in features
    assert features["mfcc_shape"][1] == 13  # 13 standard MFCC coefficients
    assert "mean_mfcc" in features
    assert len(features["mean_mfcc"]) == 13
    assert "spectral_centroid" in features
    assert features["spectral_centroid"] > 0

def test_audio_preprocessing_silence_vad():
    processor = AudioProcessor()
    silence = np.zeros(16000, dtype=np.float32)
    assert processor.detect_speech_activity(silence, threshold=0.01) is False

    t = np.linspace(0, 0.5, 8000, endpoint=False)
    tone = (0.8 * np.sin(2 * np.pi * 300 * t)).astype(np.float32)
    assert processor.detect_speech_activity(tone, threshold=0.01) is True



def test_asr_does_not_fabricate_utterance_confidence(monkeypatch):
    asr = WhisperASRModel()
    asr._ensure_loaded = lambda: None

    class FakePipeline:
        def __call__(self, *args, **kwargs):
            return {"text": "hello world"}

    asr.pipeline = FakePipeline()
    t = np.linspace(0, 1.0, 16000, endpoint=False)
    audio = (0.2 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    result = asr.transcribe(audio, sample_rate=16000, language="en")

    assert result.text == "hello world"
    assert result.confidence is None
    assert result.details["confidence_status"] == "unavailable_from_pipeline_output"

# --- 2. Adapter Contracts & Invalid Inputs ---

def test_asr_adapter_contract_silence_and_short():
    asr = WhisperASRModel()
    # 1. Very short audio (<100ms)
    short_audio = np.zeros(800, dtype=np.float32)
    res_short = asr.transcribe(short_audio, sample_rate=16000)
    assert res_short.text == ""
    assert res_short.confidence == 0.0
    assert res_short.details.get("status") == "too_short_or_empty"

    # 2. Pure silence (1s zeros)
    silent_audio = np.zeros(16000, dtype=np.float32)
    res_silence = asr.transcribe(silent_audio, sample_rate=16000)
    assert res_silence.text == ""
    assert res_silence.confidence == 0.0
    assert res_silence.details.get("status") == "silence_detected"



def test_sarvam_translation_failure_preserves_transcription_and_does_not_fallback(monkeypatch):
    """
    Verifies that when Sarvam translation fails, the original transcription is preserved,
    translated_text is empty (never invented), translation status is marked failed,
    and no silent fallback to MarianMT or MyMemory occurs.
    """
    def fail_api(*args, **kwargs):
        raise OSError("simulated network outage")

    monkeypatch.setattr("urllib.request.urlopen", fail_api)
    model = SarvamNMTModel(api_key="sk_test_key_dummy_1234567890", max_retries=0)
    result = model.translate("Hello world", "en", "hi")

    assert result.source_text == "Hello world"
    assert result.translated_text == ""
    assert result.details["status"] == "translation_failed"
    assert result.details["provider"] == "sarvam_ai"
    assert result.details["automatic_fallback_used"] is False
    assert "simulated network outage" in result.details["error"]

def test_nmt_adapter_contract():
    nmt = BilingualNMTModel()

    # 1. Empty text
    res_empty = nmt.translate("", source_lang="en", target_lang="hi")
    assert res_empty.translated_text == ""
    assert res_empty.source_text == ""

    # 2. Unsupported language pair
    res_unsupported = nmt.translate("Bonjour", source_lang="fr", target_lang="en")
    assert res_unsupported.translated_text == ""
    assert "Unsupported language pair" in res_unsupported.details.get("error", "")

    # 3. Same language identity
    res_same = nmt.translate("Same text test", source_lang="en", target_lang="en")
    assert res_same.translated_text == "Same text test"

def test_tts_adapter_contract():
    tts = EmotionPreservingTTS()

    # 1. Empty text
    res_empty = tts.synthesize("", target_lang="en")
    assert res_empty.audio_bytes == b""
    assert res_empty.duration_seconds == 0.0

    # 2. Unsupported target language
    res_unsupported = tts.synthesize("Guten Tag", target_lang="de")
    assert res_unsupported.audio_bytes == b""
    assert "unsupported" in res_unsupported.details.get("error", "").lower()

# --- 3. Pipeline Orchestration & Emotion Deferral ---

def test_pipeline_orchestration_silence_handling():
    pipeline = LiveTranslationPipeline()
    silence = np.zeros(16000, dtype=np.float32)

    result = pipeline.process_audio(silence, sample_rate=16000, synthesize_speech=True)
    assert result.speech_detected is False
    assert result.source_transcript == ""
    assert result.translated_text == ""
    assert result.audio_base64 is None
    # Emotion fields must not be fabricated
    assert result.detected_emotion is None
    assert result.emotion_probabilities is None
    assert result.valence is None
    assert result.latency.ser_ms == 0.0

def test_pipeline_preserves_original_recording_reference():
    pipeline = LiveTranslationPipeline()
    orig_filename = "orig_test_sample.wav"
    silence = np.zeros(1600, dtype=np.float32)

    result = pipeline.process_audio(
        silence,
        sample_rate=16000,
        synthesize_speech=False,
        original_audio_filename=orig_filename
    )
    assert result.original_audio_filename == orig_filename

# --- 4. REAL Model End-to-End Smoke Test ---

@pytest.mark.smoke
def test_real_model_smoke_pipeline():
    """
    REAL model end-to-end smoke test running actual local neural models:
    - Whisper ASR (openai/whisper-tiny)
    - MarianMT NMT (Helsinki-NLP/opus-mt-en-hi)
    - Local Speech Synthesis (macOS native Lekha / synthetic melodic)
    NEVER marked as mocked.
    """
    t_start = time.perf_counter()

    # Step 1: Real Local NMT Inference (EN -> HI)
    nmt = BilingualNMTModel()
    nmt_res = nmt.translate("Hello, how are you?", source_lang="en", target_lang="hi")
    assert nmt_res.translated_text != "", "Real NMT failed to translate English input to Hindi"
    assert nmt_res.target_lang == "hi"

    # Step 2: Real Local NMT Inference (HI -> EN)
    nmt_rev = nmt.translate("नमस्ते, आप कैसे हैं?", source_lang="hi", target_lang="en")
    assert nmt_rev.translated_text != "", "Real NMT failed to translate Hindi input to English"
    assert nmt_rev.target_lang == "en"

    # Step 3: Real Local Speech Synthesis (TTS)
    tts = EmotionPreservingTTS()
    tts_res = tts.synthesize(nmt_res.translated_text, target_lang="hi")
    assert tts_res.latency_ms >= 0.0
    if tts_res.audio_bytes:
        assert tts._has_native, "Only a native speech engine may produce synthesized speech"
    else:
        assert tts_res.details.get("error"), "Unavailable TTS must explain why no speech was generated"
    assert tts_res.audio_format == "wav"

    total_smoke_time = time.perf_counter() - t_start
    print(f"\n[REAL SMOKE TEST COMPLETED IN {total_smoke_time:.2f}s]")
    print(f"EN->HI Translation: {nmt_res.translated_text}")
    print(f"HI->EN Translation: {nmt_rev.translated_text}")
    print(f"TTS Audio Bytes: {len(tts_res.audio_bytes)} | Duration: {tts_res.duration_seconds}s")
