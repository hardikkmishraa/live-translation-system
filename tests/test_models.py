import numpy as np
import pytest
from backend.app.models.ser import SpeechEmotionRecognizer
from backend.app.models.nmt import BilingualNMTModel
from backend.app.models.emotion_preservation import LatentDepthEmotionPreserver
from backend.app.models.tts import EmotionPreservingTTS

def test_ser_prediction_shape_and_probabilities():
    ser = SpeechEmotionRecognizer()
    sr = 16000
    t = np.linspace(0, 0.5, int(sr * 0.5), endpoint=False)
    # High energy audio
    audio = (0.7 * np.sin(2 * np.pi * 350 * t)).astype(np.float32)

    result = ser.predict_emotion(audio, sample_rate=sr)
    assert result.top_emotion in ser.NATIVE_LABELS
    assert result.details["canonical_6_emotion"] in ser.CANONICAL_6_LABELS
    assert len(result.probabilities) == 6
    assert np.isclose(sum(result.probabilities.values()), 1.0, atol=0.05)
    assert 0.0 <= result.valence <= 1.0
    assert 0.0 <= result.arousal <= 1.0
    assert result.latency_ms >= 0.0

def test_latent_depth_emotion_preservation_eq8():
    preserver = LatentDepthEmotionPreserver()
    ser = SpeechEmotionRecognizer()
    sr = 16000
    audio = np.zeros(sr, dtype=np.float32)
    ser_res = ser.predict_emotion(audio, sample_rate=sr)

    # Force happy
    ser_res.top_emotion = "happy"
    ser_res.arousal = 0.8
    ser_res.valence = 0.7

    plan = preserver.compute_preservation_plan(ser_res, target_lang="hi")
    assert plan.emotion == "happy"
    assert plan.pitch_multiplier > 1.05
    assert plan.rate_multiplier > 1.02
    assert 0.0 <= plan.gating_weight_z <= 1.0

def test_bilingual_nmt_translation():
    nmt = BilingualNMTModel()
    # English to Hindi
    res_en_hi = nmt.translate("Hello", source_lang="en", target_lang="hi")
    assert res_en_hi.translated_text != ""
    assert res_en_hi.target_lang == "hi"

    # Hindi to English
    res_hi_en = nmt.translate("नमस्ते", source_lang="hi", target_lang="en")
    assert res_hi_en.translated_text != ""
    assert res_hi_en.target_lang == "en"

def test_tts_synthesis():
    tts = EmotionPreservingTTS()
    preserver = LatentDepthEmotionPreserver()
    ser = SpeechEmotionRecognizer()
    ser_res = ser.predict_emotion(np.zeros(1600, dtype=np.float32), 16000)
    plan = preserver.compute_preservation_plan(ser_res, target_lang="en")

    tts_res = tts.synthesize("Hello world", target_lang="en", preservation_plan=plan)
    assert tts_res.latency_ms >= 0.0
    assert tts_res.audio_format == "wav"
    if tts_res.audio_bytes:
        assert tts._has_native
        assert tts_res.duration_seconds > 0.0
    else:
        assert tts_res.details.get("error")
