import numpy as np
import pytest
from backend.app.core.audio_processor import AudioProcessor
from backend.app.models.emotion_preservation import LatentDepthEmotionPreserver
from backend.app.models.tts import EmotionPreservingTTS
from backend.app.models.base import SERResult
from backend.app.config import (
    TTS_VOICE_HI_FEMALE,
    TTS_VOICE_HI_MALE,
    TTS_VOICE_EN_FEMALE,
    TTS_VOICE_EN_MALE
)

def generate_tone(freq: float, duration_s: float = 1.0, sr: int = 16000) -> np.ndarray:
    t = np.linspace(0, duration_s, int(sr * duration_s), endpoint=False)
    # Sine wave with harmonics for richer speech-like acoustic pitch
    audio = 0.6 * np.sin(2 * np.pi * freq * t) + 0.3 * np.sin(2 * np.pi * 2 * freq * t)
    return audio.astype(np.float32)

class TestSpeakerGenderDetection:
    def test_male_pitch_detection(self):
        proc = AudioProcessor()
        # 125 Hz is prototypical adult male fundamental frequency
        audio = generate_tone(125.0, duration_s=1.2)
        res = proc.detect_speaker_gender(audio, sr=16000)
        assert res["gender"] == "male", f"Expected male, got {res}"
        assert abs(res["f0_median_hz"] - 125.0) < 5.0
        assert res["confidence"] >= 0.70

    def test_female_pitch_detection(self):
        proc = AudioProcessor()
        # 220 Hz is prototypical adult female fundamental frequency
        audio = generate_tone(220.0, duration_s=1.2)
        res = proc.detect_speaker_gender(audio, sr=16000)
        assert res["gender"] == "female", f"Expected female, got {res}"
        assert abs(res["f0_median_hz"] - 220.0) < 5.0
        assert res["confidence"] >= 0.70

class TestGenderVoiceRouting:
    @pytest.fixture
    def mock_ser(self):
        return SERResult(
            top_emotion="neutral",
            probabilities={"neutral": 1.0},
            valence=0.5,
            arousal=0.5,
            dominance=0.5,
            latency_ms=10.0,
            model_name="mock_ser",
            details={"confidence": 0.9}
        )

    def test_male_hindi_voice_selection(self, mock_ser):
        preserver = LatentDepthEmotionPreserver()
        plan = preserver.compute_preservation_plan(
            ser_result=mock_ser,
            target_lang="hi",
            speaker_gender="male"
        )
        assert plan.synthesis_voice_used == TTS_VOICE_HI_MALE
        assert plan.detected_gender == "male"

    def test_female_hindi_voice_selection(self, mock_ser):
        preserver = LatentDepthEmotionPreserver()
        plan = preserver.compute_preservation_plan(
            ser_result=mock_ser,
            target_lang="hi",
            speaker_gender="female"
        )
        assert plan.synthesis_voice_used == TTS_VOICE_HI_FEMALE
        assert plan.detected_gender == "female"

    def test_male_english_voice_selection(self, mock_ser):
        preserver = LatentDepthEmotionPreserver()
        plan = preserver.compute_preservation_plan(
            ser_result=mock_ser,
            target_lang="en",
            speaker_gender="male"
        )
        assert plan.synthesis_voice_used == TTS_VOICE_EN_MALE
        assert plan.detected_gender == "male"

    def test_female_english_voice_selection(self, mock_ser):
        preserver = LatentDepthEmotionPreserver()
        plan = preserver.compute_preservation_plan(
            ser_result=mock_ser,
            target_lang="en",
            speaker_gender="female"
        )
        assert plan.synthesis_voice_used == TTS_VOICE_EN_FEMALE
        assert plan.detected_gender == "female"

    def test_tts_synthesizes_with_correct_voice(self, mock_ser):
        tts = EmotionPreservingTTS()
        if not tts._has_native:
            pytest.skip("Native macOS say not available in this environment")

        preserver = LatentDepthEmotionPreserver()
        
        # Test Male English Synthesis
        male_plan_en = preserver.compute_preservation_plan(mock_ser, "en", speaker_gender="male")
        res_m_en = tts.synthesize("Hello this is a male voice test.", "en", male_plan_en)
        assert res_m_en.voice_used == TTS_VOICE_EN_MALE
        assert len(res_m_en.audio_bytes) > 0

        # Test Female English Synthesis
        female_plan_en = preserver.compute_preservation_plan(mock_ser, "en", speaker_gender="female")
        res_f_en = tts.synthesize("Hello this is a female voice test.", "en", female_plan_en)
        assert res_f_en.voice_used == TTS_VOICE_EN_FEMALE
        assert len(res_f_en.audio_bytes) > 0

        # Test Male Hindi Synthesis
        male_plan_hi = preserver.compute_preservation_plan(mock_ser, "hi", speaker_gender="male")
        res_m_hi = tts.synthesize("नमस्ते, यह पुरुष आवाज का परीक्षण है।", "hi", male_plan_hi)
        assert res_m_hi.voice_used == TTS_VOICE_HI_MALE
        assert len(res_m_hi.audio_bytes) > 0

        # Test Female Hindi Synthesis
        female_plan_hi = preserver.compute_preservation_plan(mock_ser, "hi", speaker_gender="female")
        res_f_hi = tts.synthesize("नमस्ते, यह महिला आवाज का परीक्षण है।", "hi", female_plan_hi)
        assert res_f_hi.voice_used == TTS_VOICE_HI_FEMALE
        assert len(res_f_hi.audio_bytes) > 0

