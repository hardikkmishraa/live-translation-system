"""
Unit and Integration Tests for Phase 13: Emotion-Aware Speech Synthesis & Prosody Control.
========================================================================================
Covers:
- IEMOCAP 4-class emotion mapping (happy, sad, angry, neutral)
- Confidence probability gating (<0.35 gates to neutral, 0.35-0.50 blends, >=0.50 full)
- Speaker gender-to-voice persona alignment across Hindi and English
- Multi-provider fallback chain (Sarvam bulbul:v4-flash -> Edge-TTS -> macOS say)
- Truthful expressiveness reporting and schema validation
"""

import os
import unittest
from unittest.mock import patch, MagicMock
import numpy as np

from backend.app.models.base import SERResult, EmotionPreservationPlan, TTSResult
from backend.app.models.emotion_preservation import LatentDepthEmotionPreserver
from backend.app.models.tts_base import BaseTTSProvider
from backend.app.models.tts_sarvam import SarvamTTSProvider
from backend.app.models.tts_edge import EdgeTTSProvider
from backend.app.models.tts_macos import MacOSTTSProvider
from backend.app.models.tts import EmotionPreservingTTS
from backend.app.config import (
    TTS_CONFIDENCE_THRESHOLD,
    TTS_MIN_CONFIDENCE_THRESHOLD,
)


class TestEmotionPreservationPlanning(unittest.TestCase):
    """Tests the prosody planning and confidence gating service."""

    def setUp(self):
        self.preserver = LatentDepthEmotionPreserver()

    def test_iemocap_happy_prosody_mapping(self):
        """High-confidence happy prediction should elevate pitch and rate."""
        ser_res = SERResult(
            top_emotion="happy",
            probabilities={"happy": 0.85, "neutral": 0.05, "sad": 0.05, "angry": 0.05},
            valence=0.8,
            arousal=0.7,
            dominance=0.6,
            latency_ms=10.0,
            model_name="iemocap-test",
            details={"confidence": 0.85}
        )
        plan = self.preserver.compute_preservation_plan(ser_res, target_lang="hi", speaker_gender="female")
        self.assertEqual(plan.emotion, "happy")
        self.assertGreater(plan.pitch_multiplier, 1.05)
        self.assertGreater(plan.rate_multiplier, 1.05)
        self.assertTrue(plan.expressive_synthesis_supported)
        self.assertFalse(plan.speaker_similarity_retained)

    def test_iemocap_sad_prosody_mapping(self):
        """High-confidence sad prediction should lower pitch, rate, and volume."""
        ser_res = SERResult(
            top_emotion="sad",
            probabilities={"sad": 0.90, "neutral": 0.05, "happy": 0.02, "angry": 0.03},
            valence=0.2,
            arousal=0.3,
            dominance=0.3,
            latency_ms=10.0,
            model_name="iemocap-test",
            details={"confidence": 0.90}
        )
        plan = self.preserver.compute_preservation_plan(ser_res, target_lang="hi", speaker_gender="male")
        self.assertEqual(plan.emotion, "sad")
        self.assertLess(plan.pitch_multiplier, 0.95)
        self.assertLess(plan.rate_multiplier, 0.95)
        self.assertLess(plan.volume_multiplier, 0.95)

    def test_iemocap_angry_prosody_mapping(self):
        """High-confidence angry prediction should increase pitch, rate, and volume."""
        ser_res = SERResult(
            top_emotion="angry",
            probabilities={"angry": 0.80, "neutral": 0.10, "sad": 0.05, "happy": 0.05},
            valence=0.2,
            arousal=0.85,
            dominance=0.8,
            latency_ms=10.0,
            model_name="iemocap-test",
            details={"confidence": 0.80}
        )
        plan = self.preserver.compute_preservation_plan(ser_res, target_lang="en", speaker_gender="male")
        self.assertEqual(plan.emotion, "angry")
        self.assertGreater(plan.pitch_multiplier, 1.10)
        self.assertGreater(plan.rate_multiplier, 1.10)
        self.assertGreater(plan.volume_multiplier, 1.10)

    def test_confidence_gating_below_minimum(self):
        """When confidence is below 0.35, emotion should be gated to neutral baseline."""
        ser_res = SERResult(
            top_emotion="angry",
            probabilities={"angry": 0.30, "neutral": 0.28, "sad": 0.22, "happy": 0.20},
            valence=0.2,
            arousal=0.8,
            dominance=0.7,
            latency_ms=10.0,
            model_name="iemocap-test",
            details={"confidence": 0.30}
        )
        plan = self.preserver.compute_preservation_plan(ser_res, target_lang="hi", speaker_gender="female")
        self.assertEqual(plan.emotion, "neutral")
        self.assertEqual(plan.pitch_multiplier, 1.0)
        self.assertEqual(plan.rate_multiplier, 1.0)
        self.assertEqual(plan.volume_multiplier, 1.0)
        self.assertIn("Gated", plan.prosody_notes)

    def test_confidence_gating_soft_blending(self):
        """When confidence is between 0.35 and 0.50, soft blending should be applied."""
        ser_res = SERResult(
            top_emotion="happy",
            probabilities={"happy": 0.42, "neutral": 0.25, "sad": 0.18, "angry": 0.15},
            valence=0.8,
            arousal=0.7,
            dominance=0.6,
            latency_ms=10.0,
            model_name="iemocap-test",
            details={"confidence": 0.42}
        )
        plan = self.preserver.compute_preservation_plan(ser_res, target_lang="hi", speaker_gender="female")
        self.assertEqual(plan.emotion, "happy")
        # Soft blend should result in intermediate values
        self.assertGreater(plan.pitch_multiplier, 1.0)
        self.assertLess(plan.pitch_multiplier, 1.16)
        self.assertIn("Soft blended", plan.prosody_notes)

    def test_speaker_gender_voice_selection(self):
        """Female vs male speaker gender should select respective personas."""
        ser_res = SERResult(
            top_emotion="happy",
            probabilities={"happy": 0.85},
            valence=0.8, arousal=0.7, dominance=0.6, latency_ms=10.0,
            model_name="iemocap-test", details={"confidence": 0.85}
        )
        plan_female = self.preserver.compute_preservation_plan(ser_res, target_lang="hi", speaker_gender="female")
        plan_male = self.preserver.compute_preservation_plan(ser_res, target_lang="hi", speaker_gender="male")

        self.assertNotEqual(plan_female.synthesis_voice_used, plan_male.synthesis_voice_used)
        self.assertEqual(plan_female.detected_gender, "female")
        self.assertEqual(plan_male.detected_gender, "male")


class TestSarvamTTSProvider(unittest.TestCase):
    """Tests Sarvam bulbul:v4-flash TTS provider."""

    def test_persona_selection(self):
        provider = SarvamTTSProvider(api_key="dummy_key")
        spk_hi_f = provider._select_speaker("hi", "female", "happy")
        spk_hi_m = provider._select_speaker("hi", "male", "happy")
        spk_en_f = provider._select_speaker("en", "female", "sad")
        spk_en_m = provider._select_speaker("en", "male", "angry")

        self.assertIn("simran", spk_hi_f)
        self.assertIn("amit", spk_hi_m)
        self.assertIn("chandrika", spk_en_f)
        self.assertIn("rustom", spk_en_m)

    @patch("backend.app.models.tts_sarvam.httpx.Client")
    def test_synthesize_success(self, mock_client_cls):
        # Mock 200 response with valid base64 audio
        mock_client = MagicMock()
        mock_client_cls.return_value.__enter__.return_value = mock_client
        dummy_wav_bytes = b"RIFF" + b"\x00" * 36 + b"data" + b"\x00" * 100
        import base64
        mock_response = MagicMock(
            status_code=200,
            json=lambda: {"request_id": "test_req", "audios": [base64.b64encode(dummy_wav_bytes).decode("ascii")]}
        )
        mock_client.post.return_value = mock_response

        provider = SarvamTTSProvider(api_key="test_key")
        plan = EmotionPreservationPlan(
            emotion="happy",
            pitch_multiplier=1.15,
            rate_multiplier=1.12,
            volume_multiplier=1.10,
            gating_weight_z=0.6,
            applied_to_tts=True,
            prosody_notes="Happy prosody"
        )
        res = provider.synthesize("नमस्ते, आप कैसे हैं?", target_lang="hi", preservation_plan=plan, speaker_gender="female")
        self.assertTrue(len(res.audio_bytes) > 0)
        self.assertEqual(res.emotion_applied, "happy")
        self.assertEqual(res.details.get("provider"), "sarvam")


class TestEdgeTTSProvider(unittest.TestCase):
    """Tests Microsoft Edge-TTS neural speech provider."""

    def test_voice_selection(self):
        provider = EdgeTTSProvider()
        hi_f = provider._select_voice("hi", "female")
        hi_m = provider._select_voice("hi", "male")
        en_f = provider._select_voice("en", "female")
        en_m = provider._select_voice("en", "male")

        self.assertIn("Swara", hi_f)
        self.assertIn("Madhur", hi_m)
        self.assertIn("Jenny", en_f)
        self.assertIn("Guy", en_m)


class TestEmotionPreservingTTSCascade(unittest.TestCase):
    """Tests the orchestrator and automatic cascading fallback."""

    def test_fallback_when_sarvam_fails(self):
        """When Sarvam provider raises error, orchestrator must seamlessly fallback to Edge or macOS."""
        tts = EmotionPreservingTTS(primary_engine="sarvam")

        # Mock sarvam failure
        tts.sarvam_provider.synthesize = MagicMock(side_effect=RuntimeError("Sarvam 503 Outage"))

        # Mock edge success
        dummy_wav_bytes = b"RIFF" + b"\x00" * 36 + b"data" + b"\x00" * 200
        mock_edge_res = TTSResult(
            audio_bytes=dummy_wav_bytes,
            audio_format="wav",
            sample_rate=16000,
            duration_seconds=1.2,
            voice_used="hi-IN-SwaraNeural",
            latency_ms=300.0,
            model_name="edge_tts_hi-IN-SwaraNeural",
            emotion_applied="happy",
            details={"provider": "edge_tts"}
        )
        tts.edge_provider.synthesize = MagicMock(return_value=mock_edge_res)

        plan = EmotionPreservationPlan(
            emotion="happy",
            pitch_multiplier=1.15,
            rate_multiplier=1.12,
            volume_multiplier=1.10,
            gating_weight_z=0.6,
            applied_to_tts=True,
            prosody_notes="Happy prosody"
        )
        res = tts.synthesize("नमस्ते", target_lang="hi", preservation_plan=plan, speaker_gender="female")

        self.assertTrue(res.details.get("fallback_used"))
        self.assertEqual(res.details.get("primary_provider_failed"), "SarvamTTSProvider")
        self.assertEqual(res.voice_used, "hi-IN-SwaraNeural")


if __name__ == "__main__":
    unittest.main()
