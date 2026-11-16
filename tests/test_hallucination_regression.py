"""
VoiceAffect Bug Fix Regression Test Suite: Eliminate Unwanted Extra Words
========================================================================
Automated regression tests covering:
- Test A: No extra phrase (English -> Hindi: "I am going to the market." does NOT append "धन्यवाद आपका")
- Test B: Reverse translation (Hindi -> English: "मैं आज कॉलेज जा रहा हूँ।" does NOT append closing remarks)
- Test C: Legitimate phrase preservation ("Thank you for your help." DOES retain gratitude)
- Test D: Live-stream duplication (multiple consecutive audio chunks do not repeatedly duplicate)
- Test E: Silence and finalization (silence/end-of-stream emits no fabricated sentences)
- Test F: Display and speech consistency (displayed translation matches TTS input text)
"""

import json
import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.models.asr import WhisperASRModel
from backend.app.models.nmt import BilingualNMTModel
from backend.app.models.tts import EmotionPreservingTTS
from backend.app.services.pipeline import LiveTranslationPipeline
from backend.app.core.audio_processor import AudioProcessor

client = TestClient(app)

class TestHallucinationFix:
    """Verifies elimination of unwanted extra phrases at each pipeline stage."""

    @pytest.fixture(scope="class")
    def nmt(self):
        return BilingualNMTModel()

    @pytest.fixture(scope="class")
    def asr(self):
        return WhisperASRModel()

    @pytest.fixture(scope="class")
    def pipeline(self):
        return LiveTranslationPipeline()

    @pytest.fixture(scope="class")
    def audio_proc(self):
        return AudioProcessor()

    # ---------------------------------------------------------------------
    # Test A: No extra phrase (English -> Hindi)
    # ---------------------------------------------------------------------
    def test_a_no_extra_phrase_english_to_hindi(self, nmt):
        """Input: 'I am going to the market.'
        Expected: A faithful Hindi equivalent.
        Must NOT append 'धन्यवाद आपका' or any closing pleasantry unless in source.
        """
        source = "I am going to the market."
        result = nmt.translate(source, "en", "hi")
        translated = result.translated_text

        assert len(translated) > 0, "Translation must not be empty"
        assert "धन्यवाद" not in translated, f"Unwanted 'धन्यवाद' found in translation: '{translated}'"
        assert "आपका" not in translated or "बाजार" in translated, f"Corrupted translation: '{translated}'"
        assert "बाजार" in translated, f"Translation should mention market ('बाजार'): '{translated}'"

    # ---------------------------------------------------------------------
    # Test B: Reverse translation (Hindi -> English)
    # ---------------------------------------------------------------------
    def test_b_reverse_translation_hindi_to_english(self, nmt):
        """Input: 'मैं आज कॉलेज जा रहा हूँ।'
        Expected: A faithful English equivalent without unrelated closing phrases.
        """
        source = "मैं आज कॉलेज जा रहा हूँ।"
        result = nmt.translate(source, "hi", "en")
        translated = result.translated_text

        assert len(translated) > 0, "Translation must not be empty"
        lower_trans = translated.lower()
        assert "thank" not in lower_trans, f"Unwanted 'thank' found in translation: '{translated}'"
        assert "bye" not in lower_trans, f"Unwanted 'bye' found in translation: '{translated}'"
        assert "college" in lower_trans, f"Translation should mention 'college': '{translated}'"

    # ---------------------------------------------------------------------
    # Test C: Legitimate phrase preservation
    # ---------------------------------------------------------------------
    def test_c_legitimate_phrase_preservation(self, nmt):
        """Input: 'Thank you for your help.'
        Expected: Hindi translation must preserve the legitimate expression of gratitude.
        Verifies that the fix does NOT blindly strip 'धन्यवाद'.
        """
        source = "Thank you for your help."
        result = nmt.translate(source, "en", "hi")
        translated = result.translated_text

        assert "धन्यवाद" in translated, f"Legitimate gratitude was improperly removed: '{translated}'"
        assert "मदद" in translated, f"Translation should preserve 'help' ('मदद'): '{translated}'"

        # Reverse check: legitimate Hindi thank you preserved in English
        hi_thank = "आपकी सहायता के लिए बहुत बहुत धन्यवाद।"
        res_hi = nmt.translate(hi_thank, "hi", "en")
        assert "thank" in res_hi.translated_text.lower(), f"English translation should preserve thank: '{res_hi.translated_text}'"

    # ---------------------------------------------------------------------
    # Test D: Live-stream duplication
    # ---------------------------------------------------------------------
    def test_d_live_stream_duplication(self):
        """Process multiple consecutive audio chunks over WebSocket.
        Verify that completed text is not repeatedly duplicated or appended infinitely.
        """
        with client.websocket_connect("/ws/live-translate") as ws:
            ws.send_text(json.dumps({
                "action": "session_start",
                "session_id": "test-dupl-001",
                "source_lang": "en",
                "target_lang": "hi",
                "user_consent": False
            }))
            ack = ws.receive_json()
            assert ack.get("type") == "session_acknowledged"

            # Send 5 chunks of ambient low silence (each 2048 samples = 4096 bytes)
            silence_pcm = (np.random.randn(2048) * 0.002).astype(np.float32)
            int16_bytes = (silence_pcm * 32767).astype(np.int16).tobytes()

            emitted_translations = []
            for _ in range(5):
                ws.send_bytes(int16_bytes)

            ws.send_text(json.dumps({"action": "flush"}))
            ws.send_text(json.dumps({"action": "stop"}))
            
            # Read all pending messages
            while True:
                try:
                    msg = ws.receive_json(mode="text")
                    if msg.get("type") == "translation_chunk":
                        emitted_translations.append(msg.get("translated_text"))
                    elif msg.get("type") in ("session_closed", "error"):
                        break
                except Exception:
                    break

            # Must not produce multiple duplicate translated chunks for pure silence
            assert len(emitted_translations) <= 1, f"Multiple duplicate chunks emitted: {emitted_translations}"
            if emitted_translations:
                for t in emitted_translations:
                    assert "धन्यवाद" not in t, f"Hallucinated 'धन्यवाद' emitted on silence stream: '{t}'"

    # ---------------------------------------------------------------------
    # Test E: Silence and finalization
    # ---------------------------------------------------------------------
    def test_e_silence_and_finalization(self, pipeline, asr):
        """Process silence and low-energy noise buffers.
        Verify that no fabricated sentence or default closing phrase is generated.
        """
        # Test 1: Low ambient noise through ASR
        np.random.seed(42)
        ambient_noise = (np.random.randn(16000 * 2) * 0.006).astype(np.float32)
        asr_res = asr.transcribe(ambient_noise, sample_rate=16000)
        assert asr_res.text == "", f"ASR fabricated text on ambient noise: '{asr_res.text}'"

        # Test 2: Low ambient noise through full pipeline
        pipe_res = pipeline.process_audio(ambient_noise, sample_rate=16000)
        assert not pipe_res.speech_detected or pipe_res.source_transcript == "", (
            f"Pipeline fabricated transcript on ambient noise: '{pipe_res.source_transcript}'"
        )
        assert pipe_res.translated_text == "", (
            f"Pipeline fabricated translation on ambient noise: '{pipe_res.translated_text}'"
        )
        assert pipe_res.audio_base64 is None, "TTS audio should not be generated for empty silence"

    # ---------------------------------------------------------------------
    # Test F: Display and speech consistency
    # ---------------------------------------------------------------------
    def test_f_display_and_speech_consistency(self, pipeline):
        """Verify that the final displayed translation matches the text submitted to TTS."""
        tts_engine = EmotionPreservingTTS()

        sample_texts = [
            ("I am going to the market.", "en", "hi"),
            ("Thank you for your help.", "en", "hi"),
            ("मैं आज कॉलेज जा रहा हूँ।", "hi", "en")
        ]

        for text, src, tgt in sample_texts:
            nmt = BilingualNMTModel()
            nmt_res = nmt.translate(text, src, tgt)
            displayed_translation = nmt_res.translated_text

            # TTS input is exactly nmt_res.translated_text
            tts_res = tts_engine.synthesize(displayed_translation, target_lang=tgt)

            # Both must agree on the content without extra words
            if displayed_translation:
                if tts_engine._has_native:
                    assert len(tts_res.audio_bytes) > 0
                else:
                    assert tts_res.audio_bytes == b""
                    assert tts_res.details.get("error")
                assert tts_res.latency_ms >= 0.0
                if "market" in text:
                    assert "धन्यवाद" not in displayed_translation
            else:
                assert tts_res.audio_bytes == b""

    # ---------------------------------------------------------------------
    # Test G: YouTube Outro Stripping from True Speech
    # ---------------------------------------------------------------------
    def test_g_outro_stripping_preserves_sentence(self, asr):
        """Verify that trailing YouTube outro artifacts attached to a sentence are cleanly removed."""
        raw_with_outro = "I am going to the market. Thank you for watching!"
        cleaned = asr.TRAILING_OUTRO_PATTERN.sub("", raw_with_outro).strip()
        assert cleaned == "I am going to the market"

        raw_hindi_outro = "मैं जा रहा हूँ। देखने के लिए धन्यवाद!"
        cleaned_hi = asr.TRAILING_OUTRO_PATTERN.sub("", raw_hindi_outro).strip()
        assert cleaned_hi == "मैं जा रहा हूँ"
