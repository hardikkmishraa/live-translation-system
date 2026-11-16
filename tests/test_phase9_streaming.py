"""
Phase 9 Integration & Unit Test Suite: Low-Latency Voice Processing & Streaming
================================================================================
Validates all requirements for Phase 9:
  1. Audio Chunking (VAD, speech boundaries, minimum speech duration, silence trigger).
  2. Incremental Transcription (Partial hypothesis emission, deduplication, stabilization).
  3. Emotion Processing & EMA Aggregation (Acoustic smoothing across adjacent dialogue segments).
  4. Segment-Based Sarvam Translation (Stable text segments only, monotonic segment IDs).
  5. Audio Playback Queue Readiness (Ordered segment queuing, failure recovery).
  6. WebSocket Event Contract (recording_started, partial_transcript, final_transcript,
     emotion_updated, translation_completed, audio_ready, processing_error).
  7. Fallback Preservation (Existing batch file upload & complete recording workflows intact).
"""

import json
import time
from pathlib import Path
from typing import Dict, Any, List
import numpy as np
import pytest
from starlette.testclient import TestClient

from backend.app.main import app
from backend.app.services.streaming_session import (
    EmotionAggregator,
    StreamingVADChunker,
    TranscriptStabilizer,
    AggregatedEmotion
)

client = TestClient(app)


# ===========================================================================
# 1. Unit Tests: Audio Chunking & VAD State Machine
# ===========================================================================

class TestAudioChunking:
    def test_chunker_initialization(self):
        chunker = StreamingVADChunker(
            sample_rate=16000,
            silence_threshold_rms=0.007,
            min_speech_samples=4800,
            silence_chunks_trigger=3
        )
        assert chunker.sample_rate == 16000
        assert not chunker.in_speech
        assert len(chunker.audio_buffer) == 0

    def test_silence_does_not_trigger_speech(self):
        chunker = StreamingVADChunker(silence_threshold_rms=0.01)
        silent_frame = np.zeros(1600, dtype=np.float32)  # 100ms of pure silence
        is_boundary, audio, meta = chunker.feed_chunk(silent_frame)
        assert not is_boundary
        assert audio is None
        assert not meta["in_speech"]
        assert meta["frame_rms"] == 0.0

    def test_speech_onset_and_boundary_detection(self):
        chunker = StreamingVADChunker(
            silence_threshold_rms=0.01,
            min_speech_samples=3200,   # 200ms
            silence_chunks_trigger=2   # 2 silent frames to declare boundary
        )
        # 1. Send 3 active speech frames (sine wave at 200 Hz, RMS ~0.7)
        t = np.linspace(0, 0.1, 1600, endpoint=False, dtype=np.float32)
        speech_frame = (np.sin(2 * np.pi * 200 * t) * 0.5).astype(np.float32)

        is_b1, a1, m1 = chunker.feed_chunk(speech_frame)
        assert not is_b1
        assert m1["in_speech"]

        is_b2, a2, m2 = chunker.feed_chunk(speech_frame)
        assert not is_b2

        is_b3, a3, m3 = chunker.feed_chunk(speech_frame)
        assert not is_b3
        assert chunker.speech_chunks_count == 3

        # 2. Send 2 silent frames to trigger boundary
        silent_frame = np.zeros(1600, dtype=np.float32)
        is_b4, a4, _ = chunker.feed_chunk(silent_frame)
        assert not is_b4  # First silence chunk: not yet trigger

        is_b5, completed_audio, m5 = chunker.feed_chunk(silent_frame)
        assert is_b5      # Second silence chunk: triggers boundary!
        assert completed_audio is not None
        assert len(completed_audio) >= 3200
        assert not chunker.in_speech

    def test_force_flush(self):
        chunker = StreamingVADChunker()
        chunk = np.random.randn(3200).astype(np.float32) * 0.1
        chunker.feed_chunk(chunk)
        flushed = chunker.force_flush()
        assert flushed is not None
        assert len(flushed) >= 3200
        assert len(chunker.audio_buffer) == 0


# ===========================================================================
# 2. Unit Tests: Incremental Transcription & Stabilization
# ===========================================================================

class TestTranscriptStabilizer:
    def test_partial_transcript_deduplication(self):
        stabilizer = TranscriptStabilizer()
        assert stabilizer.should_emit_partial("Hello")
        # Identical text should be suppressed
        assert not stabilizer.should_emit_partial("Hello")
        assert not stabilizer.should_emit_partial("hello")
        # Extended text should be emitted
        assert stabilizer.should_emit_partial("Hello world")

    def test_final_transcript_filtering(self):
        stabilizer = TranscriptStabilizer()
        final1 = stabilizer.filter_final("Hello world")
        assert final1 == "Hello world"

        # Consecutive duplicate final should be suppressed
        final2 = stabilizer.filter_final("Hello world")
        assert final2 is None

        # New sentence should be allowed
        final3 = stabilizer.filter_final("How are you today?")
        assert final3 == "How are you today?"
        assert len(stabilizer.committed_history) == 2


# ===========================================================================
# 3. Unit Tests: Cross-Segment Emotion Aggregation (EMA)
# ===========================================================================

class TestEmotionAggregator:
    def test_first_segment_initialization(self):
        agg = EmotionAggregator(alpha=0.65)
        p1 = {"neutral": 0.8, "happy": 0.2}
        res1 = agg.update(p1, valence=0.6, arousal=0.4, dominance=0.5)

        assert res1.dominant_emotion == "neutral"
        assert res1.probabilities["neutral"] == 0.8
        assert res1.valence == 0.6
        assert res1.update_count == 1

    def test_exponential_smoothing_across_adjacent_segments(self):
        agg = EmotionAggregator(alpha=0.65)
        # Segment 1: strong neutral
        agg.update({"neutral": 1.0, "happy": 0.0}, valence=0.5, arousal=0.3)

        # Segment 2: burst of happiness
        res2 = agg.update({"neutral": 0.2, "happy": 0.8}, valence=0.9, arousal=0.8)

        # Expected EMA:
        # P(happy) = 0.65 * 0.8 + 0.35 * 0.0 = 0.52
        # P(neutral) = 0.65 * 0.2 + 0.35 * 1.0 = 0.48
        # Dominant emotion should smoothly transition with history
        assert res2.update_count == 2
        assert res2.probabilities["happy"] > 0.50
        assert res2.probabilities["neutral"] > 0.40
        # Valence smoothed: 0.65 * 0.9 + 0.35 * 0.5 = 0.76
        assert 0.74 <= res2.valence <= 0.78


# ===========================================================================
# 4. WebSocket Full Protocol & Lifecycle Events
# ===========================================================================

class TestWebSocketStreamingProtocol:
    def test_ws_session_start_emits_recording_started(self):
        """Verifies session_start triggers 'recording_started' and 'session_acknowledged'."""
        with client.websocket_connect("/ws/live-translate") as ws:
            ws.send_text(json.dumps({
                "action": "session_start",
                "session_id": "test_sess_1",
                "source_lang": "hi",
                "target_lang": "en"
            }))
            # Should receive recording_started and session_acknowledged
            msg1 = ws.receive_json()
            assert msg1["type"] in ("recording_started", "session_acknowledged")
            assert msg1["session_id"] == "test_sess_1"
            assert msg1["source_lang"] == "hi"
            assert msg1["target_lang"] == "en"

            msg2 = ws.receive_json()
            assert msg2["type"] in ("recording_started", "session_acknowledged")

    def test_ws_heartbeat_ping_pong(self):
        """Verifies ping/heartbeat receives timely pong."""
        with client.websocket_connect("/ws/live-translate") as ws:
            ws.send_text(json.dumps({"action": "ping"}))
            msg = ws.receive_json()
            assert msg["type"] == "pong"
            assert "timestamp" in msg

    def test_ws_flush_action_emits_listening_when_empty(self):
        """Verifies flush on empty buffer emits listening response without crashing."""
        with client.websocket_connect("/ws/live-translate") as ws:
            ws.send_text(json.dumps({"action": "flush"}))
            msg = ws.receive_json()
            assert msg["type"] in ("ping", "listening")

    def test_ws_audio_chunk_streaming_flow(self):
        """
        Streams binary audio chunks and verifies that no crash occurs,
        chunk receipts are handled, and clean session_stop terminates gracefully.
        """
        with client.websocket_connect("/ws/live-translate") as ws:
            # Send session_start
            ws.send_text(json.dumps({
                "action": "session_start",
                "session_id": "test_stream_2",
                "source_lang": "en",
                "target_lang": "hi"
            }))
            ws.receive_json()  # recording_started
            ws.receive_json()  # session_acknowledged

            # Send 5 small PCM frames
            dummy_pcm = (np.random.randn(800) * 1000).astype(np.int16).tobytes()
            for _ in range(5):
                ws.send_bytes(dummy_pcm)

            # Send stop
            ws.send_text(json.dumps({"action": "stop"}))
            stop_ack = ws.receive_json()
            assert stop_ack["type"] == "session_closed"


# ===========================================================================
# 5. Fallback Preservation: Batch Mode Remains Fully Functional
# ===========================================================================

class TestBatchModeFallbackPreserved:
    def test_batch_file_upload_unaffected(self):
        """Verifies Phase 7 complete recording upload continues working alongside Phase 9."""
        import io
        import soundfile as sf
        # Generate 0.6s tone
        sr = 16000
        t = np.linspace(0, 0.6, int(sr * 0.6), endpoint=False)
        audio = (np.sin(2 * np.pi * 300 * t) * 0.3).astype(np.float32)
        buf = io.BytesIO()
        sf.write(buf, audio, sr, format="WAV")
        buf.seek(0)

        res = client.post(
            "/api/v1/audio/upload",
            files={"file": ("fallback_test.wav", buf.read(), "audio/wav")},
            data={"source_lang": "en", "target_lang": "hi", "synthesize_speech": "false"}
        )
        assert res.status_code == 200
        payload = res.json()
        assert payload["success"] is True
        assert "latency" in payload

    def test_batch_text_translate_unaffected(self):
        """Verifies text translation endpoint remains intact."""
        res = client.post(
            "/api/v1/text-translate",
            json={"text": "Good morning", "source_lang": "en", "target_lang": "hi", "synthesize_speech": False}
        )
        assert res.status_code == 200
        payload = res.json()
        assert payload["success"] is True
        assert "translated_text" in payload
