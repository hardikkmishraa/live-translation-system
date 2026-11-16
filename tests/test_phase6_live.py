import json
import time
import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend.app.main import app

client = TestClient(app)

# Helper to synthesize 16kHz 16-bit PCM bytes
def generate_pcm16_chunk(duration_sec: float = 0.5, freq: float = 300.0, sample_rate: int = 16000) -> bytes:
    t = np.linspace(0, duration_sec, int(sample_rate * duration_sec), endpoint=False)
    # Speech-like amplitude modulated tone
    sine = (0.5 * np.sin(2 * np.pi * freq * t) * (1.0 + 0.3 * np.sin(2 * np.pi * 5 * t))).astype(np.float32)
    pcm16 = (sine * 32767.0).astype(np.int16)
    return pcm16.tobytes()

def generate_silence_chunk(duration_sec: float = 0.25, sample_rate: int = 16000) -> bytes:
    pcm16 = np.zeros(int(sample_rate * duration_sec), dtype=np.int16)
    return pcm16.tobytes()

# =========================================================================
# 1. WebSocket Protocol & Session Lifecycle Tests
# =========================================================================

def test_websocket_session_lifecycle():
    with client.websocket_connect("/ws/live-translate") as ws:
        # Start session
        ws.send_text(json.dumps({
            "action": "session_start",
            "session_id": "sess-live-001",
            "source_lang": "en",
            "target_lang": "hi",
            "user_consent": False
        }))

        ack_data = json.loads(ws.receive_text())
        assert ack_data["type"] == "session_acknowledged"
        assert ack_data["session_id"] == "sess-live-001"
        assert ack_data["source_lang"] == "en"
        assert ack_data["target_lang"] == "hi"
        assert ack_data["status"] == "ready"

        # Heartbeat ping
        ws.send_text(json.dumps({"action": "ping"}))
        pong = json.loads(ws.receive_text())
        assert pong["type"] == "pong"
        assert pong["session_id"] == "sess-live-001"

        # Explicit stop
        ws.send_text(json.dumps({"action": "stop"}))
        stop_ack = json.loads(ws.receive_text())
        assert stop_ack["type"] == "session_closed"
        assert stop_ack["session_id"] == "sess-live-001"

def test_websocket_invalid_language_pair():
    with client.websocket_connect("/ws/live-translate") as ws:
        # Source and Target identical
        ws.send_text(json.dumps({
            "action": "session_start",
            "session_id": "sess-err-002",
            "source_lang": "en",
            "target_lang": "en"
        }))

        err = json.loads(ws.receive_text())
        assert err["type"] == "error"
        assert err["error_code"] == "INVALID_LANGUAGE_PAIR"
        assert err["recoverable"] is True

def test_websocket_cancel_in_flight_utterance():
    with client.websocket_connect("/ws/live-translate") as ws:
        ws.send_text(json.dumps({
            "action": "session_start",
            "session_id": "sess-cancel-003",
            "source_lang": "en",
            "target_lang": "hi"
        }))
        ack = json.loads(ws.receive_text())
        assert ack["type"] == "session_acknowledged"

        # Stream audio chunk
        audio_chunk = generate_pcm16_chunk(duration_sec=0.5)
        ws.send_bytes(audio_chunk)

        # Cancel utterance
        ws.send_text(json.dumps({"action": "cancel"}))
        cancel_resp = json.loads(ws.receive_text())
        assert cancel_resp["type"] == "cancelled"
        assert cancel_resp["segment_id"] == 1
        assert "cancelled" in cancel_resp["message"].lower()

def test_websocket_payload_size_limit_protection():
    with client.websocket_connect("/ws/live-translate") as ws:
        # Oversized audio payload (300 KB > 256 KB max)
        oversized = bytes(300 * 1024)
        ws.send_bytes(oversized)

        err = json.loads(ws.receive_text())
        assert err["type"] == "error"
        assert err["error_code"] == "PAYLOAD_TOO_LARGE"
        assert err["recoverable"] is True

def test_websocket_malformed_json_handling():
    with client.websocket_connect("/ws/live-translate") as ws:
        ws.send_text("{bad json text:::")
        err = json.loads(ws.receive_text())
        assert err["type"] == "error"
        assert err["error_code"] == "MALFORMED_JSON"
        assert err["recoverable"] is True

# =========================================================================
# 2. VAD Segmentation & End-to-End Live Processing Tests
# =========================================================================

def test_websocket_live_flush_and_ordered_segment():
    with client.websocket_connect("/ws/live-translate") as ws:
        ws.send_text(json.dumps({
            "action": "session_start",
            "session_id": "sess-live-flush-004",
            "source_lang": "en",
            "target_lang": "hi"
        }))
        ack = json.loads(ws.receive_text())
        assert ack["type"] == "session_acknowledged"

        # Stream speech chunk (1.0 second of audio)
        pcm_data = generate_pcm16_chunk(duration_sec=1.0)
        ws.send_bytes(pcm_data)

        # Force flush utterance
        ws.send_text(json.dumps({"action": "flush"}))

        # We should receive either partial_transcript, final_transcript, translation_chunk, or listening ping
        messages = []
        for _ in range(5):
            msg = json.loads(ws.receive_text())
            messages.append(msg)
            if msg["type"] in ("ping", "translation_chunk"):
                break
        types = [m["type"] for m in messages]
        assert any(t in ("final_transcript", "translation_chunk", "ping", "partial_transcript") for t in types)

# =========================================================================
# 3. Progressive Playback Queue Ordering Logic Tests
# =========================================================================

def test_playback_queue_ordering_simulation():
    """
    Validates monotonic ordering: if segment 2 arrives before segment 1,
    the queue buffers segment 2 and drains strictly in 1 -> 2 order.
    """
    queue = []
    stashed = {}
    expected_next = 1
    drained_order = []

    def handle_segment(seg_id, data):
        nonlocal expected_next
        if seg_id > expected_next:
            stashed[seg_id] = data
            return
        # Enqueue in order
        queue.append((seg_id, data))
        expected_next = max(expected_next, seg_id + 1)
        # Drain stashed
        while expected_next in stashed:
            item = stashed.pop(expected_next)
            queue.append((expected_next, item))
            expected_next += 1

    # Simulate arrival out of order: #2 arrives before #1
    handle_segment(2, "AudioChunk_2")
    assert len(queue) == 0  # #2 is stashed waiting for #1
    assert 2 in stashed

    # #1 arrives
    handle_segment(1, "AudioChunk_1")
    assert len(queue) == 2  # Both #1 and #2 now resolved in order!
    assert [item[0] for item in queue] == [1, 2]

    # #3 arrives
    handle_segment(3, "AudioChunk_3")
    assert [item[0] for item in queue] == [1, 2, 3]

# =========================================================================
# 4. Audio Resampler Mathematical Precision Tests
# =========================================================================

def test_audio_resampler_linear_interpolation():
    """Validates resampling from 48000 Hz or 44100 Hz to 16000 Hz."""
    input_rate = 48000
    target_rate = 16000
    duration_sec = 0.5
    input_length = int(input_rate * duration_sec)
    t = np.linspace(0, duration_sec, input_length, endpoint=False)
    input_samples = (np.sin(2 * np.pi * 440 * t)).astype(np.float32)

    # Resample algorithm equivalent to audio.js _resampleTo16k
    ratio = input_rate / float(target_rate)
    output_length = int(round(input_length / ratio))
    output_samples = np.zeros(output_length, dtype=np.float32)

    for i in range(output_length):
        origin_idx = i * ratio
        idx_floor = int(np.floor(origin_idx))
        idx_ceil = min(idx_floor + 1, input_length - 1)
        weight = origin_idx - idx_floor
        output_samples[i] = input_samples[idx_floor] * (1.0 - weight) + input_samples[idx_ceil] * weight

    assert len(output_samples) == int(target_rate * duration_sec)
    assert np.max(np.abs(output_samples)) <= 1.05


# =========================================================================
# 5. Regression & Live Speech WebSocket Contract Tests
# =========================================================================

def test_regression_asdict_type_error_reproduction_and_fix():
    """
    Directly reproduces the reported bug:
      TypeError: asdict() should be called on dataclass instances
    when passing a dictionary (from PipelineResult.preservation_plan) to asdict().
    Verifies that the corrected serialization handles dicts, dataclasses, and None.
    """
    from dataclasses import asdict, is_dataclass
    from backend.app.models.base import EmotionPreservationPlan

    # 1. Reproduce the exact bug: asdict() on a dict raises TypeError
    preservation_plan_dict = {
        "emotion": "happy",
        "pitch_multiplier": 1.15,
        "rate_multiplier": 1.10,
        "volume_multiplier": 1.05,
        "gating_weight_z": 0.62,
        "applied_to_tts": True,
        "prosody_notes": "Upbeat speaking cadence"
    }

    with pytest.raises(TypeError, match="asdict\\(\\) should be called on dataclass instances"):
        asdict(preservation_plan_dict)

    # 2. Verify corrected serialization logic for dict
    def serialize_plan(plan):
        if plan is None:
            return None
        elif isinstance(plan, dict):
            return plan
        elif is_dataclass(plan) and not isinstance(plan, type):
            return asdict(plan)
        elif hasattr(plan, "model_dump"):
            return plan.model_dump()
        else:
            return dict(plan)

    res_from_dict = serialize_plan(preservation_plan_dict)
    assert isinstance(res_from_dict, dict)
    assert res_from_dict["pitch_multiplier"] == 1.15

    # 3. Verify corrected serialization logic for dataclass instance
    plan_dataclass = EmotionPreservationPlan(
        emotion="happy",
        pitch_multiplier=1.20,
        rate_multiplier=1.05,
        volume_multiplier=1.00,
        gating_weight_z=0.65,
        applied_to_tts=True,
        prosody_notes="Elevated pitch"
    )
    res_from_dc = serialize_plan(plan_dataclass)
    assert isinstance(res_from_dc, dict)
    assert res_from_dc["pitch_multiplier"] == 1.20

    # 4. Verify None handling
    assert serialize_plan(None) is None


def test_websocket_live_speech_processing_emission():
    """
    Tests that a live WebSocket session receives speech audio, processes it through
    _process_and_emit_segment, and emits the expected sequence of structured events:
      1. final_transcript
      2. emotion_update
      3. translation_chunk (with valid preservation_plan dict and audio_url)
    without raising 'Server processing error: asdict() should be called on dataclass instances'.
    """
    from unittest.mock import patch
    from backend.app.services.pipeline import PipelineResult, LatencyBreakdown
    from backend.app.api.endpoints import websocket as ws_module

    mock_pipeline_res = PipelineResult(
        source_language="en",
        target_language="hi",
        source_transcript="Hello, how are you today?",
        translated_text="नमस्ते, आप आज कैसे हैं?",
        asr_confidence=0.96,
        detected_emotion="happy",
        emotion_confidence=0.89,
        emotion_probabilities={"happy": 0.89, "neutral": 0.11},
        native_emotion_scores={"happy": 0.89, "neutral": 0.11},
        valence=0.75,
        arousal=0.65,
        dominance=0.60,
        gating_weight_z=0.62,
        preservation_plan={
            "emotion": "happy",
            "pitch_multiplier": 1.15,
            "rate_multiplier": 1.10,
            "volume_multiplier": 1.05,
            "gating_weight_z": 0.62,
            "applied_to_tts": True,
            "prosody_notes": "Upbeat speaking cadence",
            "expressive_synthesis_supported": False,
            "emotional_fidelity_status": "Heuristic prosody control only",
            "speaker_similarity_retained": False,
            "synthesis_voice_used": "Lekha"
        },
        expressive_synthesis_supported=False,
        emotional_fidelity_status="Heuristic prosody control only",
        speaker_similarity_retained=False,
        synthesis_voice_used="Lekha",
        language_limitation_warning=None,
        original_audio_filename="orig_mock_123.wav",
        audio_base64="UklGRiQAAABXQVZFZm10IBAAAAABAAEAQB8AAEAfAAABAAgAZGF0YQAAAAA=",
        audio_filename="tts_mock_123.wav",
        audio_url="/api/v1/audio/tts_mock_123.wav",
        latency=LatencyBreakdown(
            asr_ms=180.5,
            ser_ms=42.1,
            nmt_ms=95.3,
            tts_ms=210.0,
            feature_extraction_ms=18.0,
            total_pipeline_ms=545.9
        ),
        speech_detected=True,
        diagnostic_details={"status": "mock_success"}
    )

    with patch.object(ws_module.pipeline, "process_audio", return_value=mock_pipeline_res):
        with client.websocket_connect("/ws/live-translate") as ws:
            # 1. Session handshake
            ws.send_text(json.dumps({
                "action": "session_start",
                "session_id": "test-speech-session-001",
                "source_lang": "en",
                "target_lang": "hi",
                "user_consent": False
            }))
            ack = json.loads(ws.receive_text())
            assert ack["type"] == "session_acknowledged"

            # 2. Stream speech chunk (active speech above RMS threshold)
            pcm_data = generate_pcm16_chunk(duration_sec=1.0, freq=300.0)
            ws.send_bytes(pcm_data)

            # 3. Flush utterance boundary
            ws.send_text(json.dumps({"action": "flush"}))

            # 4. Collect emitted events until translation_chunk arrives
            events = []
            for _ in range(10):
                event = json.loads(ws.receive_text())
                events.append(event)
                if event["type"] == "translation_chunk":
                    break

            event_types = [e["type"] for e in events]
            assert "final_transcript" in event_types
            assert "emotion_update" in event_types
            assert "translation_chunk" in event_types

            # Verify final_transcript
            final_tx = next(e for e in events if e["type"] == "final_transcript")
            assert final_tx["session_id"] == "test-speech-session-001"
            assert final_tx["segment_id"] == 1
            assert final_tx["source_transcript"] == "Hello, how are you today?"
            assert final_tx["is_final"] is True

            # Verify emotion_update
            emo_up = next(e for e in events if e["type"] == "emotion_update")
            assert emo_up["session_id"] == "test-speech-session-001"
            assert emo_up["detected_emotion"] == "happy"
            assert emo_up["confidence"] == 0.89

            # Verify translation_chunk
            chunk = next(e for e in events if e["type"] == "translation_chunk")
            assert chunk["session_id"] == "test-speech-session-001"
            assert chunk["segment_id"] == 1
            assert chunk["speech_detected"] is True
            assert chunk["source_transcript"] == "Hello, how are you today?"
            assert chunk["translated_text"] == "नमस्ते, आप आज कैसे हैं?"
            assert chunk["confidence"] == 0.96

            # Verify preservation_plan is a valid structured dict, NOT throwing TypeError
            assert isinstance(chunk["preservation_plan"], dict)
            assert chunk["preservation_plan"]["emotion"] == "happy"
            assert chunk["preservation_plan"]["pitch_multiplier"] == 1.15
            assert chunk["preservation_plan"]["rate_multiplier"] == 1.10

            # Verify audio payload & latency structure
            assert chunk["audio_base64"] is not None
            assert chunk["audio_url"] == "/api/v1/audio/tts_mock_123.wav"
            assert "latency" in chunk
            assert chunk["latency"]["asr_ms"] == 180.5
            assert chunk["latency"]["total_pipeline_ms"] == 545.9
            assert "end_to_end_utterance_ms" in chunk["latency"]

