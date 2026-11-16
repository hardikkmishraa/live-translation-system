import io
import time
import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.config import TEMP_STORAGE_DIR

client = TestClient(app)

def create_synthetic_wav(duration_sec: float = 1.0, sr: int = 16000) -> bytes:
    """Generates synthetic audio waveform as PCM WAV bytes."""
    t = np.linspace(0, duration_sec, int(sr * duration_sec), endpoint=False)
    waveform = (0.5 * np.sin(2 * np.pi * 220 * t) + 0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    buf = io.BytesIO()
    sf.write(buf, waveform, sr, format="WAV", subtype="PCM_16")
    buf.seek(0)
    return buf.read()


def test_voice_recording_upload_end_to_end():
    """Verify complete recording upload workflow: Audio In -> ASR -> SER -> Sarvam -> TTS -> Output."""
    wav_bytes = create_synthetic_wav(1.0)
    files = {"file": ("recording.wav", wav_bytes, "audio/wav")}
    data = {"source_lang": "hi", "target_lang": "en", "synthesize_speech": "true"}

    response = client.post("/api/v1/audio/upload", files=files, data=data)
    assert response.status_code == 200
    res_json = response.json()

    assert res_json["success"] is True
    assert "request_id" in res_json
    assert res_json["source_language"] in ("hi", "unknown")
    assert res_json["target_language"] == "en"
    assert "asr_status" in res_json
    assert "ser_status" in res_json
    assert "translation_status" in res_json
    assert "tts_status" in res_json
    assert "total_processing_duration" in res_json
    assert res_json["original_audio_url"] is not None

    # Check original recording is playable from backend
    orig_url = res_json["original_audio_url"]
    orig_res = client.get(orig_url)
    assert orig_res.status_code == 200
    assert orig_res.headers["content-type"].startswith("audio/")


def test_audio_playback_streaming_and_path_traversal():
    """Verify generated audio can be streamed and path traversal attempts are blocked."""
    # 1. Create a dummy wav in TEMP_STORAGE_DIR
    dummy_wav_name = f"test_stream_{int(time.time())}.wav"
    dummy_path = TEMP_STORAGE_DIR / dummy_wav_name
    sf.write(str(dummy_path), np.zeros(8000, dtype=np.float32), 16000, format="WAV", subtype="PCM_16")

    try:
        # Legitimate retrieval
        res = client.get(f"/api/v1/audio/{dummy_wav_name}")
        assert res.status_code == 200
        assert res.headers["content-type"].startswith("audio/")
        assert len(res.content) > 44

        # Path traversal attack attempt
        traversal_res = client.get("/api/v1/audio/../../etc/passwd")
        assert traversal_res.status_code in (400, 404)
    finally:
        dummy_path.unlink(missing_ok=True)


def test_streaming_websocket_voice_interaction():
    """Verify streaming live speech interaction over WebSocket protocol (Phase 4 & 6)."""
    with client.websocket_connect("/ws/live-translate") as ws:
        # 1. Start session
        ws.send_json({
            "action": "session_start",
            "source_lang": "hi",
            "target_lang": "en"
        })
        ack = ws.receive_json()
        assert ack["type"] == "session_acknowledged"
        assert "session_id" in ack

        # 2. Stream binary audio frames
        t = np.linspace(0, 0.2, 3200, endpoint=False)
        tone = (0.5 * np.sin(2 * np.pi * 300 * t)).astype(np.float32)
        pcm16 = (tone * 32767).astype(np.int16).tobytes()

        # Send 3 chunks of audio
        for _ in range(3):
            ws.send_bytes(pcm16)

        # 3. Send flush control action to force processing
        ws.send_json({"action": "flush"})

        # Collect event responses
        events = []
        try:
            for _ in range(5):
                msg = ws.receive_json()
                events.append(msg)
                if msg.get("type") in ("translation_chunk", "ping"):
                    break
        except Exception:
            pass

        assert len(events) > 0
        event_types = [e.get("type") for e in events]
        assert any(t in event_types for t in ["partial_transcript", "final_transcript", "emotion_update", "translation_chunk", "ping"])

        # 4. Clean shutdown
        ws.send_json({"action": "stop"})
        close_ack = ws.receive_json()
        assert close_ack["type"] == "session_closed"


def test_streaming_websocket_cancel_in_flight():
    """Verify client cancellation cleanly clears audio buffer and notifies client."""
    with client.websocket_connect("/ws/live-translate") as ws:
        ws.send_json({"action": "session_start", "source_lang": "en", "target_lang": "hi"})
        ws.receive_json()

        # Send some audio
        pcm16 = (np.ones(1600, dtype=np.int16)).tobytes()
        ws.send_bytes(pcm16)

        # Cancel in-flight utterance
        ws.send_json({"action": "cancel"})
        resp = ws.receive_json()
        assert resp["type"] == "cancelled"
        assert "Current in-flight utterance cancelled." in resp["message"]

        # Close session
        ws.send_json({"action": "stop"})
        ws.receive_json()

