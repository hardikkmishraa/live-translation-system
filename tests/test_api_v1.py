import io
import json
import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient
from backend.app.main import app
from backend.app.schemas.models import ModelStatusEnum

client = TestClient(app)

def test_v1_health():
    res = client.get("/api/v1/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "healthy"
    assert "device" in data
    assert "uptime_seconds" in data
    assert "ffmpeg_available" in data
    assert "supported_pairs" in data

def test_v1_models_status():
    res = client.get("/api/v1/models/status")
    assert res.status_code == 200
    data = res.json()
    assert "modules" in data
    modules = data["modules"]
    
    # Required cognitive modules
    for mod in ["asr", "ser", "nmt", "gating", "tts"]:
        assert mod in modules
        item = modules[mod]
        assert "status" in item
        assert "is_loaded" in item
        assert "scientific_basis" in item
        assert "supported_languages" in item
        # Status must be one of the defined ModelStatusEnum values
        valid_statuses = [e.value for e in ModelStatusEnum]
        assert item["status"] in valid_statuses

def test_v1_capabilities():
    res = client.get("/api/v1/capabilities")
    assert res.status_code == 200
    data = res.json()
    assert "supported_languages" in data
    assert len(data["supported_languages"]) >= 2
    codes = [lang["code"] for lang in data["supported_languages"]]
    assert "hi" in codes
    assert "en" in codes
    assert "allowed_audio_extensions" in data
    assert ".wav" in data["allowed_audio_extensions"]
    assert "media_limits" in data
    assert data["media_limits"]["max_upload_size_mb"] == 25

def test_v1_audio_upload_validation_unsupported_format():
    buf = io.BytesIO(b"fake data")
    files = {"file": ("unsupported_file.xyz", buf, "application/octet-stream")}
    res = client.post("/api/v1/audio/upload", files=files)
    assert res.status_code == 400
    data = res.json()
    assert data["success"] is False
    assert "error" in data
    assert "Unsupported format" in data["error"]["message"]

def test_v1_audio_upload_validation_empty_file():
    buf = io.BytesIO(b"")
    files = {"file": ("empty.wav", buf, "audio/wav")}
    res = client.post("/api/v1/audio/upload", files=files)
    assert res.status_code == 400
    data = res.json()
    assert data["success"] is False
    assert "empty" in data["error"]["message"]

def test_v1_audio_upload_valid_wav():
    sr = 16000
    t = np.linspace(0, 0.5, int(sr * 0.5), endpoint=False)
    tone = (0.5 * np.sin(2 * np.pi * 300 * t)).astype(np.float32)

    buf = io.BytesIO()
    sf.write(buf, tone, sr, format="WAV", subtype="PCM_16")
    buf.seek(0)

    files = {"file": ("test_valid.wav", buf, "audio/wav")}
    res = client.post(
        "/api/v1/audio/upload",
        files=files,
        data={"source_lang": "en", "target_lang": "hi", "synthesize_speech": "true"}
    )
    assert res.status_code == 200
    data = res.json()
    assert data["success"] is True
    assert "detected_emotion" in data
    assert "latency" in data
    assert data["latency"]["total_pipeline_ms"] > 0

def test_v1_text_translate_validation():
    # Empty text fails validation with 422
    res = client.post("/api/v1/text-translate", json={"text": "", "source_lang": "en", "target_lang": "hi"})
    assert res.status_code == 422
    data = res.json()
    assert data["success"] is False
    assert data["error"]["code"] == "VALIDATION_ERROR"

def test_v1_audio_stream_path_traversal():
    res = client.get("/api/v1/audio/../../etc/passwd")
    assert res.status_code in [403, 404]

def test_websocket_live_translate_handshake():
    with client.websocket_connect("/ws/live-translate") as ws:
        # Send flush action
        ws.send_text(json.dumps({"action": "flush"}))
        # Connection established and flushed without exception
