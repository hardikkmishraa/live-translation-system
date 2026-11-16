import io
import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient
from backend.app.main import app

client = TestClient(app)

def test_api_health():
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "device" in data
    assert "modules" in data

def test_api_text_translation():
    payload = {
        "text": "Hello, how are you?",
        "source_lang": "en",
        "target_lang": "hi",
        "emotion": "happy",
        "synthesize_speech": True
    }
    response = client.post("/api/text-translate", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["translated_text"] != ""
    assert data["emotion"] in ("happy", "unavailable", None)

def test_api_translate_audio_file():
    # Synthesize small test WAV file in memory
    sr = 16000
    t = np.linspace(0, 0.5, int(sr * 0.5), endpoint=False)
    tone = (0.5 * np.sin(2 * np.pi * 300 * t)).astype(np.float32)

    buf = io.BytesIO()
    sf.write(buf, tone, sr, format="WAV", subtype="PCM_16")
    buf.seek(0)

    files = {"file": ("test_sample.wav", buf, "audio/wav")}
    response = client.post(
        "/api/translate-audio",
        files=files,
        data={"source_lang": "en", "target_lang": "hi", "synthesize_speech": "true"}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert "detected_emotion" in data
    assert "latency" in data
    assert "total_pipeline_ms" in data["latency"]
