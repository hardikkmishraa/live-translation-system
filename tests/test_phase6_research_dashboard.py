import time
import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.services.metrics_tracker import OperationalMetricsTracker, metrics_tracker

client = TestClient(app)

# =========================================================================
# 1. Operational Metrics Tracker Unit Tests
# =========================================================================

def test_metrics_tracker_recording_and_privacy(tmp_path):
    tracker = OperationalMetricsTracker(storage_path=tmp_path / "test_metrics.json")
    assert tracker.get_overview()["total_requests"] == 0

    # Record successful request (Hindi to English)
    tracker.record_request(
        request_id="req-test-001",
        source_lang="hi",
        target_lang="en",
        success=True,
        latencies={"asr_ms": 120.0, "ser_ms": 85.0, "nmt_ms": 250.0, "tts_ms": 110.0, "total_ms": 565.0},
        detected_emotion="happy",
        tts_engine="macOS say"
    )

    # Record failed request (English to Hindi)
    tracker.record_request(
        request_id="req-test-002",
        source_lang="en",
        target_lang="hi",
        success=False,
        latencies={"asr_ms": 95.0, "ser_ms": 70.0, "nmt_ms": 0.0, "tts_ms": 0.0, "total_ms": 165.0},
        detected_emotion="neutral",
        tts_engine="macOS say",
        error_message="Translation API timeout"
    )

    overview = tracker.get_overview()
    assert overview["total_requests"] == 2
    assert overview["successful_requests"] == 1
    assert overview["failed_requests"] == 1
    assert overview["success_rate_percent"] == 50.0
    assert overview["most_frequently_used_direction"] in ("hi_to_en", "en_to_hi")
    assert overview["direction_distribution"]["hi_to_en"] == 1
    assert overview["direction_distribution"]["en_to_hi"] == 1

    # Verify average pipeline latency
    assert 300.0 < overview["average_pipeline_latency_ms"] < 400.0

    # Privacy verification: Raw audio is NEVER stored
    for req in overview["recent_requests"]:
        assert "audio" not in req
        assert "audio_bytes" not in req
        assert "waveform" not in req

    # Test reset
    tracker.reset_metrics()
    assert tracker.get_overview()["total_requests"] == 0


# =========================================================================
# 2. REST Endpoints Tests (/api/v1/research)
# =========================================================================

def test_api_research_overview_endpoint():
    res = client.get("/api/v1/research/overview")
    assert res.status_code == 200
    data = res.json()

    assert "total_requests" in data
    assert "successful_requests" in data
    assert "failed_requests" in data
    assert "success_rate_percent" in data
    assert "average_pipeline_latency_ms" in data
    assert "most_frequently_used_direction" in data
    assert "direction_distribution" in data
    assert "stage_average_latencies" in data
    assert "recent_requests" in data


def test_api_model_health_endpoint():
    res = client.get("/api/v1/research/model-health")
    assert res.status_code == 200
    data = res.json()

    assert "models" in data
    models = data["models"]

    # ASR (Whisper Tiny)
    assert "asr" in models
    assert "whisper" in models["asr"]["identifier"].lower()
    assert models["asr"]["loading_status"] in ("loaded", "ready_lazy_load")
    assert models["asr"]["availability"] in ("available_and_ready", "ready_lazy_load")

    # SER (Wav2Vec2 XLSR)
    assert "ser" in models
    assert "wav2vec2" in models["ser"]["identifier"].lower()
    assert models["ser"]["checkpoint_status"] == "verified_safetensors_loaded"
    assert models["ser"]["classes_count"] == 8
    assert "limitation_note" in models["ser"]
    assert "RAVDESS" in models["ser"]["training_corpus"]

    # NMT (Sarvam AI)
    assert "nmt" in models
    assert models["nmt"]["provider"] == "Sarvam AI"
    assert models["nmt"]["actual_model"] == "mayura:v1"
    assert "zero silent fallback" in models["nmt"]["fallback_policy"].lower()

    # TTS (macOS say)
    assert "tts" in models
    assert ("macos" in models["tts"]["engine"].lower()) or ("say" in models["tts"]["engine"].lower())
    assert "Lekha" in models["tts"]["available_voices"]["hi"]
    assert "Samantha" in models["tts"]["available_voices"]["en"]
    assert models["tts"]["expressive_synthesis_supported"] is False
    assert "Heuristic" in models["tts"]["expressive_disclosure"]


def test_api_performance_analytics_endpoint():
    res = client.get("/api/v1/research/performance")
    assert res.status_code == 200
    data = res.json()

    assert "averages_ms" in data
    assert "asr_ms" in data["averages_ms"]
    assert "ser_ms" in data["averages_ms"]
    assert "nmt_ms" in data["averages_ms"]
    assert "tts_ms" in data["averages_ms"]
    assert "chart_series" in data
    assert "timestamps" in data["chart_series"]


def test_api_evaluation_metrics_distinctions_and_accuracy():
    res = client.get("/api/v1/research/evaluation-metrics")
    assert res.status_code == 200
    data = res.json()

    # 1. ASR
    assert "asr" in data
    asr_bench = data["asr"]["offline_benchmark"]
    assert asr_bench["word_error_rate_wer"] == 0.142
    assert asr_bench["character_error_rate_cer"] == 0.068
    # Live inference distinction
    assert data["asr"]["live_inference"]["wer"] is None
    assert "unavailable" in data["asr"]["live_inference"]["status"].lower()

    # 2. Translation
    assert "translation" in data
    nmt_bench = data["translation"]["offline_benchmark"]
    assert nmt_bench["bleu_hi_to_en"] == 28.4
    assert nmt_bench["bleu_en_to_hi"] == 26.9
    assert nmt_bench["chrf_hi_to_en"] == 54.2
    assert data["translation"]["live_inference"]["bleu"] is None

    # 3. SER
    assert "ser" in data
    ser_bench = data["ser"]["offline_benchmark"]
    assert ser_bench["accuracy"] == 0.684
    assert ser_bench["precision_macro"] == 0.692
    assert ser_bench["recall_macro"] == 0.684
    assert ser_bench["f1_score_macro"] == 0.681
    assert len(ser_bench["confusion_matrix"]) == 8
    assert len(ser_bench["confusion_matrix"][0]) == 8
    # Cross-lingual disclaimer
    assert "cross_lingual_limitation" in data["ser"]
    assert "RAVDESS" in data["ser"]["cross_lingual_limitation"]["warning"]

    # 4. TTS
    assert "tts" in data
    assert data["tts"]["measured_metrics"]["average_synthesis_time_ms"] > 0
    assert data["tts"]["measured_metrics"]["audio_availability_rate"] == 1.0
    assert data["tts"]["human_evaluation"]["subjective_mos"] is None
    assert "not_collected" in data["tts"]["human_evaluation"]["status"].lower()


def test_api_reset_metrics_endpoint():
    res = client.post("/api/v1/research/reset-metrics")
    assert res.status_code == 200
    assert res.json()["status"] == "success"

    # Verify reset took effect
    overview_res = client.get("/api/v1/research/overview")
    assert overview_res.json()["total_requests"] == 0
