"""
Phase 12 Performance Verification & Latency Benchmark Script
============================================================
Measures:
  1. SER Model Loading Time (cold load from disk)
  2. First Inference Latency (cold forward pass)
  3. Subsequent Inference Latency (warm passes: median, p90, min, max)
  4. Memory Usage (RSS in MB via resource / psutil)
  5. Device Used (MPS / CPU)
  6. Total Pipeline Latency Comparison: IEMOCAP vs RAVDESS
Saves output to scripts/benchmark_phase12_results.json
"""

import os
import sys
import time
import json
import psutil
from pathlib import Path
import numpy as np
import soundfile as sf
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.config import SER_IEMOCAP_CHECKPOINT_DIR, DEVICE
from backend.app.models.ser_iemocap import IEMOCAPSpeechEmotionRecognizer
from backend.app.services.model_lifecycle import ModelLifecycleManager
from backend.app.services.pipeline import LiveTranslationPipeline


def get_rss_mb() -> float:
    process = psutil.Process(os.getpid())
    return round(process.memory_info().rss / (1024 * 1024), 2)


def main():
    print("=" * 70)
    print("PHASE 12: IEMOCAP SER PERFORMANCE VERIFICATION BENCHMARK")
    print("=" * 70)

    results = {}
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    results["device"] = device
    results["host"] = "Apple Silicon Mac (16GB Unified RAM)"
    results["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")

    # Load real audio sample
    audio_path = PROJECT_ROOT / "tests/test_audio/test_speech.wav"
    audio_data, sr = sf.read(str(audio_path), dtype="float32")
    audio_duration = len(audio_data) / float(sr)
    print(f"Test audio: {audio_path.name} | Samples: {len(audio_data):,} | Duration: {audio_duration:.2f}s | SR: {sr}Hz")

    # 1. Model Loading Time & Memory
    rss_before_load = get_rss_mb()
    print(f"\n[1] Memory before model load: {rss_before_load:.2f} MB")

    t_load_start = time.perf_counter()
    recognizer = IEMOCAPSpeechEmotionRecognizer(checkpoint_dir=SER_IEMOCAP_CHECKPOINT_DIR, device=device)
    recognizer._ensure_loaded()
    load_time_sec = time.perf_counter() - t_load_start

    rss_after_load = get_rss_mb()
    rss_model_footprint = round(rss_after_load - rss_before_load, 2)
    print(f"    Model Loading Time: {load_time_sec:.3f} s")
    print(f"    Memory after load: {rss_after_load:.2f} MB (Delta: +{rss_model_footprint:.2f} MB)")

    results["loading"] = {
        "load_time_seconds": round(load_time_sec, 3),
        "rss_before_load_mb": rss_before_load,
        "rss_after_load_mb": rss_after_load,
        "model_memory_footprint_mb": rss_model_footprint
    }

    # 2. First Inference (Cold)
    t_cold_start = time.perf_counter()
    cold_res = recognizer.predict_emotion(audio_data, sample_rate=sr)
    cold_latency_ms = (time.perf_counter() - t_cold_start) * 1000.0
    print(f"\n[2] First Inference (Cold): {cold_latency_ms:.2f} ms")
    print(f"    Predicted Emotion: {cold_res.top_emotion} (Conf: {cold_res.details['confidence']:.4f})")
    print(f"    Probabilities: {cold_res.probabilities}")

    results["cold_inference"] = {
        "latency_ms": round(cold_latency_ms, 2),
        "top_emotion": cold_res.top_emotion,
        "confidence": cold_res.details["confidence"],
        "probabilities": cold_res.probabilities
    }

    # 3. Subsequent Inferences (Warm: 10 iterations)
    warm_latencies = []
    print(f"\n[3] Running 10 warm inference iterations...")
    for i in range(10):
        t0 = time.perf_counter()
        _ = recognizer.predict_emotion(audio_data, sample_rate=sr)
        lat = (time.perf_counter() - t0) * 1000.0
        warm_latencies.append(lat)

    median_lat = float(np.median(warm_latencies))
    p90_lat = float(np.percentile(warm_latencies, 90))
    min_lat = float(np.min(warm_latencies))
    max_lat = float(np.max(warm_latencies))
    mean_lat = float(np.mean(warm_latencies))

    print(f"    Warm Inference Latency (10 runs):")
    print(f"      Mean   : {mean_lat:.2f} ms")
    print(f"      Median : {median_lat:.2f} ms")
    print(f"      P90    : {p90_lat:.2f} ms")
    print(f"      Min    : {min_lat:.2f} ms | Max: {max_lat:.2f} ms")

    results["warm_inference"] = {
        "iterations": len(warm_latencies),
        "mean_ms": round(mean_lat, 2),
        "median_ms": round(median_lat, 2),
        "p90_ms": round(p90_lat, 2),
        "min_ms": round(min_lat, 2),
        "max_ms": round(max_lat, 2)
    }

    # 4. Pipeline Latency Comparison: IEMOCAP vs RAVDESS
    print(f"\n[4] Full Translation Pipeline Comparison (Real Audio)...")
    from backend.app.services.model_lifecycle import model_lifecycle

    # (A) Pipeline with IEMOCAP
    model_lifecycle.get_ser(provider="iemocap")
    pipeline_iemocap = LiveTranslationPipeline()
    t_iemocap_start = time.perf_counter()
    res_iemocap = pipeline_iemocap.process_audio(
        audio=audio_data,
        sample_rate=sr,
        forced_source_lang="en",
        forced_target_lang="hi",
        synthesize_speech=False
    )
    total_iemocap_ms = (time.perf_counter() - t_iemocap_start) * 1000.0

    print(f"  [A] IEMOCAP Pipeline Result:")
    print(f"      Source Transcript : {res_iemocap.source_transcript}")
    print(f"      Translated Text   : {res_iemocap.translated_text}")
    print(f"      Detected Emotion  : {res_iemocap.detected_emotion} (Model: {res_iemocap.ser_model_id})")
    print(f"      Breakdown: ASR={res_iemocap.latency.asr_ms:.1f}ms | SER={res_iemocap.latency.ser_ms:.1f}ms | NMT={res_iemocap.latency.nmt_ms:.1f}ms")
    print(f"      Total Pipeline Latency: {res_iemocap.latency.total_pipeline_ms:.1f} ms")

    # (B) Pipeline with RAVDESS
    model_lifecycle.get_ser(provider="ravdess")
    pipeline_ravdess = LiveTranslationPipeline()
    t_ravdess_start = time.perf_counter()
    res_ravdess = pipeline_ravdess.process_audio(
        audio=audio_data,
        sample_rate=sr,
        forced_source_lang="en",
        forced_target_lang="hi",
        synthesize_speech=False
    )
    total_ravdess_ms = (time.perf_counter() - t_ravdess_start) * 1000.0

    print(f"\n  [B] RAVDESS Pipeline Result:")
    print(f"      Source Transcript : {res_ravdess.source_transcript}")
    print(f"      Translated Text   : {res_ravdess.translated_text}")
    print(f"      Detected Emotion  : {res_ravdess.detected_emotion} (Model: {res_ravdess.ser_model_id})")
    print(f"      Breakdown: ASR={res_ravdess.latency.asr_ms:.1f}ms | SER={res_ravdess.latency.ser_ms:.1f}ms | NMT={res_ravdess.latency.nmt_ms:.1f}ms")
    print(f"      Total Pipeline Latency: {res_ravdess.latency.total_pipeline_ms:.1f} ms")

    # Restore default IEMOCAP provider
    model_lifecycle.get_ser(provider="iemocap")

    results["pipeline_comparison"] = {
        "iemocap": {
            "model_id": res_iemocap.ser_model_id,
            "detected_emotion": res_iemocap.detected_emotion,
            "ser_ms": res_iemocap.latency.ser_ms,
            "asr_ms": res_iemocap.latency.asr_ms,
            "nmt_ms": res_iemocap.latency.nmt_ms,
            "total_pipeline_ms": res_iemocap.latency.total_pipeline_ms,
            "measured_e2e_ms": round(total_iemocap_ms, 2)
        },
        "ravdess": {
            "model_id": res_ravdess.ser_model_id,
            "detected_emotion": res_ravdess.detected_emotion,
            "ser_ms": res_ravdess.latency.ser_ms,
            "asr_ms": res_ravdess.latency.asr_ms,
            "nmt_ms": res_ravdess.latency.nmt_ms,
            "total_pipeline_ms": res_ravdess.latency.total_pipeline_ms,
            "measured_e2e_ms": round(total_ravdess_ms, 2)
        }
    }

    out_path = PROJECT_ROOT / "scripts/benchmark_phase12_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nBenchmark results saved to: {out_path}")


if __name__ == "__main__":
    main()
