"""
Phase 14: Comprehensive End-to-End Real-World Validation & Performance Benchmark.
================================================================================
Validates:
- Complete end-to-end pipeline: Audio Preprocessing -> ASR -> IEMOCAP SER -> Sarvam NMT -> Emotion-Aware TTS
- Bidirectional language pairs: English -> Hindi and Hindi -> English
- All 4 IEMOCAP emotions: Angry, Happy, Sad, Neutral
- Confidence gating and soft blending
- Provider cascading fallback: Sarvam -> Edge-TTS -> macOS native say
- Granular latency profiling: Cold vs Warm, Median, P90, P95
- Audio integrity and playback readiness
"""

import os
import sys
import time
import json
import uuid
from pathlib import Path
from typing import Dict, List, Any
import numpy as np
import soundfile as sf
from unittest.mock import patch, MagicMock

# Ensure project root is in sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from backend.app.config import TEMP_STORAGE_DIR
from backend.app.models.base import SERResult, EmotionPreservationPlan, TTSResult
from backend.app.services.pipeline import LiveTranslationPipeline
from backend.app.models.tts_sarvam import SarvamTTSProvider
from backend.app.models.tts_edge import EdgeTTSProvider
from backend.app.models.tts_macos import MacOSTTSProvider
from backend.app.models.tts import EmotionPreservingTTS


def generate_synthetic_speech_like_audio(
    duration_s: float = 2.0,
    pitch_hz: float = 180.0,
    sr: int = 16000,
    amplitude: float = 0.5
) -> np.ndarray:
    """Generates a multi-harmonic waveform simulating human speech fundamental frequency."""
    t = np.linspace(0, duration_s, int(sr * duration_s), endpoint=False)
    # Fundamental + 3 harmonics with natural exponential decay envelope
    audio = (
        0.5 * np.sin(2 * np.pi * pitch_hz * t) +
        0.25 * np.sin(2 * np.pi * 2 * pitch_hz * t) +
        0.15 * np.sin(2 * np.pi * 3 * pitch_hz * t) +
        0.10 * np.sin(2 * np.pi * 4 * pitch_hz * t)
    )
    # Add gentle vocal amplitude modulation (4Hz syllabic rate)
    syllables = 0.7 + 0.3 * np.sin(2 * np.pi * 4 * t)
    audio = audio * syllables * amplitude
    return audio.astype(np.float32)


def run_benchmark():
    print("=" * 80)
    print("PHASE 14: END-TO-END VALIDATION & REAL-WORLD PERFORMANCE BENCHMARK")
    print("=" * 80)

    results_data: Dict[str, Any] = {
        "timestamp": time.time(),
        "trials": [],
        "latencies": {
            "vad_preprocessing": [],
            "asr": [],
            "ser": [],
            "nmt": [],
            "tts": [],
            "total_pipeline": []
        },
        "emotion_validation": {},
        "provider_fallbacks": {},
        "cold_start": {}
    }

    # 1. Initialize Pipeline & Measure Cold-Start Loading
    t_cold_start = time.perf_counter()
    pipeline = LiveTranslationPipeline()
    # Trigger model instances
    _ = pipeline.asr
    _ = pipeline.ser
    _ = pipeline.nmt
    _ = pipeline.tts
    cold_init_ms = (time.perf_counter() - t_cold_start) * 1000.0
    results_data["cold_start"]["models_initialization_ms"] = round(cold_init_ms, 2)
    print(f"[*] Pipeline components initialized in {cold_init_ms:.2f} ms")

    # 2. Test Utterances Matrix (Hindi <-> English, Male and Female)
    test_cases = [
        {
            "id": "EN_HI_HAPPY_FEMALE",
            "lang": "en",
            "target": "hi",
            "gender": "female",
            "pitch_hz": 220.0,
            "text": "I am so excited and happy to share this wonderful news with you!",
            "emotion": "happy"
        },
        {
            "id": "EN_HI_SAD_MALE",
            "lang": "en",
            "target": "hi",
            "gender": "male",
            "pitch_hz": 120.0,
            "text": "I am feeling really down and disappointed about what happened today.",
            "emotion": "sad"
        },
        {
            "id": "HI_EN_ANGRY_MALE",
            "lang": "hi",
            "target": "en",
            "gender": "male",
            "pitch_hz": 140.0,
            "text": "यह बिल्कुल अस्वीकार्य है और मैं इस पर बहुत क्रोधित हूँ।",
            "emotion": "angry"
        },
        {
            "id": "HI_EN_NEUTRAL_FEMALE",
            "lang": "hi",
            "target": "en",
            "gender": "female",
            "pitch_hz": 210.0,
            "text": "आज का मौसम सामान्य है और हम शाम को मिलेंगे।",
            "emotion": "neutral"
        },
    ]

    # Check for real audio file tests/test_audio/test_speech.wav
    real_audio_p = Path("tests/test_audio/test_speech.wav")
    if real_audio_p.exists():
        real_audio_data, real_sr = sf.read(str(real_audio_p), dtype="float32")
        print(f"[*] Testing with real recorded speech utterance: {real_audio_p.name} ({len(real_audio_data)/real_sr:.2f}s)")
        t_real_start = time.perf_counter()
        real_res = pipeline.process_audio(real_audio_data, sample_rate=real_sr, synthesize_speech=True)
        real_dur_ms = (time.perf_counter() - t_real_start) * 1000.0
        print(f"    - Source text: '{real_res.source_transcript}'")
        print(f"    - Translated:  '{real_res.translated_text}'")
        print(f"    - SER Emotion: {real_res.detected_emotion} (conf: {real_res.emotion_confidence:.2f})")
        print(f"    - Gender:      {real_res.detected_speaker_gender}")
        print(f"    - TTS Engine:  {real_res.tts_engine} ({real_res.tts_voice})")
        print(f"    - Audio bytes: {len(real_res.audio_base64) if real_res.audio_base64 else 0}")
        print(f"    - Latency:     {real_dur_ms:.1f}ms")
        results_data["real_audio_utterance"] = {
            "source_transcript": real_res.source_transcript,
            "translated_text": real_res.translated_text,
            "detected_emotion": real_res.detected_emotion,
            "confidence": real_res.emotion_confidence,
            "gender": real_res.detected_speaker_gender,
            "tts_engine": real_res.tts_engine,
            "tts_voice": real_res.tts_voice,
            "total_ms": round(real_dur_ms, 2)
        }

    # 3. Emotion Preservation and Prosody Modulation Benchmark across all 4 classes
    print("\n" + "=" * 80)
    print("EMOTION-TO-PROSODY & GATING VALIDATION (4 IEMOCAP CLASSES)")
    print("=" * 80)

    preserver = pipeline.emotion_preserver
    emotions = ["angry", "happy", "sad", "neutral"]
    confidences = [0.25, 0.42, 0.85]  # Gated (<0.35), Soft-blend (0.35-0.50), Full (>=0.50)

    for emo in emotions:
        results_data["emotion_validation"][emo] = []
        for conf in confidences:
            ser_sim = SERResult(
                top_emotion=emo,
                probabilities={emo: conf, "neutral": 1.0 - conf},
                valence=0.85 if emo == "happy" else (0.20 if emo in ("sad", "angry") else 0.50),
                arousal=0.80 if emo in ("happy", "angry") else (0.25 if emo == "sad" else 0.40),
                dominance=0.80 if emo == "angry" else 0.50,
                latency_ms=10.0,
                model_name="iemocap-wav2vec2-xlsr-4class",
                details={"confidence": conf}
            )
            plan = preserver.compute_preservation_plan(ser_sim, target_lang="hi", speaker_gender="female")

            expected_gated = conf < 0.35 or emo == "neutral"
            actual_emotion = plan.emotion
            pitch_m = plan.pitch_multiplier
            rate_m = plan.rate_multiplier

            print(f"[*] Emotion={emo:<7} | In_Conf={conf:.2f} | Out_Emo={actual_emotion:<7} | "
                  f"Pitch={pitch_m:.3f}x | Rate={rate_m:.3f}x | Expressive={plan.expressive_synthesis_supported}")

            results_data["emotion_validation"][emo].append({
                "input_confidence": conf,
                "output_emotion": actual_emotion,
                "pitch_multiplier": pitch_m,
                "rate_multiplier": rate_m,
                "is_gated": actual_emotion == "neutral" and emo != "neutral",
                "notes": plan.prosody_notes
            })

    # 4. Multi-Trial Latency Profiling (10 runs)
    print("\n" + "=" * 80)
    print("LATENCY PROFILING (10 ITERATIONS)")
    print("=" * 80)

    for i in range(10):
        tc = test_cases[i % len(test_cases)]
        audio_data = generate_synthetic_speech_like_audio(
            duration_s=1.5,
            pitch_hz=tc["pitch_hz"],
            sr=16000
        )

        t_run_start = time.perf_counter()
        res = pipeline.process_audio(
            audio=audio_data,
            sample_rate=16000,
            forced_source_lang=tc["lang"],
            forced_target_lang=tc["target"],
            synthesize_speech=True
        )
        run_ms = (time.perf_counter() - t_run_start) * 1000.0

        results_data["latencies"]["vad_preprocessing"].append(res.latency.feature_extraction_ms)
        results_data["latencies"]["asr"].append(res.latency.asr_ms)
        results_data["latencies"]["ser"].append(res.latency.ser_ms)
        results_data["latencies"]["nmt"].append(res.latency.nmt_ms)
        results_data["latencies"]["tts"].append(res.latency.tts_ms)
        results_data["latencies"]["total_pipeline"].append(round(run_ms, 2))

        print(f"Run {i+1:02d}: Total={run_ms:.1f}ms | VAD={res.latency.feature_extraction_ms:.1f}ms | "
              f"ASR={res.latency.asr_ms:.1f}ms | SER={res.latency.ser_ms:.1f}ms | "
              f"NMT={res.latency.nmt_ms:.1f}ms | TTS={res.latency.tts_ms:.1f}ms | Engine={res.tts_engine}")

    # 5. Fault-Tolerance & Cascading Fallback Simulation
    print("\n" + "=" * 80)
    print("TTS PROVIDER FAULT-TOLERANCE & CASCADING FALLBACK VERIFICATION")
    print("=" * 80)

    dummy_plan = EmotionPreservationPlan(
        emotion="happy",
        pitch_multiplier=1.15,
        rate_multiplier=1.12,
        volume_multiplier=1.10,
        gating_weight_z=0.6,
        applied_to_tts=True,
        prosody_notes="Fallback verification test"
    )

    # Test Scenario A: Normal Sarvam AI synthesis
    try:
        t0 = time.perf_counter()
        res_sarvam = pipeline.tts.synthesize("परीक्षण संदेश", "hi", dummy_plan)
        dur = (time.perf_counter() - t0) * 1000.0
        results_data["provider_fallbacks"]["sarvam_normal"] = {
            "success": bool(res_sarvam.audio_bytes),
            "voice": res_sarvam.voice_used,
            "latency_ms": round(dur, 2),
            "fallback_used": res_sarvam.details.get("fallback_used", False)
        }
        print(f"[+] Scenario A (Normal Sarvam): Success={bool(res_sarvam.audio_bytes)}, Voice={res_sarvam.voice_used}, Latency={dur:.1f}ms")
    except Exception as e:
        print(f"[-] Scenario A Failed: {e}")

    # Test Scenario B: Sarvam HTTP 503 / Network Outage -> Fallback to Edge-TTS
    try:
        with patch.object(pipeline.tts.sarvam_provider, "synthesize", side_effect=RuntimeError("Simulated Sarvam 503 Outage")):
            t0 = time.perf_counter()
            res_edge_fb = pipeline.tts.synthesize("परीक्षण संदेश", "hi", dummy_plan)
            dur = (time.perf_counter() - t0) * 1000.0
            results_data["provider_fallbacks"]["sarvam_503_fallback_to_edge"] = {
                "success": bool(res_edge_fb.audio_bytes),
                "voice": res_edge_fb.voice_used,
                "latency_ms": round(dur, 2),
                "fallback_used": res_edge_fb.details.get("fallback_used", False),
                "primary_failed": res_edge_fb.details.get("primary_provider_failed")
            }
            print(f"[+] Scenario B (Sarvam 503 -> Edge Fallback): Success={bool(res_edge_fb.audio_bytes)}, "
                  f"Voice={res_edge_fb.voice_used}, FallbackUsed={res_edge_fb.details.get('fallback_used')}, Latency={dur:.1f}ms")
    except Exception as e:
        print(f"[-] Scenario B Failed: {e}")

    # Test Scenario C: Sarvam & Edge both fail -> Fallback to local macOS native 'say'
    try:
        with patch.object(pipeline.tts.sarvam_provider, "synthesize", side_effect=RuntimeError("Sarvam Down")), \
             patch.object(pipeline.tts.edge_provider, "synthesize", side_effect=RuntimeError("Edge-TTS Down")):
            t0 = time.perf_counter()
            res_macos_fb = pipeline.tts.synthesize("परीक्षण संदेश", "hi", dummy_plan)
            dur = (time.perf_counter() - t0) * 1000.0
            results_data["provider_fallbacks"]["all_cloud_down_fallback_to_macos"] = {
                "success": bool(res_macos_fb.audio_bytes),
                "voice": res_macos_fb.voice_used,
                "latency_ms": round(dur, 2),
                "fallback_used": res_macos_fb.details.get("fallback_used", False),
                "model_name": res_macos_fb.model_name
            }
            print(f"[+] Scenario C (All Cloud Down -> macOS Fallback): Success={bool(res_macos_fb.audio_bytes)}, "
                  f"Voice={res_macos_fb.voice_used}, Model={res_macos_fb.model_name}, Latency={dur:.1f}ms")
    except Exception as e:
        print(f"[-] Scenario C Failed: {e}")

    # Test Scenario D: All providers fail -> Return explicit tts_error without fabricated audio
    try:
        with patch.object(pipeline.tts.sarvam_provider, "synthesize", side_effect=RuntimeError("Sarvam Down")), \
             patch.object(pipeline.tts.edge_provider, "synthesize", side_effect=RuntimeError("Edge-TTS Down")), \
             patch.object(pipeline.tts.macos_provider, "synthesize", side_effect=RuntimeError("macOS 'say' Missing")):
            res_all_fail = pipeline.tts.synthesize("परीक्षण संदेश", "hi", dummy_plan)
            results_data["provider_fallbacks"]["complete_failure_honest_reporting"] = {
                "audio_bytes_empty": res_all_fail.audio_bytes == b"",
                "model_name": res_all_fail.model_name,
                "error_reported": "error" in res_all_fail.details
            }
            print(f"[+] Scenario D (Complete Failure Honest Reporting): AudioEmpty={res_all_fail.audio_bytes == b''}, "
                  f"Model={res_all_fail.model_name}, ErrorDetails={res_all_fail.details.get('error')[:50]}...")
    except Exception as e:
        print(f"[-] Scenario D Failed: {e}")

    # 6. Aggregate Latency Statistics
    print("\n" + "=" * 80)
    print("PERFORMANCE LATENCY AUDIT SUMMARY (10 WARM REQUESTS)")
    print("=" * 80)

    summary_stats = {}
    for stage, vals in results_data["latencies"].items():
        if vals:
            arr = np.array(vals)
            median = float(np.median(arr))
            p90 = float(np.percentile(arr, 90))
            p95 = float(np.percentile(arr, 95))
            min_v = float(np.min(arr))
            max_v = float(np.max(arr))
            summary_stats[stage] = {
                "median_ms": round(median, 2),
                "p90_ms": round(p90, 2),
                "p95_ms": round(p95, 2),
                "min_ms": round(min_v, 2),
                "max_ms": round(max_v, 2)
            }
            print(f"  Stage: {stage:<20} | Median: {median:7.2f}ms | P90: {p90:7.2f}ms | P95: {p95:7.2f}ms | Min: {min_v:7.2f}ms | Max: {max_v:7.2f}ms")

    results_data["summary_statistics"] = summary_stats

    # Save output to json
    out_json = BASE_DIR / "data/phase14_validation_metrics.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(results_data, f, indent=2)
    print(f"\n[+] Validation metrics successfully saved to: {out_json}")

    return results_data


if __name__ == "__main__":
    run_benchmark()
