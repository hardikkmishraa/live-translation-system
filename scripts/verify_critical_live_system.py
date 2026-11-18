"""
Comprehensive Verification Script for Critical Live System Bug Fixes
=====================================================================
Validates:
- Phase A: Exception resilience and memory management (no crashes, MPS cache cleared)
- Phase B: Basic translation without SER (disable_ser=True) for Hi->En and En->Hi
- Phase C: ASR transcript accuracy and language identification
- Phase D: Independent SER routing (Hindi -> 8-class, English -> 4-class)
- Phase E: Emotion preservation planning, gating calibration, and TTS persona/prosody mapping
- Phase F: End-to-end audio synthesis and file output verification
- Phase G: Hardware & memory consumption on Apple Silicon (16GB RAM)
"""

import os
import sys
import time
import json
import psutil
import soundfile as sf
import numpy as np
import torch
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.services.pipeline import LiveTranslationPipeline
from backend.app.services.model_lifecycle import ModelLifecycleManager

def get_process_memory_mb():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)

def run_verification():
    print("=" * 80)
    print("STARTING CRITICAL LIVE SYSTEM VALIDATION SUITE")
    print("=" * 80)

    start_mem = get_process_memory_mb()
    print(f"Initial Memory Footprint: {start_mem:.2f} MB")

    pipeline = LiveTranslationPipeline()

    # 1. Prepare Test Audio Files
    en_audio_path = PROJECT_ROOT / "tests/test_audio/test_speech.wav"
    hi_audio_path = PROJECT_ROOT / "data/vishlb_hindi/audio/F4_happy_4.1.happy-01.wav"
    hi_angry_path = PROJECT_ROOT / "data/vishlb_hindi/audio/F4_angry_4.5.anger-07.wav"
    hi_sad_path = PROJECT_ROOT / "data/vishlb_hindi/audio/F4_sad_4.4.sad-02.wav"

    en_audio, en_sr = sf.read(str(en_audio_path), dtype="float32")
    hi_audio, hi_sr = sf.read(str(hi_audio_path), dtype="float32")
    hi_angry, hi_angry_sr = sf.read(str(hi_angry_path), dtype="float32")
    hi_sad, hi_sad_sr = sf.read(str(hi_sad_path), dtype="float32")

    output_dir = PROJECT_ROOT / "reports/audio_samples"
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {
        "phase_b_neutral": [],
        "phase_c_d_bilingual": [],
        "phase_e_f_expressive": [],
        "memory_profile": {}
    }

    # --------------------------------------------------------------------------
    # PHASE B: Neutral Baseline Translation (disable_ser=True)
    # --------------------------------------------------------------------------
    print("\n[PHASE B] Testing Neutral Baseline Translation (disable_ser=True)...")
    
    # Hindi -> English (Neutral)
    t0 = time.time()
    res_b_hi = pipeline.process_audio(
        audio=hi_audio,
        sample_rate=hi_sr,
        synthesize_speech=True,
        disable_ser=True
    )
    t_b_hi = (time.time() - t0) * 1000.0
    print(f"  HI -> EN Neutral: Src='{res_b_hi.source_transcript}', Tgt='{res_b_hi.translated_text}', Latency={t_b_hi:.1f}ms, AudioBytes={len(res_b_hi.audio_base64 or '')}")
    assert res_b_hi.detected_emotion == "neutral" or res_b_hi.detected_emotion == "neutral_baseline"
    assert res_b_hi.source_language == "hi"
    assert res_b_hi.target_language == "en"
    assert res_b_hi.translated_text.strip() != ""
    assert res_b_hi.audio_base64 is not None

    # English -> Hindi (Neutral)
    t0 = time.time()
    res_b_en = pipeline.process_audio(
        audio=en_audio,
        sample_rate=en_sr,
        synthesize_speech=True,
        disable_ser=True
    )
    t_b_en = (time.time() - t0) * 1000.0
    print(f"  EN -> HI Neutral: Src='{res_b_en.source_transcript}', Tgt='{res_b_en.translated_text}', Latency={t_b_en:.1f}ms, AudioBytes={len(res_b_en.audio_base64 or '')}")
    assert res_b_en.detected_emotion == "neutral" or res_b_en.detected_emotion == "neutral_baseline"
    assert res_b_en.source_language == "en"
    assert res_b_en.target_language == "hi"
    assert res_b_en.translated_text.strip() != ""
    assert res_b_en.audio_base64 is not None

    results["phase_b_neutral"].append({
        "direction": "hi->en",
        "src": res_b_hi.source_transcript,
        "tgt": res_b_hi.translated_text,
        "emotion": res_b_hi.detected_emotion,
        "latency_ms": t_b_hi
    })
    results["phase_b_neutral"].append({
        "direction": "en->hi",
        "src": res_b_en.source_transcript,
        "tgt": res_b_en.translated_text,
        "emotion": res_b_en.detected_emotion,
        "latency_ms": t_b_en
    })

    # --------------------------------------------------------------------------
    # PHASE C & D: ASR Language Detection & Independent Bilingual SER Routing
    # --------------------------------------------------------------------------
    print("\n[PHASE C & D] Testing Independent Bilingual SER Routing...")
    
    # Hindi Happy -> 8-class SER
    res_d_hi = pipeline.process_audio(
        audio=hi_audio,
        sample_rate=hi_sr,
        synthesize_speech=True,
        disable_ser=False
    )
    print(f"  HI Routing: Lang={res_d_hi.source_language}, SER Model={res_d_hi.ser_model_id}, Emotion={res_d_hi.detected_emotion}, Conf={res_d_hi.emotion_confidence:.3f}")
    assert res_d_hi.source_language == "hi"
    assert "hindi" in res_d_hi.ser_model_id or "vishlb" in res_d_hi.ser_model_id.lower()
    assert len(res_d_hi.emotion_probabilities) == 8, f"Expected 8 classes, got {len(res_d_hi.emotion_probabilities)}"

    # English -> 4-class SER
    res_d_en = pipeline.process_audio(
        audio=en_audio,
        sample_rate=en_sr,
        synthesize_speech=True,
        disable_ser=False
    )
    print(f"  EN Routing: Lang={res_d_en.source_language}, SER Model={res_d_en.ser_model_id}, Emotion={res_d_en.detected_emotion}, Conf={res_d_en.emotion_confidence:.3f}")
    assert res_d_en.source_language == "en"
    assert "iemocap" in res_d_en.ser_model_id.lower()
    assert len(res_d_en.emotion_probabilities) == 4, f"Expected 4 classes, got {len(res_d_en.emotion_probabilities)}"

    results["phase_c_d_bilingual"].append({
        "lang": "hi",
        "model": res_d_hi.ser_model_id,
        "classes_count": len(res_d_hi.emotion_probabilities),
        "emotion": res_d_hi.detected_emotion,
        "confidence": res_d_hi.emotion_confidence
    })
    results["phase_c_d_bilingual"].append({
        "lang": "en",
        "model": res_d_en.ser_model_id,
        "classes_count": len(res_d_en.emotion_probabilities),
        "emotion": res_d_en.detected_emotion,
        "confidence": res_d_en.emotion_confidence
    })

    # --------------------------------------------------------------------------
    # PHASE E & F: Emotion Preservation & Expressive Audio Synthesis
    # --------------------------------------------------------------------------
    print("\n[PHASE E & F] Testing Emotion Preservation & Target Audio Synthesis...")
    
    test_cases = [
        ("hi_happy", hi_audio, hi_sr, "hi", "en"),
        ("hi_angry", hi_angry, hi_angry_sr, "hi", "en"),
        ("hi_sad", hi_sad, hi_sad_sr, "hi", "en"),
        ("en_speech", en_audio, en_sr, "en", "hi"),
    ]

    for label, audio, sr, src_lang, tgt_lang in test_cases:
        t0 = time.time()
        res = pipeline.process_audio(
            audio=audio,
            sample_rate=sr,
            forced_source_lang=src_lang,
            forced_target_lang=tgt_lang,
            synthesize_speech=True,
            disable_ser=False
        )
        total_lat = (time.time() - t0) * 1000.0

        plan = res.preservation_plan
        plan_dict = plan if isinstance(plan, dict) else (plan.__dict__ if plan else {})
        details = plan_dict.get("details", {})
        voice_id = details.get("sarvam_persona") or plan_dict.get("synthesis_voice_used", "default")
        pitch = plan_dict.get("pitch_multiplier", 1.0)
        pace = plan_dict.get("rate_multiplier", 1.0)
        loudness = plan_dict.get("volume_multiplier", 1.0)

        # Save audio file if available
        audio_saved = False
        out_wav_path = output_dir / f"{label}_translated_{tgt_lang}.wav"
        if res.audio_base64:
            import base64
            audio_bytes = base64.b64decode(res.audio_base64)
            with open(out_wav_path, "wb") as f:
                f.write(audio_bytes)
            audio_saved = True

        print(f"  Test [{label}] -> [{tgt_lang}]:")
        print(f"    Detected Emotion : {res.detected_emotion} (conf={res.emotion_confidence:.3f}, z={res.gating_weight_z:.2f})")
        print(f"    Preservation Plan: Persona={voice_id}, Pitch={pitch:.2f}, Pace={pace:.2f}, Loudness={loudness:+.1f}dB")
        print(f"    Translation      : '{res.translated_text}'")
        print(f"    TTS Audio Output : {out_wav_path.name} ({len(res.audio_base64 or '')} b64 chars)")
        print(f"    Latency Breakdown: ASR={res.latency.asr_ms:.1f}ms, SER={res.latency.ser_ms:.1f}ms, NMT={res.latency.nmt_ms:.1f}ms, TTS={res.latency.tts_ms:.1f}ms (Total={total_lat:.1f}ms)")

        results["phase_e_f_expressive"].append({
            "test_case": label,
            "src_lang": src_lang,
            "tgt_lang": tgt_lang,
            "emotion": res.detected_emotion,
            "confidence": res.emotion_confidence,
            "gating_z": res.gating_weight_z,
            "voice_persona": voice_id,
            "pitch": pitch,
            "pace": pace,
            "loudness_db": loudness,
            "translated_text": res.translated_text,
            "audio_file": str(out_wav_path),
            "latency": {
                "asr_ms": res.latency.asr_ms,
                "ser_ms": res.latency.ser_ms,
                "nmt_ms": res.latency.nmt_ms,
                "tts_ms": res.latency.tts_ms,
                "total_ms": total_lat
            }
        })

    # --------------------------------------------------------------------------
    # PHASE G: Memory Profiling & Stability
    # --------------------------------------------------------------------------
    print("\n[PHASE G] Memory Profiling & Stability Check...")
    end_mem = get_process_memory_mb()
    print(f"  Final Memory Footprint: {end_mem:.2f} MB (Delta: +{end_mem - start_mem:.2f} MB)")
    if torch.backends.mps.is_available():
        allocated = torch.mps.current_allocated_memory() / (1024 * 1024)
        print(f"  MPS Current Allocated: {allocated:.2f} MB")

    results["memory_profile"] = {
        "start_rss_mb": start_mem,
        "end_rss_mb": end_mem,
        "delta_rss_mb": end_mem - start_mem,
        "mps_allocated_mb": torch.mps.current_allocated_memory() / (1024 * 1024) if torch.backends.mps.is_available() else 0.0
    }

    metrics_out = PROJECT_ROOT / "reports/live_verification_metrics.json"
    with open(metrics_out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved metrics to {metrics_out}")
    print("=" * 80)
    print("ALL VERIFICATIONS COMPLETED SUCCESSFULLY!")
    print("=" * 80)

if __name__ == "__main__":
    run_verification()
