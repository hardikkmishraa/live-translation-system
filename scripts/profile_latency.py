"""
Profile Voice Translation Pipeline Latency (Phase 8 Baseline Profiling).
Measures each stage independently across multiple trials and reports median, p90, p95, min, max,
and tests sequential vs concurrent ASR+SER execution.
"""
import os
import sys
import time
import json
import statistics
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import soundfile as sf
import torch

# Ensure repo root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.app.config import DEVICE, SARVAM_API_KEY
from backend.app.core.audio_processor import AudioProcessor
from backend.app.services.model_lifecycle import model_lifecycle
from backend.app.services.sarvam_translation import SarvamTranslationService
from backend.app.models.base import SERResult, EmotionPreservationPlan

def run_profiling(n_runs: int = 5):
    print("=" * 70)
    print("PHASE 8: DETAILED PIPELINE LATENCY PROFILING (BASELINE)")
    print(f"Hardware Device: {DEVICE} | PyTorch version: {torch.__version__}")
    print("=" * 70)

    audio_proc = AudioProcessor()
    
    # Pre-warm models (ensure models are loaded once to separate init from inference)
    print("\n[Step 0] Pre-warming models...")
    t0_warm = time.perf_counter()
    asr = model_lifecycle.get_asr()
    ser = model_lifecycle.get_ser()
    tts = model_lifecycle.get_tts()
    gating = model_lifecycle.get_gating()
    sarvam_service = SarvamTranslationService(api_key=SARVAM_API_KEY)
    
    # Load weights
    asr._ensure_loaded()
    ser._ensure_loaded()
    print(f"Models initialized in {(time.perf_counter() - t0_warm)*1000:.1f}ms")

    # Audio files
    audio_en_path = Path("/tmp/profile_en.wav")
    audio_hi_path = Path("/tmp/profile_hi.wav")
    
    stages = {
        "audio_decoding": [],
        "audio_resampling": [],
        "voice_activity_detection": [],
        "feature_and_gender_extraction": [],
        "whisper_asr_inference": [],
        "ser_inference": [],
        "sarvam_api_translation": [],
        "tts_synthesis": [],
        "audio_encoding_b64": [],
        "total_end_to_end_sequential": [],
        "asr_ser_concurrent_total": [],
    }

    test_samples = [
        ("en_to_hi", audio_en_path, "en", "hi"),
        ("hi_to_en", audio_hi_path, "hi", "en"),
    ]

    print(f"\nRunning {n_runs} profiling iterations across language pairs...")

    for run_idx in range(n_runs):
        for pair_name, audio_path, src_lang, tgt_lang in test_samples:
            t_iter_start = time.perf_counter()

            # 1. Audio Decoding
            t0 = time.perf_counter()
            raw_bytes = audio_path.read_bytes()
            data, orig_sr = sf.read(audio_path, dtype="float32")
            if data.ndim > 1:
                data = np.mean(data, axis=1)
            t_decode = (time.perf_counter() - t0) * 1000.0
            stages["audio_decoding"].append(t_decode)

            # 2. Audio Resampling (if needed or simulated)
            t0 = time.perf_counter()
            if orig_sr != 16000:
                audio_16k = audio_proc.resample(data, orig_sr, 16000)
            else:
                audio_16k = data
            audio_16k = audio_proc.normalize_amplitude(audio_16k)
            t_resample = (time.perf_counter() - t0) * 1000.0
            stages["audio_resampling"].append(t_resample)

            # 3. Voice Activity Detection & Silence Trimming
            t0 = time.perf_counter()
            proc_audio, has_speech = audio_proc.trim_silence(audio_16k, threshold=0.007)
            t_vad = (time.perf_counter() - t0) * 1000.0
            stages["voice_activity_detection"].append(t_vad)

            # 4. Feature Extraction & Gender Analysis
            t0 = time.perf_counter()
            fft_mfcc = audio_proc.compute_fft_and_mfcc(proc_audio, 16000)
            prosody = audio_proc.extract_prosody_features(proc_audio, 16000)
            gender_res = audio_proc.detect_speaker_gender(proc_audio, 16000)
            t_feat = (time.perf_counter() - t0) * 1000.0
            stages["feature_and_gender_extraction"].append(t_feat)

            # 5. Whisper ASR Inference
            t0 = time.perf_counter()
            asr_res = asr.transcribe(proc_audio, 16000, language=src_lang)
            t_asr = (time.perf_counter() - t0) * 1000.0
            stages["whisper_asr_inference"].append(t_asr)

            # 6. SER Inference
            t0 = time.perf_counter()
            ser_res = ser.predict_emotion(proc_audio, 16000, source_language=src_lang)
            t_ser = (time.perf_counter() - t0) * 1000.0
            stages["ser_inference"].append(t_ser)

            # 7. Sarvam API Request Duration
            trans_text = asr_res.text.strip() or "Hello, how are you?"
            t0 = time.perf_counter()
            try:
                nmt_dict = sarvam_service.translate(trans_text, src_lang, tgt_lang)
                tgt_text = nmt_dict.get("translated_text", "")
            except Exception as e:
                tgt_text = "Translation placeholder"
            t_nmt = (time.perf_counter() - t0) * 1000.0
            stages["sarvam_api_translation"].append(t_nmt)

            # 8. Emotion Preservation & TTS Synthesis
            plan = gating.compute_preservation_plan(
                ser_result=ser_res,
                target_lang=tgt_lang,
                speaker_gender=gender_res.get("gender", "female")
            )
            t0 = time.perf_counter()
            tts_res = tts.synthesize(tgt_text, tgt_lang, preservation_plan=plan)
            t_tts = (time.perf_counter() - t0) * 1000.0
            stages["tts_synthesis"].append(t_tts)

            # 9. Audio Encoding (Base64)
            t0 = time.perf_counter()
            import base64
            b64_str = base64.b64encode(tts_res.audio_bytes).decode("ascii")
            t_enc = (time.perf_counter() - t0) * 1000.0
            stages["audio_encoding_b64"].append(t_enc)

            # Total sequential
            t_total = (time.perf_counter() - t_iter_start) * 1000.0
            stages["total_end_to_end_sequential"].append(t_total)

            # 10. Test Parallel ASR & SER execution
            t0_conc = time.perf_counter()
            with ThreadPoolExecutor(max_workers=2) as executor:
                future_asr = executor.submit(asr.transcribe, proc_audio, 16000, src_lang)
                future_ser = executor.submit(ser.predict_emotion, proc_audio, 16000, src_lang)
                res_asr = future_asr.result()
                res_ser = future_ser.result()
            t_conc_total = (time.perf_counter() - t0_conc) * 1000.0
            stages["asr_ser_concurrent_total"].append(t_conc_total)

            print(f"  [Run {run_idx+1} {pair_name}] Total: {t_total:.1f}ms | ASR: {t_asr:.1f}ms | SER: {t_ser:.1f}ms | ASR+SER conc: {t_conc_total:.1f}ms | NMT: {t_nmt:.1f}ms | TTS: {t_tts:.1f}ms")

    # Print summary statistics
    print("\n" + "=" * 70)
    print("STAGE-BY-STAGE LATENCY SUMMARY (METRICS OVER ALL TRIALS):")
    print("=" * 70)
    print(f"{'Stage Name':<35} | {'Median (ms)':<10} | {'p90 (ms)':<10} | {'p95 (ms)':<10} | {'Min (ms)':<9} | {'Max (ms)':<9}")
    print("-" * 95)

    results = {}
    for stage_name, vals in stages.items():
        if not vals:
            continue
        vals_sorted = sorted(vals)
        median_val = statistics.median(vals)
        p90_idx = int(0.90 * len(vals_sorted))
        p95_idx = int(0.95 * len(vals_sorted))
        p90_val = vals_sorted[min(p90_idx, len(vals_sorted)-1)]
        p95_val = vals_sorted[min(p95_idx, len(vals_sorted)-1)]
        min_val = vals_sorted[0]
        max_val = vals_sorted[-1]
        
        results[stage_name] = {
            "median": round(median_val, 2),
            "p90": round(p90_val, 2),
            "p95": round(p95_val, 2),
            "min": round(min_val, 2),
            "max": round(max_val, 2),
        }
        print(f"{stage_name:<35} | {median_val:<10.1f} | {p90_val:<10.1f} | {p95_val:<10.1f} | {min_val:<9.1f} | {max_val:<9.1f}")

    # Concurrency comparison
    median_asr = results["whisper_asr_inference"]["median"]
    median_ser = results["ser_inference"]["median"]
    sum_seq = median_asr + median_ser
    median_conc = results["asr_ser_concurrent_total"]["median"]
    print("-" * 95)
    print(f"ASR + SER Sequential Sum: {sum_seq:.1f}ms vs Concurrent: {median_conc:.1f}ms")
    diff = sum_seq - median_conc
    pct = (diff / sum_seq) * 100 if sum_seq > 0 else 0
    print(f"Concurrent speedup: {diff:+.1f}ms ({pct:+.1f}%)")

    # Save to JSON
    out_file = Path("scripts/optimized_latency_profile.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(results, indent=2))
    print(f"\nSaved structured latency metrics to {out_file}")

if __name__ == "__main__":
    run_profiling(n_runs=4)

