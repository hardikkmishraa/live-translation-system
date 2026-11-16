import os
import time
import math
import psutil
from typing import Dict, Any, List, Tuple
import numpy as np
import soundfile as sf
import torch

from backend.app.core.audio_processor import AudioProcessor
from backend.app.models.asr import WhisperASRModel
from backend.app.models.ser import SpeechEmotionRecognizer
from backend.app.models.nmt import BilingualNMTModel
from backend.app.models.tts import EmotionPreservingTTS
from backend.app.models.emotion_preservation import LatentDepthEmotionPreserver
from backend.app.services.evaluation import EvaluationService
from backend.app.services.model_lifecycle import model_lifecycle
from backend.app.utils.logger import logger

def compute_wer(reference: str, hypothesis: str) -> float:
    """Calculates Levenshtein-based Word Error Rate (WER)."""
    ref_words = reference.strip().lower().split()
    hyp_words = hypothesis.strip().lower().split()

    if not ref_words:
        return 0.0 if not hyp_words else 1.0

    d = np.zeros((len(ref_words) + 1, len(hyp_words) + 1), dtype=np.int32)
    for i in range(len(ref_words) + 1):
        d[i][0] = i
    for j in range(len(hyp_words) + 1):
        d[0][j] = j

    for i in range(1, len(ref_words) + 1):
        for j in range(1, len(hyp_words) + 1):
            if ref_words[i - 1] == hyp_words[j - 1]:
                d[i][j] = d[i - 1][j - 1]
            else:
                d[i][j] = min(
                    d[i - 1][j] + 1,      # deletion
                    d[i][j - 1] + 1,      # insertion
                    d[i - 1][j - 1] + 1   # substitution
                )
    return float(d[len(ref_words)][len(hyp_words)] / len(ref_words))

def compute_sentence_bleu(reference: str, hypothesis: str, max_n: int = 4) -> float:
    """Computes sentence-level BLEU-n with brevity penalty."""
    ref_tokens = reference.strip().lower().split()
    hyp_tokens = hypothesis.strip().lower().split()

    if not hyp_tokens or not ref_tokens:
        return 0.0

    bp = 1.0
    if len(hyp_tokens) < len(ref_tokens):
        bp = math.exp(1.0 - (len(ref_tokens) / len(hyp_tokens)))

    precisions = []
    for n in range(1, max_n + 1):
        if len(hyp_tokens) < n or len(ref_tokens) < n:
            precisions.append(1e-9)
            continue

        ref_ngrams = {}
        for i in range(len(ref_tokens) - n + 1):
            ng = tuple(ref_tokens[i:i + n])
            ref_ngrams[ng] = ref_ngrams.get(ng, 0) + 1

        hyp_ngrams = {}
        for i in range(len(hyp_tokens) - n + 1):
            ng = tuple(hyp_tokens[i:i + n])
            hyp_ngrams[ng] = hyp_ngrams.get(ng, 0) + 1

        clipped_matches = sum(min(count, ref_ngrams.get(ng, 0)) for ng, count in hyp_ngrams.items())
        total_hyp_ngrams = sum(hyp_ngrams.values())
        precisions.append(clipped_matches / total_hyp_ngrams if total_hyp_ngrams > 0 else 1e-9)

    log_sum = sum(math.log(max(p, 1e-9)) for p in precisions) / len(precisions)
    return float(bp * math.exp(log_sum))

class EmpiricalSystemBenchmark:
    """
    Empirical Scientific Benchmark for VoiceAffect (Phase 7).
    Executes real inference across all stages, measuring actual WER, BLEU,
    emotion consistency, physical naturalness, latency, and memory footprint.
    """

    def __init__(self):
        self.audio_proc = AudioProcessor()
        self.asr = model_lifecycle.get_asr()
        self.ser = model_lifecycle.get_ser()
        self.nmt = model_lifecycle.get_nmt()
        self.tts = model_lifecycle.get_tts()
        self.eval_service = EvaluationService()
        self.preserver = LatentDepthEmotionPreserver()

    def run_full_benchmark(self) -> Dict[str, Any]:
        logger.info("Executing comprehensive empirical evaluation...")
        timestamp_start = time.time()

        # 1. Measure Memory and Hardware Resource Consumption
        process = psutil.Process(os.getpid())
        mem_info = process.memory_info()
        ram_rss_mb = round(mem_info.rss / (1024 * 1024), 2)
        mps_allocated_mb = 0.0
        if torch.backends.mps.is_available():
            try:
                mps_allocated_mb = round(torch.mps.current_allocated_memory() / (1024 * 1024), 2)
            except Exception:
                mps_allocated_mb = 0.0

        # 2. Evaluate ASR on Synthetic Paired Samples
        asr_results = self._evaluate_asr()

        # 3. Evaluate NMT (BLEU & Translation Accuracy)
        nmt_results = self._evaluate_nmt()

        # 4. Evaluate SER on Known Acoustic Test Tones
        ser_results = self._evaluate_ser()

        # 5. Evaluate Output Emotion Retention on Synthesized Audio
        retention_results = self._evaluate_emotion_retention()

        # 6. Evaluate Speech Naturalness
        naturalness_results = self._evaluate_naturalness()

        # 7. Evaluate Speaker Similarity (Consent Gated)
        similarity_results = self._evaluate_speaker_similarity()

        # 8. Benchmark End-to-End Latencies
        latency_results = self._benchmark_latencies()

        total_benchmark_time = round(time.time() - timestamp_start, 2)

        return {
            "evaluation_metadata": {
                "system": "VoiceAffect Bilingual Voice Translation System",
                "phase": "Phase 7 - Empirical System Audit & Benchmark",
                "execution_timestamp": time.time(),
                "benchmark_duration_seconds": total_benchmark_time,
                "hardware_device": "Apple Silicon (MPS)" if torch.backends.mps.is_available() else "CPU",
                "python_runtime": f"Python {os.sys.version.split()[0]}",
                "ram_rss_mb": ram_rss_mb,
                "mps_vram_allocated_mb": mps_allocated_mb
            },
            "asr_evaluation": asr_results,
            "nmt_evaluation": nmt_results,
            "ser_evaluation": ser_results,
            "emotion_retention_evaluation": retention_results,
            "speech_naturalness_evaluation": naturalness_results,
            "speaker_similarity_evaluation": similarity_results,
            "latency_and_throughput": latency_results
        }

    def _evaluate_asr(self) -> Dict[str, Any]:
        """Evaluates WER on native synthesized speech; confidence is unavailable in this ASR adapter."""
        if not self.tts._has_native:
            return {
                "model": "openai/whisper-tiny",
                "status": "unavailable",
                "reason": "ASR benchmark audio requires a supported native speech engine; no tone substitute is used.",
                "sample_count": 0,
                "average_wer": None,
                "average_confidence": None,
                "confidence_status": "unavailable_from_pipeline_output",
                "eval_details": [],
            }

        test_samples = [
            ("English Audio", "Hello how are you today", "en"),
            ("Hindi Audio", "नमस्ते आप कैसे हैं", "hi"),
        ]
        eval_items = []
        for label, text, lang in test_samples:
            plan = self.preserver.compute_preservation_plan(
                self.ser.predict_emotion(np.zeros(16000, dtype=np.float32), 16000),
                target_lang=lang,
            )
            tts_res = self.tts.synthesize(text, target_lang=lang, preservation_plan=plan)
            audio_filename = tts_res.details.get("audio_filename")
            if not tts_res.audio_bytes or not audio_filename:
                eval_items.append({
                    "language": lang,
                    "reference": text,
                    "status": "tts_unavailable",
                    "reason": tts_res.details.get("error", "Native TTS returned no audio."),
                })
                continue

            audio_arr, sr = sf.read(self.tts.temp_dir / audio_filename, dtype="float32")
            if audio_arr.ndim > 1:
                audio_arr = np.mean(audio_arr, axis=1)

            asr_res = self.asr.transcribe(audio_arr, sample_rate=sr, language=lang)
            eval_items.append({
                "language": lang,
                "reference": text,
                "hypothesis": asr_res.text,
                "confidence": asr_res.confidence,
                "word_error_rate": round(compute_wer(text, asr_res.text), 3),
                "latency_ms": asr_res.latency_ms,
            })

        valid_wer = [item["word_error_rate"] for item in eval_items if "word_error_rate" in item]
        return {
            "model": "openai/whisper-tiny",
            "status": "completed" if valid_wer else "unavailable",
            "sample_count": len(valid_wer),
            "average_wer": round(float(np.mean(valid_wer)), 3) if valid_wer else None,
            "average_confidence": None,
            "confidence_status": "unavailable_from_pipeline_output",
            "eval_details": eval_items,
        }

    def _evaluate_nmt(self) -> Dict[str, Any]:
        """Evaluates bilingual translation using BLEU scores across Hindi and English."""
        en_to_hi_pairs = [
            ("Hello, how are you?", "नमस्ते, आप कैसे हैं?"),
            ("This is a wonderful day.", "यह एक अद्भुत दिन है।"),
            ("Artificial intelligence is transforming communication.", "कृत्रिम बुद्धिमत्ता संचार को बदल रही है।")
        ]

        hi_to_en_pairs = [
            ("नमस्ते, आप कैसे हैं?", "Hello, how are you?"),
            ("यह बहुत अच्छा समाचार है।", "This is very good news."),
            ("हम वास्तविक समय में अनुवाद कर रहे हैं।", "We are translating in real time.")
        ]

        en_hi_scores = []
        for src, ref in en_to_hi_pairs:
            res = self.nmt.translate(src, source_lang="en", target_lang="hi")
            bleu = compute_sentence_bleu(ref, res.translated_text)
            en_hi_scores.append({
                "source": src, "ref": ref, "hyp": res.translated_text,
                "bleu": round(bleu, 3), "model": res.model_name, "backend_details": res.details,
            })

        hi_en_scores = []
        for src, ref in hi_to_en_pairs:
            res = self.nmt.translate(src, source_lang="hi", target_lang="en")
            bleu = compute_sentence_bleu(ref, res.translated_text)
            hi_en_scores.append({
                "source": src, "ref": ref, "hyp": res.translated_text,
                "bleu": round(bleu, 3), "model": res.model_name, "backend_details": res.details,
            })

        avg_en_hi_bleu = round(float(np.mean([s["bleu"] for s in en_hi_scores])), 3)
        avg_hi_en_bleu = round(float(np.mean([s["bleu"] for s in hi_en_scores])), 3)

        return {
            "model": "configured bilingual NMT service",
            "backends_used": sorted({
                sample["model"] for sample in en_hi_scores + hi_en_scores
            }),
            "en_to_hi_pairs_evaluated": len(en_to_hi_pairs),
            "en_to_hi_average_bleu": avg_en_hi_bleu,
            "hi_to_en_pairs_evaluated": len(hi_to_en_pairs),
            "hi_to_en_average_bleu": avg_hi_en_bleu,
            "samples": {"en_to_hi": en_hi_scores, "hi_to_en": hi_en_scores}
        }

    def _evaluate_ser(self) -> Dict[str, Any]:
        """Sanity-checks SER posterior outputs on synthetic signals, not emotion accuracy."""
        t = np.linspace(0, 1.5, 24000, endpoint=False)
        # 1. High-frequency tense harmonic signal (angry / energetic)
        tone_angry = (0.5 * np.sin(2 * np.pi * 500 * t) + 0.3 * np.sin(2 * np.pi * 1000 * t)).astype(np.float32)
        # 2. Gentle low-frequency tone (calm / neutral)
        tone_calm = (0.3 * np.sin(2 * np.pi * 180 * t)).astype(np.float32)
        # 3. Silence
        silence = np.zeros(24000, dtype=np.float32)

        res_angry = self.ser.predict_emotion(tone_angry, 16000, source_language="en")
        res_calm = self.ser.predict_emotion(tone_calm, 16000, source_language="en")
        res_silence = self.ser.predict_emotion(silence, 16000, source_language="en")

        if any(
            result.top_emotion == "unavailable" or not result.probabilities
            for result in (res_angry, res_calm)
        ):
            return {
                "model": "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition",
                "status": "unavailable",
                "reason": (
                    res_angry.details.get("error")
                    or res_calm.details.get("error")
                    or "SER model is unavailable."
                ),
                "evaluation_type": "synthetic_signal_sanity_check_not_accuracy_evaluation",
                "eval_samples": [],
            }

        return {
            "model": "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition",
            "status": "completed",
            "training_corpus": "RAVDESS (24 English speakers, 8 classes)",
            "license": "Apache 2.0",
            "confidence_metric": "Softmax posterior probability P(c | X)",
            "evaluation_type": "synthetic_signal_sanity_check_not_accuracy_evaluation",
            "eval_samples": [
                {
                    "input": "High-Energy Harmonic Tone",
                    "top_emotion": res_angry.top_emotion,
                    "confidence": res_angry.details["confidence"],
                    "valence": res_angry.valence,
                    "arousal": res_angry.arousal
                },
                {
                    "input": "Low-Energy Sine Tone",
                    "top_emotion": res_calm.top_emotion,
                    "confidence": res_calm.details["confidence"],
                    "valence": res_calm.valence,
                    "arousal": res_calm.arousal
                },
                {
                    "input": "Pure Silence Waveform",
                    "top_emotion": res_silence.top_emotion,
                    "confidence": res_silence.details["confidence"],
                    "silence_fallback": res_silence.details.get("silence_fallback")
                }
            ]
        }

    def _evaluate_emotion_retention(self) -> Dict[str, Any]:
        """
        Empirically analyzes native synthesized speech when a real speech engine is available.
        Never substitutes carrier tones for speech.
        """
        if not self.tts._has_native:
            return {
                "status": "unavailable",
                "reason": "A supported native speech engine is required for generated-speech emotion evaluation.",
            }

        t = np.linspace(0, 1.5, 24000, endpoint=False)
        src_audio = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)

        # Synthesize audio with happy preservation plan
        happy_ser = self.ser.predict_emotion(src_audio, 16000)
        plan = self.preserver.compute_preservation_plan(happy_ser, target_lang="en")
        tts_res = self.tts.synthesize("I am exceptionally excited about this project!", "en", plan)

        audio_filename = tts_res.details.get("audio_filename")
        if not tts_res.audio_bytes or not audio_filename:
            return {
                "status": "unavailable",
                "reason": tts_res.details.get("error", "Native TTS returned no audio."),
            }
        gen_audio, sr = sf.read(self.tts.temp_dir / audio_filename, dtype="float32")
        if gen_audio.ndim > 1:
            gen_audio = np.mean(gen_audio, axis=1)

        consistency = self.eval_service.evaluate_emotion_consistency(src_audio, gen_audio, 16000)

        return {
            "source_predicted_emotion": consistency.source_emotion,
            "source_confidence": consistency.source_confidence,
            "generated_output_emotion": consistency.generated_emotion,
            "generated_output_confidence": consistency.generated_confidence,
            "label_match": consistency.label_match,
            "distribution_cosine_similarity": consistency.distribution_cosine_similarity,
            "retention_verdict": consistency.retention_verdict,
            "retention_explanation": consistency.retention_explanation
        }

    def _evaluate_naturalness(self) -> Dict[str, Any]:
        """Measures acoustic proxies only when genuine native synthesized speech is available."""
        if not self.tts._has_native:
            return {
                "status": "unavailable",
                "reason": "A supported native speech engine is required for speech naturalness evaluation.",
            }

        plan = self.preserver.compute_preservation_plan(
            self.ser.predict_emotion(np.zeros(16000, dtype=np.float32), 16000),
            target_lang="hi"
        )
        tts_res = self.tts.synthesize("नमस्ते, यह एक प्राकृतिक ध्वनि परीक्षण है।", "hi", plan)
        audio_filename = tts_res.details.get("audio_filename")
        if not tts_res.audio_bytes or not audio_filename:
            return {
                "status": "unavailable",
                "reason": tts_res.details.get("error", "Native TTS returned no audio."),
            }
        audio_arr, sr = sf.read(self.tts.temp_dir / audio_filename, dtype="float32")
        if audio_arr.ndim > 1:
            audio_arr = np.mean(audio_arr, axis=1)

        nat = self.eval_service.evaluate_speech_naturalness(audio_arr, sr)
        return {
            "tested_voice": "Lekha (hi_IN)",
            "estimated_snr_db": nat.estimated_snr_db,
            "pitch_continuity_ratio": nat.pitch_continuity_ratio,
            "estimated_mos_proxy": nat.estimated_mos_proxy,
            "acoustic_assessment": nat.acoustic_assessment
        }

    def _evaluate_speaker_similarity(self) -> Dict[str, Any]:
        """Evaluates speaker similarity under consent-gated conditions."""
        dummy = np.zeros(1000, dtype=np.float32)
        res_no_consent = self.eval_service.evaluate_speaker_similarity(dummy, dummy, user_consent_granted=False)
        res_consent = self.eval_service.evaluate_speaker_similarity(dummy, dummy, user_consent_granted=True)

        return {
            "without_consent": {
                "similarity_score": res_no_consent.similarity_score,
                "speaker_similarity_retained": res_no_consent.speaker_similarity_retained,
                "status": res_no_consent.evaluation_status,
                "notes": res_no_consent.notes
            },
            "with_consent": {
                "similarity_score": res_consent.similarity_score,
                "speaker_similarity_retained": res_consent.speaker_similarity_retained,
                "status": res_consent.evaluation_status,
                "notes": res_consent.notes
            }
        }

    def _benchmark_latencies(self) -> Dict[str, Any]:
        """Measures per-stage and end-to-end execution latency over repeated trials."""
        latencies = {"asr_ms": [], "ser_ms": [], "nmt_ms": [], "tts_ms": [], "total_ms": []}

        t = np.linspace(0, 1.5, 24000, endpoint=False)
        test_audio = (0.5 * np.sin(2 * np.pi * 300 * t)).astype(np.float32)

        for _ in range(3):
            t0 = time.perf_counter()

            # ASR
            t_asr = time.perf_counter()
            asr_res = self.asr.transcribe(test_audio, 16000, language="en")
            latencies["asr_ms"].append((time.perf_counter() - t_asr) * 1000.0)

            # SER
            t_ser = time.perf_counter()
            ser_res = self.ser.predict_emotion(test_audio, 16000)
            latencies["ser_ms"].append((time.perf_counter() - t_ser) * 1000.0)

            # NMT
            t_nmt = time.perf_counter()
            nmt_res = self.nmt.translate("Hello world", source_lang="en", target_lang="hi")
            latencies["nmt_ms"].append((time.perf_counter() - t_nmt) * 1000.0)

            # TTS latency is only measured when a genuine speech engine is available.
            if self.tts._has_native:
                t_tts = time.perf_counter()
                plan = self.preserver.compute_preservation_plan(ser_res, target_lang="hi")
                self.tts.synthesize("नमस्ते संसार", target_lang="hi", preservation_plan=plan)
                latencies["tts_ms"].append((time.perf_counter() - t_tts) * 1000.0)

            latencies["total_ms"].append((time.perf_counter() - t0) * 1000.0)

        return {
            "iterations": 3,
            "average_asr_ms": round(float(np.mean(latencies["asr_ms"])), 1),
            "average_ser_ms": round(float(np.mean(latencies["ser_ms"])), 1),
            "average_nmt_ms": round(float(np.mean(latencies["nmt_ms"])), 1),
            "average_tts_ms": (
                round(float(np.mean(latencies["tts_ms"])), 1)
                if latencies["tts_ms"] else None
            ),
            "tts_status": "measured" if latencies["tts_ms"] else "unavailable",
            "average_end_to_end_ms": round(float(np.mean(latencies["total_ms"])), 1)
        }

benchmark_service = EmpiricalSystemBenchmark()
