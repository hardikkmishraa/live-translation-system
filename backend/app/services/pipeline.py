import os
import time
import base64
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, asdict, is_dataclass, field
from pathlib import Path
from typing import Dict, Optional, Any, List, Union, Tuple
import numpy as np

from backend.app.config import TTS_ENGINE
from backend.app.core.audio_processor import AudioProcessor
from backend.app.models.asr import WhisperASRModel
from backend.app.models.ser import SpeechEmotionRecognizer
from backend.app.models.ser_iemocap import IEMOCAPSpeechEmotionRecognizer
from backend.app.models.nmt import BilingualNMTModel, SarvamNMTModel
from backend.app.models.emotion_preservation import LatentDepthEmotionPreserver
from backend.app.models.tts import EmotionPreservingTTS
from backend.app.models.base import ASRResult, SERResult, BaseSERModel, NMTResult, EmotionPreservationPlan, TTSResult
from backend.app.services.model_lifecycle import model_lifecycle
from backend.app.services.metrics_tracker import metrics_tracker
from backend.app.utils.logger import logger

@dataclass
class LatencyBreakdown:
    asr_ms: float
    ser_ms: float
    nmt_ms: float
    tts_ms: float
    feature_extraction_ms: float
    total_pipeline_ms: float

@dataclass
class PipelineResult:
    source_language: str
    target_language: str
    source_transcript: str
    translated_text: str
    asr_confidence: Optional[float]
    detected_emotion: Optional[str]
    emotion_confidence: Optional[float]
    emotion_probabilities: Optional[Dict[str, float]]
    native_emotion_scores: Optional[Dict[str, float]]
    valence: Optional[float]
    arousal: Optional[float]
    dominance: Optional[float]
    gating_weight_z: Optional[float]
    preservation_plan: Optional[Dict[str, Any]]
    expressive_synthesis_supported: bool
    emotional_fidelity_status: Optional[str]
    speaker_similarity_retained: bool
    synthesis_voice_used: Optional[str]
    language_limitation_warning: Optional[str]
    original_audio_filename: Optional[str]
    audio_base64: Optional[str]
    audio_filename: Optional[str]
    latency: LatencyBreakdown
    speech_detected: bool
    diagnostic_details: Dict[str, Any]
    audio_url: Optional[str] = None
    request_id: Optional[str] = None
    detected_speaker_gender: Optional[str] = None
    gender_confidence: Optional[float] = None
    asr_status: str = "success"
    ser_status: str = "success"
    ser_model_id: Optional[str] = None
    ser_model_provider: Optional[str] = None
    ser_latency_ms: Optional[float] = None
    emotion_scores: Optional[Dict[str, float]] = None
    translation_provider: Optional[str] = "sarvam"
    translation_model: Optional[str] = None
    translation_status: str = "success"
    tts_engine: Optional[str] = None
    tts_voice: Optional[str] = None
    tts_status: Optional[str] = "success"
    audio_output_reference: Optional[str] = None
    total_processing_duration: float = 0.0
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

class LiveTranslationPipeline:
    """
    Unified Bilingual Voice Translation System With Emotion Awareness (Phase 2 & Phase 5).
    Faithfully integrates:
      1. Preprocessing & VAD (AudioProcessor) + Optional FFT/MFCC extraction
      2. ASR - Whisper Transformer (openai/whisper-tiny)
      3. SER - Pretrained Wav2Vec2 XLSR Speech Emotion Recognition
      4. NMT - Sarvam AI Translation (Hindi <-> English)
      5. Emotion Preservation - Latent-Depth Prosody Gating & Fidelity Reporting
      6. TTS - Speech Synthesis (Lekha [Hindi] / Samantha [English]) + DSP Prosody Modulation
    """

    def __init__(self):
        self.audio_processor = AudioProcessor()

    @property
    def asr(self) -> WhisperASRModel:
        return model_lifecycle.get_asr()

    @property
    def ser(self) -> BaseSERModel:
        return model_lifecycle.get_ser()

    @property
    def nmt(self) -> Union[BilingualNMTModel, SarvamNMTModel]:
        return model_lifecycle.get_nmt()

    @property
    def emotion_preserver(self) -> LatentDepthEmotionPreserver:
        return model_lifecycle.get_gating()

    @property
    def tts(self) -> EmotionPreservingTTS:
        return model_lifecycle.get_tts()

    def _run_asr_safe(
        self,
        audio: np.ndarray,
        sample_rate: int,
        language: Optional[str]
    ) -> Tuple[ASRResult, str, Optional[str], Optional[str]]:
        """Executes Whisper ASR with defensive exception isolation."""
        try:
            asr_res = self.asr.transcribe(audio, sample_rate, language=language)
            if not asr_res.text.strip():
                return asr_res, "no_speech_detected", "ASR produced no transcription text.", None
            return asr_res, "success", None, None
        except Exception as e:
            logger.error(f"ASR transcription failed: {e}")
            asr_res = ASRResult(
                text="",
                language=language or "unknown",
                confidence=None,
                latency_ms=0.0,
                model_name=getattr(self.asr, "model_id", "openai/whisper-tiny"),
                details={"status": "failed", "error": str(e)}
            )
            return asr_res, "failed", None, f"ASR failed: {str(e)}"

    def _run_ser_safe(
        self,
        audio: np.ndarray,
        sample_rate: int,
        source_language: Optional[str]
    ) -> Tuple[SERResult, str, Optional[str]]:
        """Executes Wav2Vec2 SER with defensive exception isolation."""
        try:
            ser_res = self.ser.predict_emotion(audio, sample_rate, source_language=source_language)
            ser_status = "failed" if ser_res.top_emotion == "unavailable" else "success"
            warning = None
            if ser_status == "failed" and "error" in ser_res.details:
                warning = f"SER unavailable: {ser_res.details['error']}"
            return ser_res, ser_status, warning
        except Exception as e:
            logger.error(f"SER execution failed: {e}")
            ser_res = SERResult(
                top_emotion="unavailable",
                probabilities={},
                valence=0.5,
                arousal=0.5,
                dominance=0.5,
                latency_ms=0.0,
                model_name="ser_unavailable",
                is_demo=False,
                details={"status": "failed", "error": str(e)}
            )
            return ser_res, "failed", f"SER execution error: {str(e)}"

    def process_audio(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
        forced_source_lang: Optional[str] = None,
        forced_target_lang: Optional[str] = None,
        synthesize_speech: bool = True,
        original_audio_filename: Optional[str] = None,
        user_consent_for_voice_cloning: bool = False,
        request_id: Optional[str] = None
    ) -> PipelineResult:
        t_pipeline_start = time.perf_counter()
        req_id = request_id or uuid.uuid4().hex
        warnings: List[str] = []
        errors: List[str] = []

        # Step 1: Preprocessing & Voice Activity Detection (VAD)
        t_vad_start = time.perf_counter()
        proc_audio, has_speech = self.audio_processor.trim_silence(audio, threshold=0.007)
        t_vad_ms = (time.perf_counter() - t_vad_start) * 1000.0

        if not has_speech or len(proc_audio) < 1600:  # Less than 100ms
            total_ms = (time.perf_counter() - t_pipeline_start) * 1000.0
            warnings.append("No audible speech detected; silence gate active.")
            return PipelineResult(
                request_id=req_id,
                source_language=forced_source_lang or "unknown",
                target_language=forced_target_lang or "unknown",
                source_transcript="",
                translated_text="",
                asr_status="no_speech_detected",
                ser_status="skipped_silence",
                ser_model_id=None,
                ser_model_provider=getattr(self.ser, "_provider_name", "iemocap"),
                ser_latency_ms=0.0,
                asr_confidence=0.0,
                detected_emotion=None,
                emotion_confidence=None,
                emotion_probabilities=None,
                native_emotion_scores={},
                emotion_scores={},
                valence=None,
                arousal=None,
                dominance=None,
                gating_weight_z=None,
                preservation_plan=None,
                expressive_synthesis_supported=False,
                emotional_fidelity_status="No audible speech detected; silence gate active.",
                speaker_similarity_retained=False,
                translation_provider="sarvam",
                translation_model=None,
                translation_status="noop",
                tts_engine=TTS_ENGINE,
                tts_voice=None,
                tts_status="skipped",
                synthesis_voice_used=None,
                language_limitation_warning=None,
                original_audio_filename=original_audio_filename,
                audio_output_reference=None,
                audio_base64=None,
                audio_filename=None,
                latency=LatencyBreakdown(0.0, 0.0, 0.0, 0.0, 0.0, round(total_ms, 2)),
                total_processing_duration=round(total_ms, 2),
                speech_detected=False,
                warnings=warnings,
                errors=errors,
                diagnostic_details={"status": "silence_or_below_vad_threshold"}
            )

        # Step 2: Diagnostic Feature Extraction & Speaker Gender Classification
        t_feat_start = time.perf_counter()
        fft_mfcc = self.audio_processor.compute_fft_and_mfcc(proc_audio, sample_rate)
        prosody = self.audio_processor.extract_prosody_features(proc_audio, sample_rate)
        gender_analysis = self.audio_processor.detect_speaker_gender(proc_audio, sample_rate)
        detected_gender = gender_analysis.get("gender", "female")
        gender_conf = gender_analysis.get("confidence", 0.8)
        feat_ms = (time.perf_counter() - t_feat_start) * 1000.0

        # Step 3 & 4: Inference of ASR and SER
        # Benchmarked for Phase 8: On Apple Silicon MPS, concurrent execution causes Metal GPU queue contention (+36% slower, 505ms vs 370ms).
        # Hence sequential is the faster, optimal execution model for macOS M-series unified memory.
        run_concurrent = os.environ.get("CONCURRENT_ASR_SER", "false").lower() == "true"
        t_asr_ser_start = time.perf_counter()
        if run_concurrent and forced_source_lang:
            with ThreadPoolExecutor(max_workers=2) as executor:
                future_asr = executor.submit(self._run_asr_safe, proc_audio, sample_rate, forced_source_lang)
                future_ser = executor.submit(self._run_ser_safe, proc_audio, sample_rate, forced_source_lang)
                asr_res, asr_status, asr_warn, asr_err = future_asr.result()
                ser_res, ser_status, ser_warn = future_ser.result()
        else:
            asr_res, asr_status, asr_warn, asr_err = self._run_asr_safe(proc_audio, sample_rate, forced_source_lang)
            ser_lang = forced_source_lang or asr_res.language
            ser_res, ser_status, ser_warn = self._run_ser_safe(proc_audio, sample_rate, ser_lang)

        t_asr_ser_total_ms = (time.perf_counter() - t_asr_ser_start) * 1000.0

        if asr_warn:
            warnings.append(asr_warn)
        if asr_err:
            errors.append(asr_err)
        if ser_warn:
            warnings.append(ser_warn)

        source_lang = forced_source_lang or asr_res.language or "unknown"
        if ser_res.details.get("language_limitation_warning"):
            warnings.append(ser_res.details["language_limitation_warning"])

        # If ASR produced no transcript or failed, return early without translation
        # Requirement: If ASR fails, do not attempt translation with fabricated input.
        if not asr_res.text.strip():
            total_ms = (time.perf_counter() - t_pipeline_start) * 1000.0
            fid_msg = (
                "ASR transcription failed; translation aborted to avoid fabricated input."
                if asr_status == "failed"
                else "Acoustic emotion classified, but no speech transcript recognized for translation."
            )
            return PipelineResult(
                request_id=req_id,
                source_language=source_lang,
                target_language=forced_target_lang or ("en" if source_lang.startswith("hi") else "hi"),
                source_transcript="",
                translated_text="",
                asr_status=asr_status,
                ser_status=ser_status,
                ser_model_id=ser_res.model_name,
                ser_model_provider=ser_res.details.get("model_provider", getattr(self.ser, "_provider_name", "iemocap")),
                ser_latency_ms=ser_res.latency_ms,
                asr_confidence=asr_res.confidence,
                detected_emotion=ser_res.top_emotion,
                emotion_confidence=ser_res.details.get("confidence", 0.0),
                emotion_probabilities=ser_res.probabilities,
                native_emotion_scores=ser_res.details.get("native_8_class_probabilities"),
                emotion_scores=ser_res.probabilities or ser_res.details.get("native_8_class_probabilities"),
                valence=ser_res.valence,
                arousal=ser_res.arousal,
                dominance=ser_res.dominance,
                gating_weight_z=0.5,
                preservation_plan=None,
                detected_speaker_gender=detected_gender,
                gender_confidence=gender_conf,
                expressive_synthesis_supported=False,
                emotional_fidelity_status=fid_msg,
                speaker_similarity_retained=False,
                translation_provider="sarvam",
                translation_model=None,
                translation_status="noop" if asr_status != "failed" else "failed",
                tts_engine=TTS_ENGINE,
                tts_voice=None,
                tts_status="skipped",
                synthesis_voice_used=None,
                language_limitation_warning=ser_res.details.get("language_limitation_warning"),
                original_audio_filename=original_audio_filename,
                audio_output_reference=None,
                audio_base64=None,
                audio_filename=None,
                latency=LatencyBreakdown(
                    asr_ms=asr_res.latency_ms,
                    ser_ms=ser_res.latency_ms,
                    nmt_ms=0.0,
                    tts_ms=0.0,
                    feature_extraction_ms=round(feat_ms, 2),
                    total_pipeline_ms=round(total_ms, 2)
                ),
                total_processing_duration=round(total_ms, 2),
                speech_detected=False,
                warnings=warnings,
                errors=errors,
                diagnostic_details={
                    "status": "no_speech_transcript_recognized" if asr_status != "failed" else "asr_failed",
                    "asr_details": asr_res.details
                }
            )

        # Determine target language: Hindi <-> English bilingual pair
        if forced_target_lang:
            target_lang = forced_target_lang
        else:
            target_lang = "en" if source_lang.startswith("hi") else "hi"

        # Step 5: Neural Machine Translation (Sarvam AI)
        # Requirement: Pass ASR transcription to Sarvam AI. If Sarvam fails, preserve original transcription.
        try:
            nmt_res = self.nmt.translate(asr_res.text, source_lang=source_lang, target_lang=target_lang)
            raw_status = nmt_res.details.get("status", "success" if nmt_res.translated_text else "failed")
            translation_status = "failed" if raw_status in ("failed", "translation_failed") else ("success" if nmt_res.translated_text else "failed")
            if translation_status == "failed":
                err_msg = nmt_res.details.get("error", "Translation request failed")
                errors.append(f"Translation failed: {err_msg}")
        except Exception as e:
            logger.error(f"NMT translation error: {e}")
            translation_status = "failed"
            errors.append(f"Translation exception: {str(e)}")
            nmt_res = NMTResult(
                source_text=asr_res.text,
                translated_text="",
                source_lang=source_lang,
                target_lang=target_lang,
                latency_ms=0.0,
                model_name="sarvam_error",
                details={"status": "translation_failed", "error": str(e), "provider": "sarvam"}
            )

        # Step 6: Emotion Preservation Service (Latent-Depth Prosody Planning & Gender Matching)
        preservation_plan = self.emotion_preserver.compute_preservation_plan(
            ser_result=ser_res,
            target_lang=target_lang,
            user_consent_for_voice_cloning=user_consent_for_voice_cloning,
            speaker_gender=detected_gender,
            gender_confidence=gender_conf
        )

        # Step 7: Text-to-Speech Synthesis (TTS) with Prosody Conditioning & Matched Gender
        # Requirement: If TTS fails, still return translated text.
        tts_audio_b64 = None
        tts_audio_fname = None
        tts_res_obj = None
        tts_status = "skipped"

        if synthesize_speech and nmt_res.translated_text:
            try:
                tts_res_obj = self.tts.synthesize(
                    text=nmt_res.translated_text,
                    target_lang=target_lang,
                    preservation_plan=preservation_plan,
                    speaker_gender=detected_gender
                )
                if tts_res_obj.audio_bytes:
                    tts_status = "success"
                    tts_audio_b64 = base64.b64encode(tts_res_obj.audio_bytes).decode("ascii")
                    tts_audio_fname = tts_res_obj.details.get("audio_filename")
                else:
                    tts_status = "failed" if "error" in tts_res_obj.details else "unavailable"
                    if "error" in tts_res_obj.details:
                        warnings.append(f"TTS synthesis failed: {tts_res_obj.details['error']}")
            except Exception as e:
                logger.error(f"TTS synthesis error: {e}")
                tts_status = "failed"
                warnings.append(f"TTS execution exception: {str(e)}")
                tts_res_obj = None

        tts_ms = tts_res_obj.latency_ms if tts_res_obj else 0.0
        total_pipeline_ms = (time.perf_counter() - t_pipeline_start) * 1000.0

        latency = LatencyBreakdown(
            asr_ms=asr_res.latency_ms,
            ser_ms=ser_res.latency_ms,
            nmt_ms=nmt_res.latency_ms,
            tts_ms=tts_ms,
            feature_extraction_ms=round(feat_ms, 2),
            total_pipeline_ms=round(total_pipeline_ms, 2)
        )

        # Convert preservation_plan to dictionary
        if preservation_plan is None:
            preservation_plan_dict = None
        elif isinstance(preservation_plan, dict):
            preservation_plan_dict = preservation_plan
        elif is_dataclass(preservation_plan) and not isinstance(preservation_plan, type):
            preservation_plan_dict = asdict(preservation_plan)
        elif hasattr(preservation_plan, "model_dump"):
            preservation_plan_dict = preservation_plan.model_dump()
        else:
            preservation_plan_dict = dict(preservation_plan)

        if not (tts_res_obj is not None and tts_res_obj.audio_bytes):
            if preservation_plan_dict is not None:
                preservation_plan_dict["applied_to_tts"] = False
                preservation_plan_dict["synthesis_voice_used"] = "none"

        tts_audio_url = f"/api/v1/audio/{tts_audio_fname}" if tts_audio_fname else None

        translation_provider = nmt_res.details.get("provider", "sarvam")
        translation_model = nmt_res.details.get("model", nmt_res.model_name)
        tts_voice = (
            tts_res_obj.voice_used
            if tts_res_obj is not None and tts_res_obj.audio_bytes
            else None
        )

        # Operational telemetry recording for Phase 6 Research Dashboard
        is_success = (translation_status == "success") and (asr_status != "failed")
        metrics_tracker.record_request(
            request_id=req_id,
            source_lang=source_lang,
            target_lang=target_lang,
            success=is_success,
            latencies={
                "asr_ms": asr_res.latency_ms,
                "ser_ms": ser_res.latency_ms,
                "nmt_ms": nmt_res.latency_ms,
                "tts_ms": tts_ms,
                "total_ms": total_pipeline_ms
            },
            detected_emotion=ser_res.top_emotion,
            tts_engine=TTS_ENGINE,
            error_message=errors[0] if errors else None
        )

        # Phase 8 Structured Latency Audit Log
        logger.info(
            f"[LATENCY_AUDIT] req_id={req_id} | total={total_pipeline_ms:.1f}ms | "
            f"vad={t_vad_ms:.1f}ms | feat={feat_ms:.1f}ms | asr={asr_res.latency_ms:.1f}ms | "
            f"ser={ser_res.latency_ms:.1f}ms | parallel_asr_ser={t_asr_ser_total_ms:.1f}ms | "
            f"nmt={nmt_res.latency_ms:.1f}ms | tts={tts_ms:.1f}ms"
        )

        return PipelineResult(
            request_id=req_id,
            source_language=source_lang,
            target_language=target_lang,
            source_transcript=asr_res.text,
            translated_text=nmt_res.translated_text,
            asr_status=asr_status,
            ser_status=ser_status,
            ser_model_id=ser_res.model_name,
            ser_model_provider=ser_res.details.get("model_provider", getattr(self.ser, "_provider_name", "iemocap")),
            ser_latency_ms=ser_res.latency_ms,
            asr_confidence=asr_res.confidence,
            detected_emotion=ser_res.top_emotion,
            detected_speaker_gender=detected_gender,
            gender_confidence=gender_conf,
            emotion_confidence=ser_res.details.get("confidence", 0.0),
            emotion_probabilities=ser_res.probabilities,
            native_emotion_scores=ser_res.details.get("native_8_class_probabilities"),
            emotion_scores=ser_res.probabilities or ser_res.details.get("native_8_class_probabilities"),
            valence=ser_res.valence,
            arousal=ser_res.arousal,
            dominance=ser_res.dominance,
            gating_weight_z=preservation_plan.gating_weight_z,
            preservation_plan=preservation_plan_dict,
            expressive_synthesis_supported=preservation_plan.expressive_synthesis_supported,
            emotional_fidelity_status=preservation_plan.emotional_fidelity_status,
            speaker_similarity_retained=preservation_plan.speaker_similarity_retained,
            translation_provider=translation_provider,
            translation_model=translation_model,
            translation_status=translation_status,
            tts_engine=tts_res_obj.details.get("provider", TTS_ENGINE) if (tts_res_obj and tts_res_obj.audio_bytes) else TTS_ENGINE,
            tts_voice=tts_voice,
            tts_status=tts_status,
            synthesis_voice_used=tts_voice or preservation_plan.synthesis_voice_used,
            language_limitation_warning=ser_res.details.get("language_limitation_warning"),
            original_audio_filename=original_audio_filename,
            audio_output_reference=tts_audio_url,
            audio_base64=tts_audio_b64,
            audio_filename=tts_audio_fname,
            audio_url=tts_audio_url,
            latency=latency,
            total_processing_duration=round(total_pipeline_ms, 2),
            speech_detected=True,
            warnings=warnings,
            errors=errors,
            diagnostic_details={
                "mfcc_summary": {
                    "num_frames": fft_mfcc["num_frames"],
                    "spectral_centroid": fft_mfcc["spectral_centroid"],
                    "pitch_f0_hz": prosody["pitch_f0_hz"]
                },
                "speaker_gender": gender_analysis,
                "parallel_asr_ser_ms": round(t_asr_ser_total_ms, 2),
                "asr_model": asr_res.model_name,
                "ser_model": ser_res.model_name,
                "nmt_model": nmt_res.model_name,
                "nmt_details": nmt_res.details,
                "tts_model": tts_res_obj.model_name if tts_res_obj else "none",
                "tts_details": tts_res_obj.details if tts_res_obj else {},
                "emotion_fidelity_verdict": preservation_plan.emotional_fidelity_status
            }
        )

    def get_system_health(self) -> Dict[str, Any]:
        return {
            "asr": asdict(self.asr.get_health()),
            "ser": asdict(self.ser.get_health()),
            "nmt": asdict(self.nmt.get_health()),
            "tts": asdict(self.tts.get_health()),
            "audio_processor": {
                "sample_rate": self.audio_processor.target_sample_rate,
                "n_mfcc": self.audio_processor.n_mfcc,
                "n_mels": self.audio_processor.n_mels
            }
        }
