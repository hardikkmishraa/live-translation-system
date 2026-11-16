import io
import time
import uuid
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, UploadFile, File, Form, HTTPException, status
from fastapi.responses import FileResponse
import soundfile as sf

from backend.app.config import TEMP_STORAGE_DIR, MAX_UPLOAD_SIZE_MB, ALLOWED_AUDIO_EXTENSIONS, TTS_ENGINE
from backend.app.core.ffmpeg_utils import convert_to_wav
from backend.app.core.audio_processor import AudioProcessor
from backend.app.services.pipeline import LiveTranslationPipeline
from backend.app.models.base import SERResult
from backend.app.utils.security import generate_safe_filename, validate_safe_path, safe_unlink
from backend.app.schemas.translation import (
    AudioTranslationResponse,
    TextTranslationRequest,
    TextTranslationResponse,
    LatencyBreakdownSchema,
    EmotionPreservationPlanSchema
)

router = APIRouter()
pipeline = LiveTranslationPipeline()
audio_proc = AudioProcessor()

MAX_DURATION_SECONDS = 180.0

@router.post("/audio/upload", response_model=AudioTranslationResponse, summary="Upload & Translate Audio Recording")
@router.post("/translate-audio", response_model=AudioTranslationResponse, include_in_schema=False)
async def upload_audio_recording(
    file: UploadFile = File(..., description="Audio or video file (max 25MB)"),
    source_lang: Optional[str] = Form(None, description="Source language ('hi', 'en', or null for auto)"),
    target_lang: Optional[str] = Form(None, description="Target language ('hi' or 'en')"),
    synthesize_speech: bool = Form(True, description="Whether to synthesize translated speech audio"),
    consent_voice_cloning: bool = Form(False, description="Explicit user consent for voice cloning (deactivated by default)")
):
    """
    Validates uploaded media format and size, converts to 16kHz mono WAV via sandboxed FFmpeg,
    and runs the full Bilingual Emotion-Aware Speech Translation pipeline.
    Preserves a playable reference to the original recording.
    """
    req_id = uuid.uuid4().hex

    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is missing a valid filename."
        )

    suffix = Path(file.filename).suffix.lower()
    if suffix not in ALLOWED_AUDIO_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported format '{suffix}'. Allowed formats: {', '.join(sorted(ALLOWED_AUDIO_EXTENSIONS))}"
        )

    raw_filename = generate_safe_filename("upload_raw", suffix)
    conv_filename = generate_safe_filename("orig_recording", ".wav")
    raw_path = TEMP_STORAGE_DIR / raw_filename
    conv_path = TEMP_STORAGE_DIR / conv_filename

    success_completed = False
    try:
        # Stream file to disk while validating size limit
        max_bytes = MAX_UPLOAD_SIZE_MB * 1024 * 1024
        total_size = 0
        with open(raw_path, "wb") as buffer:
            while chunk := await file.read(1024 * 64):
                total_size += len(chunk)
                if total_size > max_bytes:
                    raise HTTPException(
                        status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                        detail=f"Uploaded file exceeds maximum limit of {MAX_UPLOAD_SIZE_MB}MB."
                    )
                buffer.write(chunk)

        if total_size == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Uploaded file is empty (0 bytes)."
            )

        # Transcode to standard 16kHz mono PCM WAV via safe FFmpeg
        # Performance optimization: Bypass ffmpeg subprocess when audio is already 16kHz mono WAV
        try:
            needs_transcode = True
            if suffix == ".wav":
                try:
                    info_raw = sf.info(str(raw_path))
                    if info_raw.samplerate == 16000 and info_raw.channels == 1 and info_raw.format == "WAV":
                        import shutil
                        shutil.copyfile(raw_path, conv_path)
                        info = info_raw
                        needs_transcode = False
                except Exception:
                    needs_transcode = True

            if needs_transcode:
                convert_to_wav(raw_path, conv_path, sample_rate=16000)
                info = sf.info(str(conv_path))
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Corrupted or invalid audio file: {e}"
            )

        # Inspect duration limit
        if info.duration > MAX_DURATION_SECONDS:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Audio duration ({info.duration:.1f}s) exceeds maximum limit of {MAX_DURATION_SECONDS}s."
            )

        # Load waveform
        try:
            audio_array, sr = audio_proc.load_audio(conv_path)
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Corrupted or invalid audio content: {e}"
            )

        # Process through full pipeline with Phase 2 & Phase 5 models
        result = pipeline.process_audio(
            audio=audio_array,
            sample_rate=sr,
            forced_source_lang=source_lang,
            forced_target_lang=target_lang,
            synthesize_speech=synthesize_speech,
            original_audio_filename=conv_path.name,
            user_consent_for_voice_cloning=consent_voice_cloning,
            request_id=req_id
        )

        pres_plan_schema = None
        if result.preservation_plan:
            pres_plan_schema = EmotionPreservationPlanSchema(**result.preservation_plan)

        orig_url = f"/api/v1/audio/{conv_path.name}"
        tts_audio_url = f"/api/v1/audio/{result.audio_filename}" if result.audio_filename else None

        response = AudioTranslationResponse(
            success=True,
            request_id=result.request_id,
            source_language=result.source_language,
            target_language=result.target_language,
            source_transcript=result.source_transcript,
            translated_text=result.translated_text,
            original_transcription=result.source_transcript,
            translated_transcription=result.translated_text,
            asr_status=result.asr_status,
            ser_status=result.ser_status,
            ser_model_id=result.ser_model_id,
            ser_model_provider=result.ser_model_provider,
            ser_latency_ms=result.ser_latency_ms,
            asr_confidence=result.asr_confidence,
            detected_emotion=result.detected_emotion,
            detected_speaker_gender=result.detected_speaker_gender,
            gender_confidence=result.gender_confidence,
            emotion_confidence=result.emotion_confidence,
            emotion_probabilities=result.emotion_probabilities,
            native_emotion_scores=result.native_emotion_scores,
            emotion_scores=result.emotion_scores,
            valence=result.valence,
            arousal=result.arousal,
            dominance=result.dominance,
            gating_weight_z=result.gating_weight_z,
            preservation_plan=pres_plan_schema,
            expressive_synthesis_supported=result.expressive_synthesis_supported,
            emotional_fidelity_status=result.emotional_fidelity_status,
            speaker_similarity_retained=result.speaker_similarity_retained,
            translation_provider=result.translation_provider,
            translation_model=result.translation_model,
            translation_status=result.translation_status,
            translation_error=result.diagnostic_details.get("nmt_details", {}).get("error"),
            tts_engine=result.tts_engine,
            tts_voice=result.tts_voice,
            tts_status=result.tts_status,
            synthesis_voice_used=result.synthesis_voice_used,
            language_limitation_warning=result.language_limitation_warning,
            original_audio_url=orig_url,
            audio_output_reference=tts_audio_url or result.audio_output_reference,
            audio_url=tts_audio_url,
            audio_base64=result.audio_base64,
            latency=LatencyBreakdownSchema(
                asr_ms=result.latency.asr_ms,
                ser_ms=result.latency.ser_ms,
                nmt_ms=result.latency.nmt_ms,
                tts_ms=result.latency.tts_ms,
                feature_extraction_ms=result.latency.feature_extraction_ms,
                total_pipeline_ms=result.latency.total_pipeline_ms
            ),
            total_processing_duration=result.total_processing_duration,
            total_pipeline_latency_ms=result.total_processing_duration,
            speech_detected=result.speech_detected,
            warnings=result.warnings,
            errors=result.errors,
            diagnostics=result.diagnostic_details
        )
        success_completed = True
        return response

    finally:
        # Safely clean up raw uploaded file; conv_path is preserved on success for playback
        safe_unlink(raw_path)
        if not success_completed:
            safe_unlink(conv_path)

@router.post("/text-translate", response_model=TextTranslationResponse, summary="Direct Text Translation with Prosody Conditioning")
@router.post("/translate/text", response_model=TextTranslationResponse, include_in_schema=False)
async def translate_text_endpoint(req: TextTranslationRequest):
    """
    Direct sequence-to-sequence text translation with prosody conditioning
    and explicit emotional fidelity reporting.
    """
    t_start = time.perf_counter()
    req_id = uuid.uuid4().hex
    warnings = []
    errors = []

    nmt_res = pipeline.nmt.translate(req.text, req.source_lang, req.target_lang)
    translation_status = nmt_res.details.get("status", "success" if nmt_res.translated_text else "failed")
    if translation_status == "failed":
        err_msg = nmt_res.details.get("error", "Translation failed")
        errors.append(f"Translation failed: {err_msg}")

    # Formulate prosody plan for requested target emotion style
    target_emo = (req.emotion or "neutral").lower()
    ser_sim = SERResult(
        top_emotion=target_emo,
        probabilities={target_emo: 1.0},
        valence=0.70 if target_emo in ("happy", "surprised") else (0.25 if target_emo in ("sad", "angry") else 0.50),
        arousal=0.75 if target_emo in ("happy", "angry", "surprised") else (0.25 if target_emo == "sad" else 0.40),
        dominance=0.70 if target_emo == "angry" else 0.50,
        latency_ms=0.0,
        model_name="simulated_target_intent",
        details={"confidence": 1.0}
    )

    pres_plan = pipeline.emotion_preserver.compute_preservation_plan(
        ser_result=ser_sim,
        target_lang=req.target_lang,
        user_consent_for_voice_cloning=req.user_consent_voice_cloning,
        speaker_gender=req.speaker_gender
    )

    audio_b64 = None
    audio_fname = None
    tts_res = None
    tts_status = "skipped"

    if req.synthesize_speech and nmt_res.translated_text:
        try:
            tts_res = pipeline.tts.synthesize(
                nmt_res.translated_text,
                req.target_lang,
                preservation_plan=pres_plan,
                speaker_gender=req.speaker_gender
            )
            if tts_res.audio_bytes:
                tts_status = "success"
                import base64
                audio_b64 = base64.b64encode(tts_res.audio_bytes).decode("ascii")
                audio_fname = tts_res.details.get("audio_filename")
            else:
                tts_status = "failed" if "error" in tts_res.details else "unavailable"
                if "error" in tts_res.details:
                    warnings.append(f"TTS synthesis failed: {tts_res.details['error']}")
        except Exception as e:
            tts_status = "failed"
            warnings.append(f"TTS execution exception: {str(e)}")

    total_ms = (time.perf_counter() - t_start) * 1000.0
    audio_url = f"/api/v1/audio/{audio_fname}" if audio_fname else None

    return TextTranslationResponse(
        success=bool(nmt_res.translated_text or not req.text.strip()),
        request_id=req_id,
        source_text=req.text,
        translated_text=nmt_res.translated_text,
        source_lang=req.source_lang,
        target_lang=req.target_lang,
        emotion=pres_plan.emotion,
        detected_speaker_gender=req.speaker_gender or (pres_plan.detected_gender if hasattr(pres_plan, "detected_gender") else None),
        preservation_plan=EmotionPreservationPlanSchema(
            emotion=pres_plan.emotion,
            pitch_multiplier=pres_plan.pitch_multiplier,
            rate_multiplier=pres_plan.rate_multiplier,
            volume_multiplier=pres_plan.volume_multiplier,
            gating_weight_z=pres_plan.gating_weight_z,
            applied_to_tts=bool(tts_res is not None and tts_res.audio_bytes),
            prosody_notes=pres_plan.prosody_notes,
            expressive_synthesis_supported=pres_plan.expressive_synthesis_supported,
            emotional_fidelity_status=pres_plan.emotional_fidelity_status,
            speaker_similarity_retained=pres_plan.speaker_similarity_retained,
            synthesis_voice_used=(
                tts_res.voice_used if tts_res is not None and tts_res.audio_bytes else "none"
            ),
            user_consent_granted=pres_plan.user_consent_granted,
            details=pres_plan.details
        ),
        expressive_synthesis_supported=pres_plan.expressive_synthesis_supported,
        emotional_fidelity_status=pres_plan.emotional_fidelity_status,
        speaker_similarity_retained=pres_plan.speaker_similarity_retained,
        translation_provider=nmt_res.details.get("provider", "sarvam"),
        translation_model=nmt_res.details.get("model", nmt_res.model_name),
        translation_status=translation_status,
        translation_error=nmt_res.details.get("error"),
        tts_engine=tts_res.details.get("provider", TTS_ENGINE) if (tts_res and tts_res.audio_bytes) else TTS_ENGINE,
        tts_voice=tts_res.voice_used if tts_res is not None and tts_res.audio_bytes else None,
        tts_status=tts_status,
        synthesis_voice_used=(
            tts_res.voice_used if tts_res is not None and tts_res.audio_bytes else None
        ),
        audio_output_reference=audio_url,
        audio_url=audio_url,
        audio_base64=audio_b64,
        latency_ms=round(total_ms, 2),
        total_processing_duration=round(total_ms, 2),
        warnings=warnings,
        errors=errors,
        diagnostics={
            "nmt_model": nmt_res.model_name,
            "nmt_details": nmt_res.details,
            "tts_model": tts_res.model_name if tts_res else "not_requested",
            "tts_details": tts_res.details if tts_res else {},
        }
    )

@router.get("/audio/{filename}", summary="Stream Synthesized Audio")
async def stream_audio_file(filename: str):
    """
    Serves generated TTS audio files or preserved original recordings securely
    with strict path traversal validation.
    """
    target = TEMP_STORAGE_DIR / Path(filename).name
    validated = validate_safe_path(target, TEMP_STORAGE_DIR)

    if not validated.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Requested audio file not found or has expired."
        )
    return FileResponse(validated, media_type="audio/wav")
