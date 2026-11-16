from typing import Optional, Dict, Any
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, status
from pydantic import BaseModel, Field

from backend.app.services.evaluation import evaluation_service, ComprehensiveEvaluationReport
from backend.app.core.audio_processor import AudioProcessor
from backend.app.utils.logger import logger

router = APIRouter(prefix="/evaluation", tags=["Evaluation"])
audio_processor = AudioProcessor()

class ConsistencyEvaluationResponse(BaseModel):
    source_emotion: str
    source_confidence: float
    generated_emotion: str
    generated_confidence: float
    label_match: bool
    distribution_cosine_similarity: float
    retention_verdict: str
    retention_explanation: str
    generated_speech_naturalness: Dict[str, Any]
    speaker_similarity: Dict[str, Any]

@router.get("/report", summary="Scientific Methodology & Capability Audit")
async def get_evaluation_report() -> Dict[str, Any]:
    """
    Returns a comprehensive scientific evaluation report establishing:
      - Speech Emotion Recognition provenance, training corpus limits, and confidence interpretation
      - Speech Synthesis capabilities (honestly reporting lack of neural emotion/speaker conditioning)
      - Truthful reproducibility status of CAMuLeNet and TrustTranslate latent-depth gating
      - Speaker similarity separation and consent-gating policy
    """
    return evaluation_service.get_comprehensive_report()

@router.post("/consistency", response_model=ConsistencyEvaluationResponse, summary="Evaluate Emotion Consistency Between Source and Synthesized Audio")
async def evaluate_consistency(
    source_file: UploadFile = File(..., description="Source speech audio file"),
    generated_file: UploadFile = File(..., description="Synthesized output speech audio file"),
    user_consent_granted: bool = Form(False, description="User consent for speaker identity assessment")
):
    """
    Empirically evaluates whether emotional characteristics are retained in the synthesized speech output.
    Never assumes emotion retention based on the source emotion alone.
    Re-runs SER on the synthesized audio and checks for label alignment and probability distribution drift.
    """
    try:
        src_bytes = await source_file.read()
        gen_bytes = await generated_file.read()

        src_audio, src_sr = audio_processor.decode_audio(src_bytes, source_file.filename)
        gen_audio, gen_sr = audio_processor.decode_audio(gen_bytes, generated_file.filename)

        # Preprocess both to 16kHz
        src_prep = audio_processor.preprocess_audio(src_audio, src_sr, target_sr=16000)
        gen_prep = audio_processor.preprocess_audio(gen_audio, gen_sr, target_sr=16000)

        consistency = evaluation_service.evaluate_emotion_consistency(
            src_prep.waveform,
            gen_prep.waveform,
            sample_rate=16000
        )

        naturalness = evaluation_service.evaluate_speech_naturalness(
            gen_prep.waveform,
            sample_rate=16000
        )

        speaker_sim = evaluation_service.evaluate_speaker_similarity(
            src_prep.waveform,
            gen_prep.waveform,
            user_consent_granted=user_consent_granted
        )

        return ConsistencyEvaluationResponse(
            source_emotion=consistency.source_emotion,
            source_confidence=consistency.source_confidence,
            generated_emotion=consistency.generated_emotion,
            generated_confidence=consistency.generated_confidence,
            label_match=consistency.label_match,
            distribution_cosine_similarity=consistency.distribution_cosine_similarity,
            retention_verdict=consistency.retention_verdict,
            retention_explanation=consistency.retention_explanation,
            generated_speech_naturalness={
                "estimated_snr_db": naturalness.estimated_snr_db,
                "pitch_continuity_ratio": naturalness.pitch_continuity_ratio,
                "estimated_mos_proxy": naturalness.estimated_mos_proxy,
                "acoustic_assessment": naturalness.acoustic_assessment
            },
            speaker_similarity={
                "similarity_score": speaker_sim.similarity_score,
                "speaker_similarity_retained": speaker_sim.speaker_similarity_retained,
                "evaluation_status": speaker_sim.evaluation_status,
                "consent_requirement_met": speaker_sim.consent_requirement_met,
                "notes": speaker_sim.notes
            }
        )
    except Exception as e:
        logger.error(f"Error in evaluate_consistency: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Consistency evaluation failed: {str(e)}"
        )
