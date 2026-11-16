from typing import Dict, Optional, Any, List
from pydantic import BaseModel, Field

class LatencyBreakdownSchema(BaseModel):
    asr_ms: float = Field(0.0, description="Speech recognition latency")
    ser_ms: float = Field(0.0, description="Emotion classification latency")
    nmt_ms: float = Field(0.0, description="Machine translation latency")
    tts_ms: float = Field(0.0, description="Speech synthesis latency")
    feature_extraction_ms: float = Field(0.0, description="Audio preprocessing and FFT/MFCC extraction latency")
    total_pipeline_ms: float = Field(0.0, description="End-to-end processing latency")

class EmotionPreservationPlanSchema(BaseModel):
    emotion: str = Field(..., description="Target emotional state")
    pitch_multiplier: float = Field(1.0, description="Vocal pitch scaling factor (Δf0)")
    rate_multiplier: float = Field(1.0, description="Speaking tempo multiplier (Δr)")
    volume_multiplier: float = Field(1.0, description="Amplitude gain multiplier (Δv)")
    gating_weight_z: Optional[float] = Field(None, description="Latent-depth gating factor (z_l Eq. 8)")
    applied_to_tts: bool = Field(True, description="Whether prosody plan was applied during synthesis")
    prosody_notes: str = Field(..., description="Acoustic modulation notes")
    expressive_synthesis_supported: bool = Field(False, description="Whether synthesis engine natively supports emotion conditioning")
    emotional_fidelity_status: str = Field(..., description="Truthful report on whether emotional fidelity was achieved in synthesis")
    speaker_similarity_retained: bool = Field(False, description="Whether speaker identity was cloned/retained")
    synthesis_voice_used: str = Field("none", description="Voice profile used for synthesis")
    detected_gender: Optional[str] = Field(None, description="Acoustically detected speaker gender ('male' or 'female')")
    gender_confidence: Optional[float] = Field(None, description="Confidence in speaker gender detection")
    user_consent_granted: bool = Field(False, description="Whether explicit user consent for voice cloning was provided")
    details: Dict[str, Any] = Field(default_factory=dict, description="Methodology status and engine details")

class AudioTranslationResponse(BaseModel):
    success: bool = Field(True)
    request_id: Optional[str] = Field(None, description="Unique transaction identifier for tracing")
    source_language: str = Field(..., description="Detected or provided source language code")
    target_language: str = Field(..., description="Target language code")
    source_transcript: str = Field(..., description="Recognized speech text")
    translated_text: str = Field(..., description="Translated target text")
    original_transcription: Optional[str] = Field(None, description="Alias for source_transcript")
    translated_transcription: Optional[str] = Field(None, description="Alias for translated_text")
    asr_status: Optional[str] = Field("success", description="ASR execution status ('success', 'no_speech_detected', 'failed')")
    ser_status: Optional[str] = Field("success", description="SER execution status ('success', 'failed', 'skipped_silence', 'unavailable')")
    ser_model_id: Optional[str] = Field(None, description="Active SER model identifier ('iemocap-wav2vec2-xlsr-4class', etc.)")
    ser_model_provider: Optional[str] = Field(None, description="Active SER model provider ('iemocap' or 'ravdess')")
    ser_latency_ms: Optional[float] = Field(None, description="SER inference latency in milliseconds")
    asr_confidence: Optional[float] = Field(
        None,
        description="Whisper utterance confidence when exposed by the inference backend; otherwise null."
    )
    detected_emotion: Optional[str] = Field(None, description="Dominant emotion label")
    detected_speaker_gender: Optional[str] = Field(None, description="Acoustically detected speaker gender ('male' or 'female')")
    gender_confidence: Optional[float] = Field(None, description="Confidence in speaker gender detection")
    emotion_confidence: Optional[float] = Field(None, description="Emotion classification confidence (softmax posterior)")
    emotion_probabilities: Optional[Dict[str, float]] = Field(None, description="6-class probability distribution")
    native_emotion_scores: Optional[Dict[str, float]] = Field(None, description="Native 8-class probabilities from Wav2Vec2 SER")
    emotion_scores: Optional[Dict[str, float]] = Field(None, description="Acoustic emotion scores dictionary")
    valence: Optional[float] = Field(None, description="Valence coordinate (0 to 1)")
    arousal: Optional[float] = Field(None, description="Arousal coordinate (0 to 1)")
    dominance: Optional[float] = Field(None, description="Dominance coordinate (0 to 1)")
    gating_weight_z: Optional[float] = Field(None, description="Latent-depth gating weight (Eq. 8)")
    preservation_plan: Optional[EmotionPreservationPlanSchema] = Field(None, description="Emotion preservation plan & fidelity report")
    expressive_synthesis_supported: bool = Field(False, description="Whether expressive synthesis is supported by current voice engine")
    emotional_fidelity_status: Optional[str] = Field(None, description="Report on emotional fidelity achieved in generated audio")
    speaker_similarity_retained: bool = Field(False, description="Whether speaker identity is preserved (requires explicit consent)")
    translation_provider: Optional[str] = Field("sarvam", description="Translation service provider")
    translation_model: Optional[str] = Field(None, description="Translation model identifier")
    translation_status: Optional[str] = Field("success", description="Status of translation stage ('success', 'failed', 'noop')")
    translation_error: Optional[str] = Field(None, description="Translation error details if translation failed")
    tts_engine: Optional[str] = Field(None, description="TTS synthesis engine name")
    tts_voice: Optional[str] = Field(None, description="TTS voice profile name")
    tts_status: Optional[str] = Field("success", description="TTS synthesis status ('success', 'failed', 'unavailable', 'skipped')")
    synthesis_voice_used: Optional[str] = Field(None, description="TTS voice profile used")
    language_limitation_warning: Optional[str] = Field(None, description="Warning if non-English speech was evaluated on English SER")
    original_audio_url: Optional[str] = Field(None, description="URL to play or download original input recording")
    audio_output_reference: Optional[str] = Field(None, description="Reference URL or filename for generated speech audio")
    audio_url: Optional[str] = Field(None, description="URL to download or stream synthesized WAV")
    audio_base64: Optional[str] = Field(None, description="Base64-encoded WAV audio data")
    latency: LatencyBreakdownSchema = Field(...)
    total_processing_duration: Optional[float] = Field(None, description="Total pipeline latency in milliseconds")
    total_pipeline_latency_ms: Optional[float] = Field(None, description="Alias for total_processing_duration in milliseconds")
    speech_detected: bool = Field(True, description="Whether VAD detected active speech")
    warnings: List[str] = Field(default_factory=list, description="Non-fatal execution warnings")
    errors: List[str] = Field(default_factory=list, description="Stage failure errors")
    diagnostics: Dict[str, Any] = Field(default_factory=dict, description="Active model identifiers and execution details")

class TextTranslationRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=2000, description="Input text to translate")
    source_lang: str = Field("en", description="Source language code (en or hi)")
    target_lang: str = Field("hi", description="Target language code (hi or en)")
    emotion: Optional[str] = Field("neutral", description="Target emotion style for prosody heuristics")
    speaker_gender: Optional[str] = Field(None, description="Desired speaker voice gender ('male' or 'female')")
    synthesize_speech: bool = Field(True, description="Whether to synthesize translated speech audio")
    user_consent_voice_cloning: bool = Field(False, description="Explicit user consent for voice conditioning")

class TextTranslationResponse(BaseModel):
    success: bool = Field(True)
    request_id: Optional[str] = Field(None, description="Unique transaction identifier for tracing")
    source_text: str
    translated_text: str
    source_lang: str
    target_lang: str
    emotion: Optional[str] = Field(None, description="Target emotion")
    detected_speaker_gender: Optional[str] = Field(None, description="Speaker voice gender used for synthesis")
    preservation_plan: Optional[EmotionPreservationPlanSchema] = None
    expressive_synthesis_supported: bool = False
    emotional_fidelity_status: Optional[str] = None
    speaker_similarity_retained: bool = False
    translation_provider: Optional[str] = Field("sarvam", description="Translation service provider")
    translation_model: Optional[str] = Field(None, description="Translation model identifier")
    translation_status: Optional[str] = Field("success", description="Status of translation stage ('success', 'failed', 'noop')")
    translation_error: Optional[str] = Field(None, description="Translation error details if translation failed")
    tts_engine: Optional[str] = Field(None, description="TTS synthesis engine name")
    tts_voice: Optional[str] = Field(None, description="TTS voice profile name")
    tts_status: Optional[str] = Field("success", description="TTS synthesis status ('success', 'failed', 'unavailable', 'skipped')")
    synthesis_voice_used: Optional[str] = None
    audio_output_reference: Optional[str] = None
    audio_url: Optional[str] = None
    audio_base64: Optional[str] = None
    latency_ms: float
    total_processing_duration: Optional[float] = None
    warnings: List[str] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)
    diagnostics: Dict[str, Any] = Field(default_factory=dict)
