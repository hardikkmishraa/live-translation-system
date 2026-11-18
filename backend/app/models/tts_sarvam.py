"""
Sarvam AI Text-to-Speech Provider (bulbul:v4-flash).
=====================================================
Integrates Sarvam AI's low-latency neural TTS API with emotion prosody modulation
(pace, pitch, loudness, and expressive persona mapping).
"""

import base64
import json
import time
import uuid
from typing import Dict, Any, Optional
import httpx
import soundfile as sf
import numpy as np

from backend.app.config import (
    SARVAM_API_KEY,
    SARVAM_TTS_MODEL,
    TEMP_STORAGE_DIR,
    TTS_VOICE_HI_FEMALE,
    TTS_VOICE_HI_MALE,
    TTS_VOICE_EN_FEMALE,
    TTS_VOICE_EN_MALE,
)
from backend.app.models.base import TTSResult, EmotionPreservationPlan, ModelHealthStatus
from backend.app.models.tts_base import BaseTTSProvider
from backend.app.utils.logger import logger


class SarvamTTSProvider(BaseTTSProvider):
    """
    Production-grade Sarvam AI TTS client using the bulbul:v4-flash neural model.
    Supports Hindi ('hi-IN') and Indian English ('en-IN') with expressive prosodic modulation.
    """

    ENDPOINT = "https://api.sarvam.ai/text-to-speech"

    # Expressive persona mapping conditioned on emotion and detected speaker gender
    SPEAKER_PERSONAS = {
        "hi": {
            "female": {
                "happy": "simran_hi_social_excited",
                "sad": "roopa_hi_narration",
                "angry": "ritu_hi_reels",
                "neutral": "kavya_hi_conversational",
                "disgust": "ritu_hi_reels",
                "fear": "roopa_hi_narration",
                "surprise": "simran_hi_social_excited",
                "sarcastic": "ritu_hi_reels",
            },
            "male": {
                "happy": "amit_hi_conversational",
                "sad": "ratan_hi_documentary",
                "angry": "shubh_hi_stories_dramatic",
                "neutral": "aditya_hi_conversational",
                "disgust": "shubh_hi_stories_dramatic",
                "fear": "ratan_hi_documentary",
                "surprise": "amit_hi_conversational",
                "sarcastic": "shubh_hi_stories_dramatic",
            },
        },
        "en": {
            "female": {
                "happy": "sophia_en_conversational",
                "sad": "chandrika_en_stories",
                "angry": "zarina_en_conversation",
                "neutral": "amelia_en_conversational",
                "disgust": "zarina_en_conversation",
                "fear": "chandrika_en_stories",
                "surprise": "sophia_en_conversational",
                "sarcastic": "zarina_en_conversation",
            },
            "male": {
                "happy": "ashwin_en_sports_energetic",
                "sad": "rohan_en_recovery",
                "angry": "rustom_en_suspense",
                "neutral": "dev_en_conversational",
                "disgust": "rustom_en_suspense",
                "fear": "rohan_en_recovery",
                "surprise": "ashwin_en_sports_energetic",
                "sarcastic": "rustom_en_suspense",
            },
        },
    }

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        timeout: float = 12.0,
    ):
        self.api_key = (api_key or SARVAM_API_KEY or "").strip()
        self.model = model or SARVAM_TTS_MODEL
        self.timeout = timeout
        self.temp_dir = TEMP_STORAGE_DIR

    def _select_speaker(self, lang_code: str, gender: str, emotion: str) -> str:
        lang_key = "hi" if lang_code.startswith("hi") else "en"
        gender_key = "male" if gender == "male" else "female"
        emotion_key = emotion.lower()

        # Fallback to configured default if persona mapping not found
        lang_dict = self.SPEAKER_PERSONAS.get(lang_key, {}).get(gender_key, {})
        speaker = lang_dict.get(emotion_key, lang_dict.get("neutral"))

        if not speaker:
            if lang_key == "hi":
                speaker = TTS_VOICE_HI_MALE if gender_key == "male" else TTS_VOICE_HI_FEMALE
            else:
                speaker = TTS_VOICE_EN_MALE if gender_key == "male" else TTS_VOICE_EN_FEMALE
        return speaker

    def synthesize(
        self,
        text: str,
        target_lang: str,
        preservation_plan: EmotionPreservationPlan,
        speaker_gender: Optional[str] = None
    ) -> TTSResult:
        t_start = time.perf_counter()
        clean_text = text.strip()

        if not self.api_key:
            raise RuntimeError("SARVAM_API_KEY is not configured for SarvamTTSProvider.")

        if not clean_text:
            latency_ms = (time.perf_counter() - t_start) * 1000.0
            return TTSResult(
                audio_bytes=b"",
                audio_format="wav",
                sample_rate=16000,
                duration_seconds=0.0,
                voice_used="none",
                latency_ms=round(latency_ms, 2),
                model_name=self.model,
                emotion_applied=preservation_plan.emotion,
            )

        # Sarvam API limits input string to max 500 characters
        if len(clean_text) > 490:
            logger.warning(f"[SarvamTTSProvider] Text length {len(clean_text)} exceeds 490 chars. Truncating to avoid 400 rejection.")
            cut = clean_text[:490].rfind(" ")
            if cut > 200:
                clean_text = clean_text[:cut] + "..."
            else:
                clean_text = clean_text[:487] + "..."

        # Map target language code
        t_code = target_lang.lower().strip()
        is_hindi = t_code.startswith("hi") or t_code == "hin"
        sarvam_lang = "hi-IN" if is_hindi else "en-IN"

        # Determine gender and speaker persona
        gender = (speaker_gender or preservation_plan.detected_gender or "female").strip().lower()
        speaker = self._select_speaker("hi" if is_hindi else "en", gender, preservation_plan.emotion)

        # Acoustic parameter mapping
        # Sarvam bulbul v3 & v4 accepts pitch [-0.5, 0.5], pace [0.5, 2.0], loudness [0.5, 2.0]
        pace = float(np.clip(preservation_plan.rate_multiplier, 0.50, 2.00))
        # Map pitch_multiplier (e.g. 1.15) to additive offset in [-0.5, 0.5]
        pitch_offset = float(np.clip(round((preservation_plan.pitch_multiplier - 1.0) * 1.5, 2), -0.5, 0.5))
        loudness = float(np.clip(preservation_plan.volume_multiplier, 0.50, 2.00))

        headers = {
            "api-subscription-key": self.api_key,
            "Content-Type": "application/json"
        }
        payload = {
            "inputs": [clean_text],
            "target_language_code": sarvam_lang,
            "speaker": speaker,
            "pitch": pitch_offset,
            "pace": pace,
            "loudness": loudness,
            "speech_sample_rate": 16000,
            "enable_preprocessing": True,
            "model": self.model,
        }

        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.post(self.ENDPOINT, headers=headers, json=payload)

            # If bulbul:v4-flash is temporarily unavailable (HTTP 503), try bulbul:v3 with baseline persona
            active_model = self.model
            if response.status_code == 503 and "bulbul:v4-flash" in self.model:
                logger.warning("[SarvamTTSProvider] bulbul:v4-flash 503 unavailable; retrying with bulbul:v3 fallback")
                fallback_model = "bulbul:v3"
                if is_hindi:
                    fallback_speaker = "kavya" if gender == "female" else "aditya"
                else:
                    fallback_speaker = "simran" if gender == "female" else "dev"
                payload["model"] = fallback_model
                payload["speaker"] = fallback_speaker
                with httpx.Client(timeout=self.timeout) as client:
                    response = client.post(self.ENDPOINT, headers=headers, json=payload)
                if response.status_code == 200:
                    speaker = fallback_speaker
                    active_model = fallback_model

            if response.status_code != 200:
                raise RuntimeError(
                    f"Sarvam TTS API error ({response.status_code}): {response.text}"
                )

            data = response.json()
            audios = data.get("audios", [])
            if not audios or not audios[0]:
                raise RuntimeError("Sarvam TTS API returned an empty audio list.")

            audio_bytes = base64.b64decode(audios[0])
            if len(audio_bytes) < 44:
                raise RuntimeError("Sarvam TTS returned corrupt audio bytes.")

            uid = uuid.uuid4().hex
            wav_out = self.temp_dir / f"tts_{uid}.wav"
            with open(wav_out, "wb") as f:
                f.write(audio_bytes)

            # Probe duration using soundfile
            try:
                info = sf.info(str(wav_out))
                duration_sec = info.duration
                sample_rate = info.samplerate
            except Exception:
                # Estimate for 16kHz 16-bit mono PCM (32,000 bytes/sec)
                duration_sec = max(0.1, (len(audio_bytes) - 44) / 32000.0)
                sample_rate = 16000

            latency_ms = (time.perf_counter() - t_start) * 1000.0

            return TTSResult(
                audio_bytes=audio_bytes,
                audio_format="wav",
                sample_rate=sample_rate,
                duration_seconds=round(duration_sec, 2),
                voice_used=preservation_plan.synthesis_voice_used or speaker,
                latency_ms=round(latency_ms, 2),
                model_name=f"sarvam_{active_model}_{speaker}",
                emotion_applied=preservation_plan.emotion,
                is_demo=False,
                details={
                    "provider": "sarvam",
                    "model": active_model,
                    "speaker": speaker,
                    "target_language_code": sarvam_lang,
                    "pace": pace,
                    "pitch_offset": pitch_offset,
                    "loudness": loudness,
                    "audio_filename": wav_out.name,
                    "download_ready": True,
                    "expressive_synthesis": True,
                }
            )
        except Exception as e:
            logger.error(f"[SarvamTTSProvider] Synthesis failed: {e}")
            raise

    def get_health(self) -> ModelHealthStatus:
        has_key = bool(self.api_key)
        return ModelHealthStatus(
            module="TTS (Sarvam bulbul:v4-flash)",
            model_name=f"sarvam/{self.model}",
            is_loaded=has_key,
            device="cloud_rest",
            status_message=(
                "Sarvam AI TTS API configured and ready."
                if has_key
                else "SARVAM_API_KEY is missing."
            ),
            details={
                "provider": "sarvam",
                "model": self.model,
                "endpoint": self.ENDPOINT,
                "has_api_key": has_key,
            }
        )
