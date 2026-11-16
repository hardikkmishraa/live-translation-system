"""
Abstract base class and contract for Text-to-Speech (TTS) providers.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Any, Optional
import numpy as np

from backend.app.models.base import TTSResult, EmotionPreservationPlan, ModelHealthStatus


class BaseTTSProvider(ABC):
    """Abstract interface that all speech synthesis engines must implement."""

    @abstractmethod
    def synthesize(
        self,
        text: str,
        target_lang: str,
        preservation_plan: EmotionPreservationPlan,
        speaker_gender: Optional[str] = None
    ) -> TTSResult:
        """
        Synthesizes text into audio with prosody and emotion conditioning.
        
        Args:
            text: Normalized text to synthesize.
            target_lang: Target language code ('hi', 'en', 'hi-IN', 'en-IN', 'en-US').
            preservation_plan: Prosody and emotional parameter modulation plan.
            speaker_gender: Detected speaker gender ('male' or 'female') for voice matching.
            
        Returns:
            TTSResult containing audio bytes (16kHz WAV format preferred), duration, and metadata.
        """
        pass

    @abstractmethod
    def get_health(self) -> ModelHealthStatus:
        """Returns health and readiness status of the TTS provider."""
        pass
