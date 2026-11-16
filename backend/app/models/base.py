from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any
import numpy as np

@dataclass
class ModelHealthStatus:
    module: str
    model_name: str
    is_loaded: bool
    device: str
    status_message: str
    details: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ASRResult:
    text: str
    language: str
    confidence: Optional[float]
    latency_ms: float
    model_name: str
    is_demo: bool = False
    details: Dict[str, Any] = field(default_factory=dict)

@dataclass
class SERResult:
    top_emotion: str
    probabilities: Dict[str, float]
    valence: float
    arousal: float
    dominance: float
    latency_ms: float
    model_name: str
    is_demo: bool = False
    details: Dict[str, Any] = field(default_factory=dict)

@dataclass
class NMTResult:
    source_text: str
    translated_text: str
    source_lang: str
    target_lang: str
    latency_ms: float
    model_name: str
    is_demo: bool = False
    details: Dict[str, Any] = field(default_factory=dict)

@dataclass
class EmotionPreservationPlan:
    emotion: str
    pitch_multiplier: float
    rate_multiplier: float
    volume_multiplier: float
    gating_weight_z: float  # Latent-depth gating weight Eq. (8)
    applied_to_tts: bool
    prosody_notes: str
    expressive_synthesis_supported: bool = False
    emotional_fidelity_status: str = "Uncontrolled"
    speaker_similarity_retained: bool = False
    synthesis_voice_used: str = "none"
    detected_gender: Optional[str] = None
    gender_confidence: Optional[float] = None
    user_consent_granted: bool = False
    details: Dict[str, Any] = field(default_factory=dict)

@dataclass
class TTSResult:
    audio_bytes: bytes
    audio_format: str
    sample_rate: int
    duration_seconds: float
    voice_used: str
    latency_ms: float
    model_name: str
    emotion_applied: str
    is_demo: bool = False
    details: Dict[str, Any] = field(default_factory=dict)

class BaseASRModel(ABC):
    """Abstract interface for Automatic Speech Recognition."""
    @abstractmethod
    def transcribe(self, audio: np.ndarray, sample_rate: int, language: Optional[str] = None) -> ASRResult:
        pass

    @abstractmethod
    def get_health(self) -> ModelHealthStatus:
        pass

class BaseSERModel(ABC):
    """Abstract interface for Speech Emotion Recognition."""
    @abstractmethod
    def predict_emotion(self, audio: np.ndarray, sample_rate: int, acoustic_features: Optional[Dict[str, Any]] = None) -> SERResult:
        pass

    @abstractmethod
    def get_health(self) -> ModelHealthStatus:
        pass

class BaseNMTModel(ABC):
    """Abstract interface for Hindi <-> English Neural Machine Translation."""
    @abstractmethod
    def translate(self, text: str, source_lang: str, target_lang: str) -> NMTResult:
        pass

    @abstractmethod
    def get_health(self) -> ModelHealthStatus:
        pass

class BaseEmotionPreserver(ABC):
    """Abstract interface for Latent-Depth Gating & Emotion Preservation (Eq. 8)."""
    @abstractmethod
    def compute_preservation_plan(self, ser_result: SERResult, target_lang: str) -> EmotionPreservationPlan:
        pass

class BaseTTSModel(ABC):
    """Abstract interface for expressive, emotion-modulated Text-to-Speech synthesis."""
    @abstractmethod
    def synthesize(self, text: str, target_lang: str, preservation_plan: EmotionPreservationPlan) -> TTSResult:
        pass

    @abstractmethod
    def get_health(self) -> ModelHealthStatus:
        pass
