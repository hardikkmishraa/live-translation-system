"""
Streaming Session Management & Cognitive State Accumulator (Phase 9)
=====================================================================
Provides low-latency speech chunking, VAD boundary detection, incremental ASR
hypothesis stabilization, and cross-segment emotional state aggregation.

Core Components:
  1. EmotionAggregator:
     Exponential Moving Average (EMA) smoothing for acoustic emotion probabilities
     and dimensional Valence-Arousal-Dominance (VAD) vectors across consecutive
     speech segments:
         P_smooth(t) = alpha * P_curr(t) + (1 - alpha) * P_smooth(t - 1)
         V_smooth(t) = alpha * V_curr(t) + (1 - alpha) * V_smooth(t - 1)
     Prevents sudden unnatural emotional shifts between short continuous utterances.

  2. StreamingVADChunker:
     Adaptive real-time acoustic frame energy analysis, speech onset detection,
     silence window tracking, and safe pre/post-roll audio preservation.

  3. TranscriptStabilizer:
     Monitors incremental partial ASR hypotheses, stabilizes committed phrases,
     and prevents duplicate word emission.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

from backend.app.utils.logger import logger


@dataclass
class AggregatedEmotion:
    """Represents temporally-smoothed emotional cognition across segments."""
    dominant_emotion: str
    confidence: float
    probabilities: Dict[str, float]
    valence: float
    arousal: float
    dominance: float
    update_count: int


class EmotionAggregator:
    """
    Maintains and smooths paralinguistic and emotion predictions across adjacent
    dialogue segments using an Exponential Moving Average (EMA) filter.
    """

    def __init__(self, alpha: float = 0.65):
        """
        Args:
            alpha: Smoothing factor in [0.0, 1.0].
                   alpha=1.0 uses only current segment.
                   alpha=0.65 provides 65% weight on current segment,
                   35% weight on historical conversational momentum.
        """
        self.alpha = float(np.clip(alpha, 0.1, 1.0))
        self.smoothed_probabilities: Optional[Dict[str, float]] = None
        self.smoothed_valence: Optional[float] = None
        self.smoothed_arousal: Optional[float] = None
        self.smoothed_dominance: Optional[float] = None
        self.update_count: int = 0

    def update(
        self,
        current_probabilities: Optional[Dict[str, float]],
        valence: Optional[float] = None,
        arousal: Optional[float] = None,
        dominance: Optional[float] = None,
    ) -> AggregatedEmotion:
        """
        Integrates a newly recognized emotion distribution into the rolling state.
        """
        self.update_count += 1
        curr_p = current_probabilities or {"neutral": 1.0}
        v = valence if valence is not None else 0.5
        a = arousal if arousal is not None else 0.5
        d = dominance if dominance is not None else 0.5

        if self.smoothed_probabilities is None:
            # First segment initializes baseline directly
            self.smoothed_probabilities = {k: float(v) for k, v in curr_p.items()}
            self.smoothed_valence = float(v)
            self.smoothed_arousal = float(a)
            self.smoothed_dominance = float(d)
        else:
            # Union of keys across dictionaries
            all_keys = set(self.smoothed_probabilities.keys()) | set(curr_p.keys())
            updated_probs: Dict[str, float] = {}
            for k in all_keys:
                prior = self.smoothed_probabilities.get(k, 0.0)
                curr = curr_p.get(k, 0.0)
                updated_probs[k] = float(self.alpha * curr + (1.0 - self.alpha) * prior)

            # Re-normalize to sum to 1.0
            sum_prob = sum(updated_probs.values())
            if sum_prob > 1e-6:
                self.smoothed_probabilities = {k: p / sum_prob for k, p in updated_probs.items()}
            else:
                self.smoothed_probabilities = updated_probs

            # EMA on dimensional axes
            self.smoothed_valence = float(self.alpha * v + (1.0 - self.alpha) * (self.smoothed_valence or 0.5))
            self.smoothed_arousal = float(self.alpha * a + (1.0 - self.alpha) * (self.smoothed_arousal or 0.5))
            self.smoothed_dominance = float(self.alpha * d + (1.0 - self.alpha) * (self.smoothed_dominance or 0.5))

        # Determine dominant emotion from smoothed distribution
        top_emo = max(self.smoothed_probabilities.items(), key=lambda item: item[1])
        top_name = top_emo[0]
        top_conf = float(top_emo[1])

        return AggregatedEmotion(
            dominant_emotion=top_name,
            confidence=round(top_conf, 3),
            probabilities={k: round(v, 4) for k, v in self.smoothed_probabilities.items()},
            valence=round(self.smoothed_valence or 0.5, 3),
            arousal=round(self.smoothed_arousal or 0.5, 3),
            dominance=round(self.smoothed_dominance or 0.5, 3),
            update_count=self.update_count
        )

    def reset(self):
        """Clears accumulated emotional history."""
        self.smoothed_probabilities = None
        self.smoothed_valence = None
        self.smoothed_arousal = None
        self.smoothed_dominance = None
        self.update_count = 0


class StreamingVADChunker:
    """
    Manages streaming PCM audio frames, energy Voice Activity Detection (VAD),
    speech boundary triggering, and audio buffer roll management.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        silence_threshold_rms: float = 0.007,
        min_speech_samples: int = 4800,        # 300 ms minimum speech
        min_chunk_samples_for_ser: int = 12800, # 800 ms minimum for full SER model
        silence_chunks_trigger: int = 3,       # ~600-750 ms silence to trigger completion
        max_buffer_samples: int = 240000,      # 15s max buffer limit
        pre_roll_samples: int = 2400           # 150 ms pre-roll to preserve initial phonemes
    ):
        self.sample_rate = sample_rate
        self.silence_threshold_rms = silence_threshold_rms
        self.min_speech_samples = min_speech_samples
        self.min_chunk_samples_for_ser = min_chunk_samples_for_ser
        self.silence_chunks_trigger = silence_chunks_trigger
        self.max_buffer_samples = max_buffer_samples
        self.pre_roll_samples = pre_roll_samples

        self.audio_buffer = np.zeros(0, dtype=np.float32)
        self.in_speech: bool = False
        self.speech_chunks_count: int = 0
        self.silence_chunks_count: int = 0

    def feed_chunk(self, float_samples: np.ndarray) -> Tuple[bool, Optional[np.ndarray], Dict[str, Any]]:
        """
        Ingests a 16kHz float32 audio chunk.
        Returns:
            (is_utterance_boundary, completed_segment_audio_or_None, metadata)
        """
        if len(float_samples) == 0:
            return False, None, {"in_speech": self.in_speech, "frame_rms": 0.0}

        frame_rms = float(np.sqrt(np.mean(float_samples ** 2)))

        # VAD State Updates
        if frame_rms >= self.silence_threshold_rms:
            if not self.in_speech:
                self.in_speech = True
            self.silence_chunks_count = 0
            self.speech_chunks_count += 1
        else:
            if self.in_speech:
                self.silence_chunks_count += 1

        self.audio_buffer = np.concatenate([self.audio_buffer, float_samples])

        meta = {
            "in_speech": self.in_speech,
            "frame_rms": round(frame_rms, 5),
            "buffer_samples": len(self.audio_buffer)
        }

        # Safety: clamp max buffer size to avoid memory overflow
        if len(self.audio_buffer) >= self.max_buffer_samples:
            logger.warning(f"[VADChunker] Max buffer limit ({self.max_buffer_samples}) exceeded. Forcing boundary.")
            self.silence_chunks_count = self.silence_chunks_trigger

        # Check for completed utterance boundary
        if (
            self.in_speech
            and self.silence_chunks_count >= self.silence_chunks_trigger
            and len(self.audio_buffer) >= self.min_speech_samples
            and self.speech_chunks_count >= 2
        ):
            utterance_audio = self.audio_buffer.copy()
            # Reset chunker state
            self.audio_buffer = np.zeros(0, dtype=np.float32)
            self.in_speech = False
            self.speech_chunks_count = 0
            self.silence_chunks_count = 0
            meta["boundary_reason"] = "silence_trigger"
            return True, utterance_audio, meta

        # Clean prolonged idle silence to prevent baseline drift while keeping pre-roll buffer
        if not self.in_speech and len(self.audio_buffer) > self.pre_roll_samples:
            self.audio_buffer = self.audio_buffer[-self.pre_roll_samples :]

        return False, None, meta

    def force_flush(self) -> Optional[np.ndarray]:
        """Forces commitment of whatever speech has accumulated in the buffer."""
        if len(self.audio_buffer) >= 1600:  # At least 100ms
            utterance_audio = self.audio_buffer.copy()
            self.audio_buffer = np.zeros(0, dtype=np.float32)
            self.in_speech = False
            self.speech_chunks_count = 0
            self.silence_chunks_count = 0
            return utterance_audio
        return None

    def reset(self):
        """Resets the chunker buffer and state."""
        self.audio_buffer = np.zeros(0, dtype=np.float32)
        self.in_speech = False
        self.speech_chunks_count = 0
        self.silence_chunks_count = 0


class TranscriptStabilizer:
    """
    Stabilizes streaming ASR hypotheses, suppressing stutter and duplicate
    transcripts across adjacent chunks.
    """

    def __init__(self):
        self.last_emitted_partial: str = ""
        self.last_emitted_final: str = ""
        self.committed_history: List[str] = []

    def should_emit_partial(self, partial_text: str) -> bool:
        """Determines whether a partial transcript represents meaningful new text."""
        clean = (partial_text or "").strip()
        if not clean:
            return False
        if clean.lower() == self.last_emitted_partial.lower():
            return False
        # Do not emit partial if it merely repeats the committed final
        if clean.lower() == self.last_emitted_final.lower():
            return False
        self.last_emitted_partial = clean
        return True

    def filter_final(self, final_text: str) -> Optional[str]:
        """
        Validates and deduplicates a final transcript candidate before NMT and synthesis.
        Returns cleaned text or None if duplicate.
        """
        clean = (final_text or "").strip()
        if not clean:
            return None
        # Suppress identical consecutive finals
        if self.last_emitted_final and clean.lower() == self.last_emitted_final.lower():
            return None

        self.last_emitted_final = clean
        self.last_emitted_partial = ""
        self.committed_history.append(clean)
        return clean

    def reset(self):
        """Resets stabilization state."""
        self.last_emitted_partial = ""
        self.last_emitted_final = ""
        self.committed_history = []
