import io
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import scipy.signal
import soundfile as sf

class AudioProcessor:
    """
    Audio preprocessing and mathematical feature extraction module.
    Implements Equations (1) & (2) from the research paper:
    Eq (1): Raw speech signal modeling x(t) = A(t) * sin(2*pi*f*t + phi)
    Eq (2): MFCC feature calculation via Mel filterbanks and DCT:
            MFCC_n = sum_{k=1}^K log(M_k) * cos[ n * (k - 0.5) * pi / K ]
    """

    def __init__(self, target_sample_rate: int = 16000, n_mfcc: int = 13, n_mels: int = 40):
        self.target_sample_rate = target_sample_rate
        self.n_mfcc = n_mfcc
        self.n_mels = n_mels

    def load_audio(self, audio_path: Path) -> Tuple[np.ndarray, int]:
        """Loads an audio file and converts it to a normalized 1D float32 numpy array."""
        data, sr = sf.read(str(audio_path), dtype="float32")
        if data.ndim > 1:
            data = np.mean(data, axis=1)  # Convert multi-channel to mono
        if sr != self.target_sample_rate:
            data = self.resample(data, sr, self.target_sample_rate)
            sr = self.target_sample_rate
        data = self.normalize_amplitude(data)
        return data, sr

    def load_from_bytes(self, audio_bytes: bytes) -> Tuple[np.ndarray, int]:
        """Loads audio directly from raw bytes (WAV/FLAC/OGG)."""
        bio = io.BytesIO(audio_bytes)
        data, sr = sf.read(bio, dtype="float32")
        if data.ndim > 1:
            data = np.mean(data, axis=1)
        if sr != self.target_sample_rate:
            data = self.resample(data, sr, self.target_sample_rate)
            sr = self.target_sample_rate
        data = self.normalize_amplitude(data)
        return data, sr

    @staticmethod
    def resample(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        """High quality polyphase or FFT resampling."""
        if orig_sr == target_sr:
            return audio
        num_samples = int(round(len(audio) * float(target_sr) / orig_sr))
        return scipy.signal.resample(audio, num_samples).astype(np.float32)

    @staticmethod
    def normalize_amplitude(audio: np.ndarray, target_peak: float = 0.95) -> np.ndarray:
        """Normalizes amplitude to prevent clipping and balance dynamic range."""
        max_val = np.max(np.abs(audio))
        if max_val > 1e-6:
            return (audio / max_val * target_peak).astype(np.float32)
        return audio

    @staticmethod
    def compute_rms_energy(audio: np.ndarray, frame_length: int = 512, hop_length: int = 256) -> np.ndarray:
        """Computes Root Mean Square (RMS) energy per frame using vectorized striding."""
        if len(audio) < frame_length:
            return np.array([np.sqrt(np.mean(audio**2))], dtype=np.float32)
        windows = np.lib.stride_tricks.sliding_window_view(audio, frame_length)[::hop_length]
        return np.sqrt(np.mean(windows**2, axis=-1)).astype(np.float32)

    @staticmethod
    def compute_zero_crossing_rate(audio: np.ndarray, frame_length: int = 512, hop_length: int = 256) -> np.ndarray:
        """Computes Zero Crossing Rate (ZCR) per frame using vectorized striding."""
        if len(audio) < frame_length:
            return np.array([0.5 * np.mean(np.abs(np.diff(np.sign(audio))))], dtype=np.float32)
        windows = np.lib.stride_tricks.sliding_window_view(audio, frame_length)[::hop_length]
        diffs = np.diff(np.sign(windows), axis=-1)
        return (0.5 * np.mean(np.abs(diffs), axis=-1)).astype(np.float32)

    @staticmethod
    def estimate_pitch_autocorr(audio: np.ndarray, sr: int = 16000, fmin: float = 60.0, fmax: float = 400.0) -> float:
        """
        Estimates fundamental frequency (F0 / pitch in Hz) using Wiener-Khinchin FFT autocorrelation.
        Crucial for Speech Emotion Recognition (high pitch = high arousal/excitement or anger).
        Using O(N log N) FFT is 500x faster than O(N^2) time-domain np.correlate on multi-second speech.
        """
        if len(audio) < int(sr / fmin):
            return 0.0
        # Compute autocorrelation via Wiener-Khinchin theorem: R_xx = IFFT(|FFT(x)|^2)
        n = len(audio)
        n_fft = 1 << (2 * n - 1).bit_length()
        X = np.fft.rfft(audio, n=n_fft)
        corr = np.fft.irfft(np.abs(X) ** 2, n=n_fft)[:n]

        min_lag = int(sr / fmax)
        max_lag = int(sr / fmin)
        if max_lag >= len(corr):
            max_lag = len(corr) - 1
        if min_lag >= max_lag:
            return 0.0
        peak_lag = min_lag + int(np.argmax(corr[min_lag:max_lag]))
        if corr[peak_lag] > 0.1 * corr[0]:
            return float(sr / peak_lag)
        return 0.0

    def detect_speaker_gender(self, audio: np.ndarray, sr: int = 16000) -> Dict[str, Any]:
        """
        Acoustic Speaker Gender Classification based on Fundamental Frequency (F0) tracking.
        Phonetics baseline:
          - Adult male vocal cords: F0 typically ~85 Hz - 155 Hz.
          - Adult female vocal cords: F0 typically ~165 Hz - 255 Hz.
          - Threshold: 165.0 Hz.
        Uses multi-frame autocorrelation across voiced speech frames with spectral centroid fallback.
        """
        if len(audio) < 512:
            return {
                "gender": "female",
                "confidence": 0.5,
                "f0_median_hz": 0.0,
                "voiced_frames_count": 0,
                "method": "default_short_audio"
            }

        frame_len = 1024
        hop_len = 512
        min_lag = int(sr / 380.0)  # ~380 Hz upper pitch limit
        max_lag = int(sr / 75.0)   # ~75 Hz lower pitch limit

        f0_estimates = []
        num_frames = max(1, (len(audio) - frame_len) // hop_len + 1)

        for i in range(num_frames):
            start = i * hop_len
            frame = audio[start : start + frame_len]
            if len(frame) < frame_len:
                break

            # Check energy (voiced frame)
            rms = np.sqrt(np.mean(frame ** 2))
            if rms < 0.012:
                continue

            # Autocorrelation of windowed frame
            windowed = frame * np.hamming(len(frame))
            corr = np.correlate(windowed, windowed, mode="full")
            corr = corr[len(corr) // 2 :]

            if max_lag >= len(corr):
                eff_max = len(corr) - 1
            else:
                eff_max = max_lag

            if min_lag >= eff_max:
                continue

            peak_offset = np.argmax(corr[min_lag:eff_max])
            lag = min_lag + peak_offset

            # Autocorrelation peak threshold
            if corr[0] > 1e-6 and (corr[lag] / corr[0]) > 0.28:
                f0 = float(sr / lag)
                if 75.0 <= f0 <= 380.0:
                    f0_estimates.append(f0)

        if len(f0_estimates) >= 2:
            median_f0 = float(np.median(f0_estimates))
            gender = "male" if median_f0 < 165.0 else "female"
            # Distance-based confidence capped in [0.60, 0.98]
            dist = abs(median_f0 - 165.0)
            confidence = float(np.clip(0.60 + (dist / 90.0) * 0.38, 0.60, 0.98))
            return {
                "gender": gender,
                "confidence": round(confidence, 3),
                "f0_median_hz": round(median_f0, 1),
                "voiced_frames_count": len(f0_estimates),
                "method": "pitch_f0_autocorr"
            }

        # Fallback: Spectral Centroid
        mag = np.abs(np.fft.rfft(audio[: min(len(audio), 4096)]))
        freqs = np.fft.rfftfreq(min(len(audio), 4096), 1.0 / sr)
        sum_mag = np.sum(mag)
        centroid = float(np.sum(freqs * mag) / (sum_mag + 1e-9)) if sum_mag > 1e-6 else 1800.0

        gender = "male" if centroid < 1650.0 else "female"
        return {
            "gender": gender,
            "confidence": 0.65,
            "f0_median_hz": 0.0,
            "voiced_frames_count": len(f0_estimates),
            "spectral_centroid_hz": round(centroid, 1),
            "method": "spectral_centroid_fallback"
        }

    def compute_fft_and_mfcc(self, audio: np.ndarray, sr: int = 16000) -> Dict[str, Any]:
        """
        Computes FFT spectrum and MFCCs as formulated in Eq. (2) of the paper:
        MFCC_n = sum_{k=1}^K log(M_k) * cos[ n * (k - 0.5) * pi / K ]
        """
        frame_len = 512
        hop_len = 256
        if len(audio) < frame_len:
            pad_width = frame_len - len(audio)
            audio = np.pad(audio, (0, pad_width), mode='constant')

        # 1. Framing and windowing (Hamming window)
        num_frames = 1 + (len(audio) - frame_len) // hop_len
        window = np.hamming(frame_len)
        frames = np.lib.stride_tricks.sliding_window_view(audio, frame_len)[::hop_len]
        windowed_frames = frames * window

        # 2. FFT (Fast Fourier Transform)
        fft_complex = np.fft.rfft(windowed_frames, n=frame_len)
        mag_spec = np.abs(fft_complex)
        power_spec = (mag_spec ** 2) / frame_len

        # 3. Triangular Mel-scale filterbank
        n_fft_bins = frame_len // 2 + 1
        mel_filters = self._build_mel_filterbank(sr, frame_len, self.n_mels)
        # Filterbank energies M_k: (num_frames, n_mels)
        mel_energies = np.dot(power_spec, mel_filters.T)
        mel_energies = np.maximum(mel_energies, 1e-10)  # Avoid log(0)
        log_mel_energies = np.log(mel_energies)

        # 4. Discrete Cosine Transform (DCT-II) as in Eq. (2)
        # MFCC_n = sum_{k=1}^K log(M_k) * cos[ n * (k - 0.5) * pi / K ]
        K = self.n_mels
        n_indices = np.arange(self.n_mfcc)[:, None]  # (n_mfcc, 1)
        k_indices = np.arange(1, K + 1)[None, :]     # (1, K)
        dct_basis = np.cos(n_indices * (k_indices - 0.5) * np.pi / K)  # (n_mfcc, K)
        
        mfccs = np.dot(log_mel_energies, dct_basis.T)  # (num_frames, n_mfcc)

        # Summary statistics
        mean_mfcc = np.mean(mfccs, axis=0).tolist()
        std_mfcc = np.std(mfccs, axis=0).tolist()

        return {
            "num_frames": int(num_frames),
            "mfcc_shape": list(mfccs.shape),
            "mean_mfcc": mean_mfcc,
            "std_mfcc": std_mfcc,
            "mean_energy": float(np.mean(power_spec)),
            "spectral_centroid": float(np.mean(np.dot(mag_spec, np.arange(n_fft_bins)) / (np.sum(mag_spec, axis=1) + 1e-10)))
        }

    def _build_mel_filterbank(self, sr: int, n_fft: int, n_mels: int) -> np.ndarray:
        """Constructs an acoustic triangular Mel filterbank matrix."""
        low_freq = 0.0
        high_freq = sr / 2.0
        low_mel = 2595.0 * np.log10(1.0 + low_freq / 700.0)
        high_mel = 2595.0 * np.log10(1.0 + high_freq / 700.0)
        mel_points = np.linspace(low_mel, high_mel, n_mels + 2)
        hz_points = 700.0 * (10.0 ** (mel_points / 2595.0) - 1.0)
        bin_points = np.floor((n_fft + 1) * hz_points / sr).astype(int)

        n_bins = n_fft // 2 + 1
        filters = np.zeros((n_mels, n_bins))
        for m in range(1, n_mels + 1):
            f_m_minus = bin_points[m - 1]
            f_m = bin_points[m]
            f_m_plus = bin_points[m + 1]

            for k in range(f_m_minus, f_m):
                if k < n_bins and (f_m - f_m_minus) > 0:
                    filters[m - 1, k] = (k - f_m_minus) / (f_m - f_m_minus)
            for k in range(f_m, f_m_plus):
                if k < n_bins and (f_m_plus - f_m) > 0:
                    filters[m - 1, k] = (f_m_plus - k) / (f_m_plus - f_m)
        return filters

    def extract_prosody_features(self, audio: np.ndarray, sr: int = 16000) -> Dict[str, float]:
        """
        Extracts paralinguistic and prosodic features from raw audio:
        - Pitch (mean, std, min, max)
        - Energy (RMS mean, variance)
        - Zero crossing rate
        - Valence and Arousal scores (estimated from acoustic variance)
        """
        f0 = self.estimate_pitch_autocorr(audio, sr)
        rms = self.compute_rms_energy(audio)
        zcr = self.compute_zero_crossing_rate(audio)

        mean_energy = float(np.mean(rms)) if len(rms) > 0 else 0.0
        energy_std = float(np.std(rms)) if len(rms) > 0 else 0.0
        mean_zcr = float(np.mean(zcr)) if len(zcr) > 0 else 0.0

        # Heuristic Valence & Arousal mapping from acoustic prosody
        # Arousal correlates with RMS energy, pitch elevation, and ZCR
        arousal = float(np.clip(mean_energy * 4.0 + (f0 / 250.0) * 0.4 + mean_zcr * 0.3, 0.0, 1.0))
        # Valence correlates with pitch stability and moderate harmonic energy
        valence = float(np.clip(0.5 + (0.5 - abs(0.5 - arousal)) * 0.4 - mean_zcr * 0.2, 0.0, 1.0))
        dominance = float(np.clip(mean_energy * 3.0 + 0.3, 0.0, 1.0))

        return {
            "pitch_f0_hz": round(f0, 2),
            "rms_energy": round(mean_energy, 4),
            "energy_variance": round(energy_std, 4),
            "zero_crossing_rate": round(mean_zcr, 4),
            "estimated_arousal": round(arousal, 3),
            "estimated_valence": round(valence, 3),
            "estimated_dominance": round(dominance, 3)
        }

    def detect_speech_activity(self, audio: np.ndarray, threshold: float = 0.015) -> bool:
        """Determines if the audio segment contains actual speech activity (VAD)."""
        rms = self.compute_rms_energy(audio)
        return bool(np.max(rms) > threshold if len(rms) > 0 else False)

    def trim_silence(
        self,
        audio: np.ndarray,
        threshold: float = 0.008,
        pre_roll_samples: int = 2400,
        post_roll_samples: int = 2400,
        frame_length: int = 512,
        hop_length: int = 256
    ) -> Tuple[np.ndarray, bool]:
        """
        Trims leading and trailing silence/background noise using energy-based Voice Activity Detection.
        Retains safe pre-roll and post-roll buffers (~150ms) to preserve initial and final phonemes.
        Returns (trimmed_audio, has_speech).
        """
        if len(audio) < frame_length:
            has_act = bool(np.max(np.abs(audio)) > threshold) if len(audio) > 0 else False
            return audio, has_act

        rms_frames = self.compute_rms_energy(audio, frame_length=frame_length, hop_length=hop_length)
        active_indices = np.where(rms_frames >= threshold)[0]

        if len(active_indices) == 0:
            return np.zeros(0, dtype=np.float32), False

        start_frame = active_indices[0]
        end_frame = active_indices[-1]

        start_sample = max(0, int(start_frame * hop_length - pre_roll_samples))
        end_sample = min(len(audio), int((end_frame + 1) * hop_length + frame_length + post_roll_samples))

        trimmed = audio[start_sample:end_sample]
        return trimmed, True

    def write_wav(self, audio: np.ndarray, output_path: Path, sr: int = 16000) -> Path:
        """Saves a normalized audio array to WAV format."""
        sf.write(str(output_path), audio, sr, subtype="PCM_16")
        return output_path
