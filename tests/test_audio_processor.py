import numpy as np
import pytest
from backend.app.core.audio_processor import AudioProcessor

def test_audio_processor_initialization():
    proc = AudioProcessor(target_sample_rate=16000, n_mfcc=13, n_mels=40)
    assert proc.target_sample_rate == 16000
    assert proc.n_mfcc == 13
    assert proc.n_mels == 40

def test_normalization():
    proc = AudioProcessor()
    audio = np.array([0.1, -0.8, 0.4, 0.9], dtype=np.float32)
    normalized = proc.normalize_amplitude(audio, target_peak=0.95)
    assert np.isclose(np.max(np.abs(normalized)), 0.95, atol=1e-4)

def test_rms_and_zcr():
    proc = AudioProcessor()
    # 1 second of 440 Hz tone at 16kHz
    sr = 16000
    t = np.linspace(0, 1.0, sr, endpoint=False)
    tone = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)

    rms = proc.compute_rms_energy(tone)
    zcr = proc.compute_zero_crossing_rate(tone)
    assert len(rms) > 0
    assert len(zcr) > 0
    assert np.mean(rms) > 0.1
    assert np.mean(zcr) > 0.01

def test_mfcc_and_fft_eq2():
    proc = AudioProcessor(target_sample_rate=16000, n_mfcc=13, n_mels=40)
    sr = 16000
    t = np.linspace(0, 0.5, int(sr * 0.5), endpoint=False)
    tone = (0.6 * np.sin(2 * np.pi * 300 * t)).astype(np.float32)

    features = proc.compute_fft_and_mfcc(tone, sr=sr)
    assert "mean_mfcc" in features
    assert len(features["mean_mfcc"]) == 13
    assert features["num_frames"] > 0
    assert features["spectral_centroid"] > 0

def test_prosody_extraction():
    proc = AudioProcessor()
    sr = 16000
    t = np.linspace(0, 0.5, int(sr * 0.5), endpoint=False)
    tone = (0.4 * np.sin(2 * np.pi * 200 * t)).astype(np.float32)

    prosody = proc.extract_prosody_features(tone, sr=sr)
    assert 0.0 <= prosody["estimated_arousal"] <= 1.0
    assert 0.0 <= prosody["estimated_valence"] <= 1.0
    assert 0.0 <= prosody["estimated_dominance"] <= 1.0
    assert prosody["rms_energy"] > 0.0
