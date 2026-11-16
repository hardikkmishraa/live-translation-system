"""
Unit and Integration Tests for Vishal B. Hindi SER Dataset Preprocessing.
========================================================================
Tests:
- 8-class emotion resolution from folder structures and filenames.
- Speaker extraction (M1-M4, F1-F4, speaker_X).
- Audio conversion and standardization to 16 kHz mono WAV.
- Silent audio detection and rejection.
- Strict speaker-independent split isolation (zero speaker leakage).
- End-to-end processing pipeline on a synthetic 8-class mock corpus.
"""

import os
import json
import tempfile
import pytest
import numpy as np
import soundfile as sf
from pathlib import Path

from scripts.preprocess_vishlb import (
    resolve_emotion_from_path,
    resolve_speaker_from_path,
    convert_and_normalize_audio,
    assign_split,
    run_preprocessing,
    LABEL_TO_ID_8,
    ID_TO_LABEL_8,
    DEFAULT_SPEAKER_SPLITS
)


def test_8_class_label_mapping_integrity():
    """Verify that all 8 Hindi emotion classes are mapped properly."""
    assert len(LABEL_TO_ID_8) == 8
    expected = {"angry", "disgust", "fear", "happy", "neutral", "sad", "sarcastic", "surprise"}
    assert set(LABEL_TO_ID_8.keys()) == expected
    for lbl, idx in LABEL_TO_ID_8.items():
        assert ID_TO_LABEL_8[str(idx)] == lbl


def test_resolve_emotion_from_subdirectories_and_tokens():
    """Test emotion resolution across directory hierarchies and filenames."""
    test_cases = [
        (Path("data/raw/Angry/M1_S1.wav"), "angry"),
        (Path("data/raw/anger/1001.wav"), "angry"),
        (Path("data/raw/Disgust/F1_S2.wav"), "disgust"),
        (Path("data/raw/Fear/M2_S3.wav"), "fear"),
        (Path("data/raw/Happy/F2_S4.wav"), "happy"),
        (Path("data/raw/Neutral/M3_S5.wav"), "neutral"),
        (Path("data/raw/Sad/F3_S6.wav"), "sad"),
        (Path("data/raw/Sarcastic/M4_S7.wav"), "sarcastic"),
        (Path("data/raw/Surprise/F4_S8.wav"), "surprise"),
        # Filename token cases (flat folder)
        (Path("data/raw/audio_ang_01.wav"), "angry"),
        (Path("data/raw/speaker_happy_take2.wav"), "happy"),
        (Path("data/raw/f1_sar_05.wav"), "sarcastic"),
        (Path("data/raw/m1_sur_09.wav"), "surprise"),
    ]

    for p, expected_emo in test_cases:
        emo = resolve_emotion_from_path(p)
        assert emo == expected_emo, f"Failed for {p}: got {emo}, expected {expected_emo}"


def test_resolve_speaker_patterns():
    """Test extracting speaker identifiers (M1-M4, F1-F4, speaker_X)."""
    assert resolve_speaker_from_path(Path("Angry/M1_S1.wav")) == "M1"
    assert resolve_speaker_from_path(Path("Happy/F3_S2.wav")) == "F3"
    assert resolve_speaker_from_path(Path("Sad/actor_2_sad.wav")) == "speaker_2"
    assert resolve_speaker_from_path(Path("Neutral/SPEAKER-4_01.wav")) == "speaker_4"


def test_speaker_independent_splits_zero_leakage():
    """Ensure train, val, and test partitions have zero speaker overlap."""
    train_spks = set(DEFAULT_SPEAKER_SPLITS["train_speakers"])
    val_spks = set(DEFAULT_SPEAKER_SPLITS["val_speakers"])
    test_spks = set(DEFAULT_SPEAKER_SPLITS["test_speakers"])

    assert len(train_spks & val_spks) == 0, "Speaker leakage: train & val!"
    assert len(train_spks & test_spks) == 0, "Speaker leakage: train & test!"
    assert len(val_spks & test_spks) == 0, "Speaker leakage: val & test!"

    for spk in DEFAULT_SPEAKER_SPLITS["train_speakers"]:
        assert assign_split(spk) == "train"
    for spk in DEFAULT_SPEAKER_SPLITS["val_speakers"]:
        assert assign_split(spk) == "val"
    for spk in DEFAULT_SPEAKER_SPLITS["test_speakers"]:
        assert assign_split(spk) == "test"


def test_audio_conversion_and_silence_rejection():
    """Test peak normalization and silent audio rejection on synthetic files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        valid_in = tmp_path / "valid.wav"
        silent_in = tmp_path / "silent.wav"
        out_wav = tmp_path / "out.wav"

        # Valid 1-sec tone
        t = np.linspace(0, 1.0, 16000, endpoint=False)
        tone = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
        sf.write(str(valid_in), tone, 16000)

        # Silent 1-sec file
        sf.write(str(silent_in), np.zeros(16000, dtype=np.float32), 16000)

        # Test valid
        success, dur, err, metrics = convert_and_normalize_audio(valid_in, out_wav)
        assert success is True
        assert out_wav.exists()
        assert np.isclose(metrics["peak"], 0.89125, atol=0.01)

        # Test silent
        success, dur, err, _ = convert_and_normalize_audio(silent_in, out_wav)
        assert success is False
        assert "silent" in err.lower()


def test_full_preprocessing_mock_8_class_corpus():
    """Verify preprocessing on an end-to-end mock corpus across all 8 classes."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        raw_dir = tmp_path / "raw"
        out_dir = tmp_path / "processed"
        raw_dir.mkdir()

        # Create folders for all 8 emotions
        emotions = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Sarcastic", "Surprise"]
        speakers = ["M1", "F3", "M4"]  # M1 -> train, F3 -> val, M4 -> test

        t = np.linspace(0, 1.0, 16000, endpoint=False)
        tone = (0.4 * np.sin(2 * np.pi * 350 * t)).astype(np.float32)

        for emo in emotions:
            emo_folder = raw_dir / emo
            emo_folder.mkdir()
            for spk in speakers:
                fname = f"{spk}_{emo}_take1.wav"
                sf.write(str(emo_folder / fname), tone, 16000)

        summary = run_preprocessing(raw_dir, out_dir)

        assert summary["total_files_discovered"] == 24  # 8 emotions * 3 speakers
        assert summary["successfully_processed"] == 24
        assert summary["rejected_or_corrupted"] == 0

        # Verify all 8 emotions have exactly 3 samples
        for emo in LABEL_TO_ID_8.keys():
            assert summary["emotion_distribution"][emo] == 3

        # Verify split isolation (8 per speaker)
        assert summary["splits_distribution"]["train"] == 8  # M1
        assert summary["splits_distribution"]["val"] == 8    # F3
        assert summary["splits_distribution"]["test"] == 8   # M4

        # Verify manifest.csv
        manifest = out_dir / "manifest.csv"
        assert manifest.exists()
        lines = manifest.read_text().strip().splitlines()
        assert len(lines) == 25  # Header + 24 rows
