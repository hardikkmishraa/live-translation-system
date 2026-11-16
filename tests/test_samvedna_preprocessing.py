"""
Unit and Integration Tests for SAMVEDNA Hindi Dataset Preprocessing & MP4 Extraction.
=====================================================================================
Tests:
- MP4 with audio stream extraction.
- MP4 without audio stream rejection (video-only).
- Corrupted MP4 container handling.
- Hindi filename parsing (1-letter 'H' and 2-letter 'HI', 'H' and 'HA').
- Emotion extraction across all 6 classes (angry, happy, sad, neutral, disgust, fear).
- Audio conversion to 16 kHz mono WAV with peak normalization.
- Speaker-independent split partitioning with zero actor leakage.
- Pipeline execution and extraction report verification.
"""

import os
import json
import tempfile
import subprocess
import pytest
import numpy as np
import soundfile as sf
from pathlib import Path

from scripts.preprocess_samvedna import (
    parse_filename,
    inspect_audio_stream,
    extract_and_normalize_audio,
    assign_split,
    run_preprocessing,
    EMOTION_CODE_MAP,
    LANGUAGE_CODE_MAP,
    LABEL_TO_ID,
    DEFAULT_SPLIT_CONFIG,
    FFMPEG_PATH
)


def test_parse_valid_filenames_including_variants():
    """Verify that both standard and 2-letter SAMVEDNA variants parse accurately."""
    cases = [
        # Standard 1-letter
        ("A1_H_A_S1.mp3", "A1", "H", "Hindi", "angry", "S1", "mp3"),
        ("A25_H_H_S18.wav", "A25", "H", "Hindi", "happy", "S18", "wav"),
        ("A12_E_S_S4.mp4", "A12", "E", "English", "sad", "S4", "mp4"),
        ("A5_P_N_S7.m4a", "A5", "P", "Punjabi", "neutral", "S7", "m4a"),
        ("A3_H_D_S12.mp3", "A3", "H", "Hindi", "disgust", "S12", "mp3"),
        ("A9_H_F_S9.mp3", "A9", "H", "Hindi", "fear", "S9", "mp3"),
        # 2-letter variants from actual dataset downloads
        ("A6_HI_A_S1.mp4", "A6", "H", "Hindi", "angry", "S1", "mp4"),
        ("A6_HI_HA_S8.mp4", "A6", "H", "Hindi", "happy", "S8", "mp4"),
        ("A6_HI_D_S5.mp4", "A6", "H", "Hindi", "disgust", "S5", "mp4"),
        ("A6_HI_N_S5.mp4", "A6", "H", "Hindi", "neutral", "S5", "mp4"),
        ("A7_HI_F_S1.mp4", "A7", "H", "Hindi", "fear", "S1", "mp4"),
        ("A10_HI_S_S1.mp4", "A10", "H", "Hindi", "sad", "S1", "mp4"),
        ("A8_EN_A_S5.mp4", "A8", "E", "English", "angry", "S5", "mp4"),
        ("A7_PB_A_S8.mp4", "A7", "P", "Punjabi", "angry", "S8", "mp4"),
    ]

    for fname, exp_actor, exp_lcode, exp_lang, exp_emo, exp_sent, exp_fmt in cases:
        p = Path(fname)
        parsed = parse_filename(p)
        assert parsed is not None, f"Failed to parse {fname}"
        assert parsed["actor"] == exp_actor
        assert parsed["language_code"] == exp_lcode
        assert parsed["language"] == exp_lang
        assert parsed["emotion"] == exp_emo
        assert parsed["sentence"] == exp_sent
        assert parsed["format"] == exp_fmt


def test_parse_invalid_filenames():
    """Verify that malformed or non-compliant filenames are rejected."""
    invalid_cases = [
        "random_audio.wav",
        "A1_H_X_S1.mp3",       # Unknown emotion 'X'
        "A999_Z_A_S1.mp3",     # Invalid language 'Z'
        "A1_H_A.mp3",          # Missing sentence token
        "not_an_audio.txt",
    ]

    for fname in invalid_cases:
        p = Path(fname)
        assert parse_filename(p) is None


def test_speaker_independent_splits():
    """Ensure zero actor leakage across train, val, and test partitions."""
    train_actors = set(DEFAULT_SPLIT_CONFIG["train_actors"])
    val_actors = set(DEFAULT_SPLIT_CONFIG["val_actors"])
    test_actors = set(DEFAULT_SPLIT_CONFIG["test_actors"])

    assert len(train_actors & val_actors) == 0, "Speaker leakage between train and val!"
    assert len(train_actors & test_actors) == 0, "Speaker leakage between train and test!"
    assert len(val_actors & test_actors) == 0, "Speaker leakage between val and test!"

    for actor in DEFAULT_SPLIT_CONFIG["train_actors"]:
        assert assign_split(actor, DEFAULT_SPLIT_CONFIG) == "train"
    for actor in DEFAULT_SPLIT_CONFIG["val_actors"]:
        assert assign_split(actor, DEFAULT_SPLIT_CONFIG) == "val"
    for actor in DEFAULT_SPLIT_CONFIG["test_actors"]:
        assert assign_split(actor, DEFAULT_SPLIT_CONFIG) == "test"


def test_mp4_with_audio_extraction():
    """Generate a synthetic MP4 with audio and verify extraction to 16 kHz mono WAV."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        mp4_in = tmp_path / "valid_av.mp4"
        wav_out = tmp_path / "extracted.wav"

        # Generate 1-second synthetic MP4 with test video + 1000Hz sine tone
        cmd = [
            str(FFMPEG_PATH), "-y", "-v", "error",
            "-f", "lavfi", "-i", "testsrc=duration=1:size=160x120:rate=10",
            "-f", "lavfi", "-i", "sine=frequency=1000:duration=1:sample_rate=44100",
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-c:a", "aac",
            str(mp4_in)
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        assert res.returncode == 0, f"FFmpeg synthetic generation failed: {res.stderr}"

        # Test audio stream detection
        has_audio, info = inspect_audio_stream(mp4_in)
        assert has_audio is True
        assert "Audio:" in info

        # Test audio extraction & normalization
        success, duration, err, metrics = extract_and_normalize_audio(mp4_in, wav_out, target_sr=16000)
        assert success is True, f"Extraction failed: {err}"
        assert wav_out.exists()
        assert round(duration, 1) == 1.0

        # Verify output WAV properties
        data, sr = sf.read(str(wav_out))
        assert sr == 16000
        assert data.ndim == 1  # Mono
        assert np.isclose(np.max(np.abs(data)), 0.89125, atol=0.01)
        assert metrics["rms"] > 0.05


def test_mp4_without_audio_rejection():
    """Verify that video-only MP4 files lacking an audio stream are detected and rejected."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        video_only = tmp_path / "video_only.mp4"
        wav_out = tmp_path / "should_not_exist.wav"

        # Generate 1-second video-only MP4 (no audio stream)
        cmd = [
            str(FFMPEG_PATH), "-y", "-v", "error",
            "-f", "lavfi", "-i", "testsrc=duration=1:size=160x120:rate=10",
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-an",  # Explicitly disable audio
            str(video_only)
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        assert res.returncode == 0, f"FFmpeg video-only generation failed: {res.stderr}"

        # Verify detection
        has_audio, info = inspect_audio_stream(video_only)
        assert has_audio is False

        # Verify extraction rejection
        success, duration, err, _ = extract_and_normalize_audio(video_only, wav_out, target_sr=16000)
        assert success is False
        assert "no audio" in err.lower()
        assert not wav_out.exists()


def test_corrupted_mp4_rejection():
    """Verify that damaged or corrupted MP4 files are cleanly rejected without crashing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        corrupted = tmp_path / "corrupted.mp4"
        wav_out = tmp_path / "out.wav"

        # Write corrupted header garbage
        corrupted.write_bytes(b"\x00\x00\x00\x18ftypisom" + os.urandom(256))

        success, duration, err, _ = extract_and_normalize_audio(corrupted, wav_out, target_sr=16000)
        assert success is False
        assert not wav_out.exists()


def test_silent_audio_rejection():
    """Verify that silent audio is caught and rejected."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        silent_wav = tmp_path / "silent.wav"
        wav_out = tmp_path / "out.wav"

        # Write 1-second silent WAV
        sf.write(str(silent_wav), np.zeros(16000, dtype=np.float32), 16000)

        success, duration, err, _ = extract_and_normalize_audio(silent_wav, wav_out, target_sr=16000)
        assert success is False
        assert "silent" in err.lower()


def test_actual_extracted_hindi_dataset():
    """Verify the real extracted Hindi audio assets in data/samvedna_hindi/."""
    out_dir = Path("data/samvedna_hindi")
    manifest_file = out_dir / "manifest.csv"
    summary_file = out_dir / "dataset_summary.json"
    report_file = out_dir / "extraction_report.json"

    assert manifest_file.exists(), "manifest.csv missing"
    assert summary_file.exists(), "dataset_summary.json missing"
    assert report_file.exists(), "extraction_report.json missing"

    report = json.loads(report_file.read_text())
    assert report["total_files_inspected"] >= 18
    assert report["successful_extractions"] == 6
    assert report["final_hindi_samples_per_emotion"] == {
        "angry": 1,
        "happy": 1,
        "sad": 1,
        "neutral": 1,
        "disgust": 1,
        "fear": 1
    }

    # Verify each extracted audio file physically exists and is playable
    lines = manifest_file.read_text().strip().splitlines()
    assert len(lines) == 7  # Header + 6 samples
    for line in lines[1:]:
        parts = line.split(",")
        filepath = out_dir / parts[1]
        assert filepath.exists(), f"Extracted audio file missing: {filepath}"
        data, sr = sf.read(str(filepath))
        assert sr == 16000
        assert data.ndim == 1
        assert len(data) / sr > 1.0  # Duration > 1.0s
        assert np.max(np.abs(data)) > 0.1  # Audible
