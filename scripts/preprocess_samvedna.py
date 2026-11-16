#!/usr/bin/env python3
"""
SAMVEDNA Hindi Speech Emotion Recognition Preprocessing & Audio Extraction Pipeline.
====================================================================================
Upgraded to support direct audio extraction from SAMVEDNA MP4 video recordings:
- Recursively inspects MP4, MP3, and WAV recordings.
- Detects whether an MP4 contains a valid audio stream using FFmpeg.
- Extracts only the audio stream without video frames or facial cues.
- Resamples and standardizes to mono 16 kHz, 16-bit PCM WAV.
- Applies peak audio normalization (-1.0 dBFS) and rejects silent or corrupted tracks.
- Filters strictly for Hindi speech (language codes 'H' or 'HI').
- Parses SAMVEDNA tokens: [Actor]_[Language]_[Emotion]_[Sentence].[ext].
  Supports both 1-letter ('H', 'E', 'P', 'HA') and 2-letter ('HI', 'EN', 'PB') variants.
- Enforces zero speaker leakage across speaker-independent splits.
- Detects duplicate recordings.
- Generates manifest.csv, label_mapping.json, dataset_summary.json, and extraction_report.json.
"""

import os
import re
import sys
import json
import shutil
import hashlib
import argparse
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Set

import numpy as np
import soundfile as sf

try:
    import imageio_ffmpeg
    FFMPEG_PATH = imageio_ffmpeg.get_ffmpeg_exe()
except Exception:
    FFMPEG_PATH = shutil.which("ffmpeg") or "ffmpeg"

# Project base directory
BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_INPUT_DIR = BASE_DIR / "data" / "raw_samvedna"
DEFAULT_OUTPUT_DIR = BASE_DIR / "data" / "samvedna_hindi"

# 6 SAMVEDNA Emotion Mappings
# Supports both 1-letter codes ('A', 'H', 'S', 'N', 'D', 'F') and 2-letter ('HA')
EMOTION_CODE_MAP = {
    "A": "angry",
    "H": "happy",
    "HA": "happy",
    "S": "sad",
    "N": "neutral",
    "D": "disgust",
    "F": "fear"
}

FOLDER_EMOTION_MAP = {
    "anger": "angry",
    "angry": "angry",
    "happy": "happy",
    "sad": "sad",
    "neutral": "neutral",
    "disgust": "disgust",
    "fear": "fear"
}

LANGUAGE_CODE_MAP = {
    "H": "Hindi",
    "HI": "Hindi",
    "E": "English",
    "EN": "English",
    "P": "Punjabi",
    "PB": "Punjabi"
}

LABEL_TO_ID = {
    "angry": 0,
    "happy": 1,
    "sad": 2,
    "neutral": 3,
    "disgust": 4,
    "fear": 5
}

ID_TO_LABEL = {str(v): k for k, v in LABEL_TO_ID.items()}

# Strict Speaker-Independent Split Strategy (25 Actors)
# Train: A1-A17 (17 actors: 9 Male, 8 Female, ~68%)
# Validation: A18-A21 (4 actors: 2 Male, 2 Female, ~16%)
# Test: A22-A25 (4 actors: 2 Male, 2 Female, ~16%)
DEFAULT_SPLIT_CONFIG = {
    "train_actors": [f"A{i}" for i in range(1, 18)],
    "val_actors": [f"A{i}" for i in range(18, 22)],
    "test_actors": [f"A{i}" for i in range(22, 26)]
}

# Regex supporting SAMVEDNA naming variants:
# [Actor]_[Language]_[Emotion]_[Sentence].[ext]
# e.g., A6_HI_A_S1.mp4, A6_HI_HA_S8.mp4, A1_H_S_S1.mp3
FILENAME_PATTERN = re.compile(
    r"^(A\d+)_(H|HI|E|EN|P|PB)_([A-Z]{1,2})_(S\d+)\.(mp3|wav|mp4|m4a|ogg|flac)$",
    re.IGNORECASE
)


def parse_filename(filepath: Path) -> Optional[Dict[str, str]]:
    """
    Parses actor, language, emotion, sentence, and extension from filename.
    Cross-checks with parent directory emotion name if applicable.
    """
    match = FILENAME_PATTERN.match(filepath.name)
    if not match:
        return None

    actor, lang_raw, emo_raw, sent, ext = match.groups()
    actor = actor.upper()
    lang_raw = lang_raw.upper()
    emo_raw = emo_raw.upper()
    sent = sent.upper()
    ext = ext.lower()

    if lang_raw not in LANGUAGE_CODE_MAP:
        return None

    language = LANGUAGE_CODE_MAP[lang_raw]
    norm_lang_code = "H" if language == "Hindi" else ("E" if language == "English" else "P")

    # Resolve emotion
    emotion = EMOTION_CODE_MAP.get(emo_raw)
    if not emotion:
        parent_name = filepath.parent.name.lower()
        if parent_name in FOLDER_EMOTION_MAP:
            emotion = FOLDER_EMOTION_MAP[parent_name]

    if not emotion or emotion not in LABEL_TO_ID:
        return None

    norm_emo_code = "A" if emotion == "angry" else (
        "H" if emotion == "happy" else (
            "S" if emotion == "sad" else (
                "N" if emotion == "neutral" else (
                    "D" if emotion == "disgust" else "F"
                )
            )
        )
    )

    return {
        "filename": filepath.name,
        "actor": actor,
        "language_raw": lang_raw,
        "language_code": norm_lang_code,
        "language": language,
        "emotion_raw": emo_raw,
        "emotion_code": norm_emo_code,
        "emotion": emotion,
        "sentence": sent,
        "format": ext,
        "filepath": str(filepath.resolve())
    }


def inspect_audio_stream(filepath: Path) -> Tuple[bool, str]:
    """
    Inspects media file to determine if a valid audio stream exists.
    Returns (has_audio, stream_info_or_error).
    """
    cmd = [str(FFMPEG_PATH), "-i", str(filepath)]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
        stderr = res.stderr or ""
        
        # Check for Audio stream description in FFmpeg output
        for line in stderr.splitlines():
            if "Audio:" in line:
                return True, line.strip()
                
        return False, "No audio stream detected in container"
    except Exception as e:
        return False, f"FFmpeg inspection failed: {e}"


def compute_file_hash(filepath: Path) -> str:
    """Computes SHA-256 hash of a file for exact duplicate detection."""
    sha = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            sha.update(chunk)
    return sha.hexdigest()


def extract_and_normalize_audio(
    input_path: Path,
    output_path: Path,
    target_sr: int = 16000
) -> Tuple[bool, float, str, Dict[str, Any]]:
    """
    Extracts audio from MP4/MP3/WAV, resamples to 16 kHz mono WAV,
    applies peak normalization (-1.0 dBFS), and verifies non-silence.
    Returns (success, duration_seconds, error_message, audio_metrics).
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_wav = output_path.with_suffix(".tmp.wav")
    metrics = {"sample_rate": target_sr, "channels": 1, "duration": 0.0, "peak": 0.0, "rms": 0.0}

    # Step 1: Verify audio stream presence
    has_audio, stream_info = inspect_audio_stream(input_path)
    if not has_audio:
        return False, 0.0, f"Rejected: {stream_info}", metrics

    # Step 2: Extract audio stream strictly without video frames
    ffmpeg_cmd = [
        str(FFMPEG_PATH), "-y", "-v", "error",
        "-i", str(input_path),
        "-vn",                    # Discard video track completely
        "-acodec", "pcm_s16le",   # 16-bit linear PCM
        "-ac", "1",               # Mono
        "-ar", str(target_sr),    # 16 kHz sample rate
        str(temp_wav)
    ]

    try:
        res = subprocess.run(ffmpeg_cmd, capture_output=True, text=True, timeout=30)
        if res.returncode != 0:
            if temp_wav.exists():
                temp_wav.unlink(missing_ok=True)
            return False, 0.0, f"FFmpeg extraction failed: {res.stderr.strip()}", metrics

        if not temp_wav.exists() or temp_wav.stat().st_size == 0:
            return False, 0.0, "Extracted audio file is empty (0 bytes)", metrics

        # Step 3: Validate audio signals with soundfile
        data, sr = sf.read(str(temp_wav), dtype="float32")
        duration = len(data) / float(sr)
        metrics["duration"] = round(duration, 3)

        if duration < 0.2:
            temp_wav.unlink(missing_ok=True)
            return False, duration, f"Audio too short ({duration:.2f}s < 0.2s threshold)", metrics

        # Step 4: Silent audio verification
        rms = float(np.sqrt(np.mean(data ** 2)))
        metrics["rms"] = round(rms, 6)
        if rms < 1e-4:
            temp_wav.unlink(missing_ok=True)
            return False, duration, f"Rejected: Audio is silent (RMS = {rms:.6f} < 0.0001)", metrics

        # Step 5: Peak normalization to -1.0 dBFS (~0.89125)
        max_val = float(np.max(np.abs(data)))
        metrics["peak"] = round(max_val, 4)
        if max_val > 0:
            norm_factor = 0.89125 / max_val
            norm_data = np.clip(data * norm_factor, -1.0, 1.0)
        else:
            norm_data = data

        # Step 6: Write finalized standardized WAV
        sf.write(str(output_path), norm_data, target_sr, subtype="PCM_16")
        temp_wav.unlink(missing_ok=True)
        return True, duration, "", metrics

    except Exception as e:
        if temp_wav.exists():
            temp_wav.unlink(missing_ok=True)
        return False, 0.0, f"Extraction exception: {str(e)}", metrics


def assign_split(actor: str, split_config: Dict[str, List[str]]) -> str:
    """Assigns split strictly by actor ID to guarantee zero speaker leakage."""
    if actor in split_config["train_actors"]:
        return "train"
    elif actor in split_config["val_actors"]:
        return "val"
    elif actor in split_config["test_actors"]:
        return "test"
    # Fallback to train if unknown actor
    return "train"


def run_preprocessing(
    input_dir: Path,
    output_dir: Path,
    hindi_only: bool = True,
    split_config: Optional[Dict[str, List[str]]] = None
) -> Dict[str, Any]:
    """
    Executes the full inspection, extraction, and manifest generation pipeline.
    Produces manifest.csv, label_mapping.json, dataset_summary.json, and extraction_report.json.
    """
    split_config = split_config or DEFAULT_SPLIT_CONFIG
    output_dir.mkdir(parents=True, exist_ok=True)
    audio_out_dir = output_dir / "audio"
    audio_out_dir.mkdir(parents=True, exist_ok=True)

    extraction_report = {
        "dataset_name": "SAMVEDNA-Hindi-Audio",
        "input_directory": str(input_dir.resolve()),
        "output_directory": str(output_dir.resolve()),
        "hindi_only_filter": hindi_only,
        "total_files_inspected": 0,
        "format_breakdown": {"mp4": 0, "mp3": 0, "wav": 0, "other": 0},
        "language_breakdown": {"Hindi": 0, "English": 0, "Punjabi": 0, "Unknown": 0},
        "files_containing_audio": 0,
        "video_only_or_silent_files": 0,
        "successful_extractions": 0,
        "rejected_or_corrupted": 0,
        "duplicate_files_detected": 0,
        "final_hindi_samples_per_emotion": {emo: 0 for emo in LABEL_TO_ID.keys()},
        "splits_distribution": {"train": 0, "val": 0, "test": 0},
        "actor_distribution": {},
        "duplicates": [],
        "errors": []
    }

    if not input_dir.exists():
        extraction_report["errors"].append(f"Input directory does not exist: {input_dir}")
        return extraction_report

    # Discover all candidate media files
    all_files = sorted(list(input_dir.rglob("*")))
    media_files = [
        f for f in all_files
        if f.is_file() and f.suffix.lower() in [".mp4", ".mp3", ".wav", ".m4a", ".ogg", ".flac"]
    ]
    extraction_report["total_files_inspected"] = len(media_files)

    manifest_rows = []
    seen_utterance_keys: Set[str] = set()
    seen_file_hashes: Dict[str, str] = {}

    for file_path in media_files:
        ext = file_path.suffix.lower().replace(".", "")
        if ext in extraction_report["format_breakdown"]:
            extraction_report["format_breakdown"][ext] += 1
        else:
            extraction_report["format_breakdown"]["other"] += 1

        parsed = parse_filename(file_path)
        if not parsed:
            extraction_report["language_breakdown"]["Unknown"] += 1
            extraction_report["rejected_or_corrupted"] += 1
            extraction_report["errors"].append({
                "file": file_path.name,
                "error": "Failed to parse filename matching SAMVEDNA naming convention"
            })
            continue

        extraction_report["language_breakdown"][parsed["language"]] += 1

        # Check audio stream presence for reporting
        has_audio, _ = inspect_audio_stream(file_path)
        if has_audio:
            extraction_report["files_containing_audio"] += 1
        else:
            extraction_report["video_only_or_silent_files"] += 1

        # Hindi filtering
        if hindi_only and parsed["language_code"] != "H":
            continue

        # Duplicate checking
        utterance_key = f"{parsed['actor']}_{parsed['language_code']}_{parsed['emotion_code']}_{parsed['sentence']}"
        file_hash = compute_file_hash(file_path)

        if utterance_key in seen_utterance_keys or file_hash in seen_file_hashes:
            extraction_report["duplicate_files_detected"] += 1
            extraction_report["duplicates"].append({
                "file": file_path.name,
                "utterance_key": utterance_key,
                "reason": "Duplicate utterance key or duplicate SHA-256 content"
            })
            continue

        seen_utterance_keys.add(utterance_key)
        seen_file_hashes[file_hash] = file_path.name

        # Destination standardized WAV
        dest_filename = f"{utterance_key}.wav"
        dest_path = audio_out_dir / dest_filename

        success, duration, err, metrics = extract_and_normalize_audio(
            file_path, dest_path, target_sr=16000
        )

        if not success:
            extraction_report["rejected_or_corrupted"] += 1
            if "silent" in err.lower() or "no audio" in err.lower():
                extraction_report["video_only_or_silent_files"] += 1
            extraction_report["errors"].append({"file": file_path.name, "error": err})
            continue

        # Valid extraction
        split = assign_split(parsed["actor"], split_config)
        extraction_report["successful_extractions"] += 1
        extraction_report["final_hindi_samples_per_emotion"][parsed["emotion"]] += 1
        extraction_report["splits_distribution"][split] += 1
        extraction_report["actor_distribution"][parsed["actor"]] = (
            extraction_report["actor_distribution"].get(parsed["actor"], 0) + 1
        )

        manifest_rows.append({
            "utterance_id": dest_filename.replace(".wav", ""),
            "filepath": str(dest_path.relative_to(output_dir)),
            "absolute_filepath": str(dest_path.resolve()),
            "actor_id": parsed["actor"],
            "language": parsed["language"],
            "emotion": parsed["emotion"],
            "label_id": LABEL_TO_ID[parsed["emotion"]],
            "sentence_id": parsed["sentence"],
            "duration_seconds": round(duration, 3),
            "sample_rate": 16000,
            "peak_amplitude": metrics["peak"],
            "rms_energy": metrics["rms"],
            "split": split
        })

    # Write manifest.csv
    manifest_csv = output_dir / "manifest.csv"
    if manifest_rows:
        header = list(manifest_rows[0].keys())
        lines = [",".join(header)]
        for r in manifest_rows:
            lines.append(",".join(str(r[k]) for k in header))
        manifest_csv.write_text("\n".join(lines), encoding="utf-8")
    else:
        manifest_csv.write_text("utterance_id,filepath,absolute_filepath,actor_id,language,emotion,label_id,sentence_id,duration_seconds,sample_rate,peak_amplitude,rms_energy,split\n")

    # Write label_mapping.json
    label_map_file = output_dir / "label_mapping.json"
    label_map_file.write_text(json.dumps({
        "label_to_id": LABEL_TO_ID,
        "id_to_label": ID_TO_LABEL
    }, indent=2), encoding="utf-8")

    # Write extraction_report.json
    report_file = output_dir / "extraction_report.json"
    report_file.write_text(json.dumps(extraction_report, indent=2), encoding="utf-8")

    # Write dataset_summary.json (matches Phase 10 IEMOCAP schema)
    summary_data = {
        "dataset_name": "SAMVEDNA-Hindi",
        "total_samples": extraction_report["successful_extractions"],
        "emotion_distribution": extraction_report["final_hindi_samples_per_emotion"],
        "splits": extraction_report["splits_distribution"],
        "actors": extraction_report["actor_distribution"],
        "sample_rate": 16000,
        "format": "mono_16bit_pcm_wav",
        "audio_stream_verified": True
    }
    summary_file = output_dir / "dataset_summary.json"
    summary_file.write_text(json.dumps(summary_data, indent=2), encoding="utf-8")

    return extraction_report


def main():
    parser = argparse.ArgumentParser(description="SAMVEDNA Hindi Audio Extraction & Preprocessing")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR, help="Raw SAMVEDNA data directory")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Preprocessed output directory")
    parser.add_argument("--all-languages", action="store_true", help="Process English & Punjabi as well as Hindi")
    args = parser.parse_args()

    print(f"=== SAMVEDNA Audio Extraction & Preprocessing ===")
    print(f"Input Directory : {args.input_dir}")
    print(f"Output Directory: {args.output_dir}")
    print(f"Language Mode   : {'All (Hindi, English, Punjabi)' if args.all_languages else 'Hindi Only (H / HI)'}")

    report = run_preprocessing(args.input_dir, args.output_dir, hindi_only=not args.all_languages)

    print("\n--- Extraction Summary ---")
    print(f"Total Files Inspected     : {report['total_files_inspected']}")
    print(f"Files Containing Audio    : {report['files_containing_audio']}")
    print(f"Format Breakdown          : {report['format_breakdown']}")
    print(f"Language Breakdown        : {report['language_breakdown']}")
    print(f"Successful Extractions    : {report['successful_extractions']}")
    print(f"Video-Only / Silent Files : {report['video_only_or_silent_files']}")
    print(f"Duplicates Detected       : {report['duplicate_files_detected']}")
    print(f"Final Hindi Samples/Emo   : {report['final_hindi_samples_per_emotion']}")
    print(f"Split Distribution        : {report['splits_distribution']}")
    print(f"Extraction Report Saved   : {args.output_dir / 'extraction_report.json'}")


if __name__ == "__main__":
    main()
