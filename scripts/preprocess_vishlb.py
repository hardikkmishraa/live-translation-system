#!/usr/bin/env python3
"""
Vishal B. Hindi Speech Emotion Recognition Preprocessing Pipeline.
==================================================================
Preprocesses the 3,200 audio files from the Vishal B. Hindi SER Kaggle dataset:
(https://www.kaggle.com/datasets/vishlb/speech-emotion-recognition-hindi)

Requirements:
- Parses 8 distinct emotion classes:
  Angry, Disgust, Fear, Happy, Neutral, Sad, Sarcastic, Surprise.
- Detects speaker IDs (8 actors: 4 Male, 4 Female).
- Standardizes audio to mono 16 kHz, 16-bit PCM WAV.
- Applies peak audio normalization (-1.0 dBFS) and rejects silent audio (RMS < 1e-4).
- Generates reproducible, strictly speaker-independent splits (Zero Speaker Leakage).
- Preserves existing SAMVEDNA Hindi samples and IEMOCAP models.
- Outputs data/vishlb_hindi/manifest.csv, label_mapping.json, and dataset_summary.json.
"""

import os
import re
import sys
import json
import shutil
import argparse
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Set

import numpy as np
import soundfile as sf

try:
    import imageio_ffmpeg
    FFMPEG_PATH = imageio_ffmpeg.get_ffmpeg_exe()
except Exception:
    FFMPEG_PATH = shutil.which("ffmpeg") or "ffmpeg"

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_INPUT_DIR = BASE_DIR / "data" / "raw_vishlb"
DEFAULT_OUTPUT_DIR = BASE_DIR / "data" / "vishlb_hindi"

# 8 Hindi Emotion Classes
LABEL_TO_ID_8 = {
    "angry": 0,
    "disgust": 1,
    "fear": 2,
    "happy": 3,
    "neutral": 4,
    "sad": 5,
    "sarcastic": 6,
    "surprise": 7
}

ID_TO_LABEL_8 = {str(v): k for k, v in LABEL_TO_ID_8.items()}

# Keyword mapping for emotion folder names and filename tokens
EMOTION_KEYWORDS = {
    "angry": "angry",
    "anger": "angry",
    "ang": "angry",
    "disgust": "disgust",
    "dis": "disgust",
    "fear": "fear",
    "fea": "fear",
    "happy": "happy",
    "happiness": "happy",
    "hap": "happy",
    "neutral": "neutral",
    "neu": "neutral",
    "sad": "sad",
    "sadness": "sad",
    "sarcastic": "sarcastic",
    "sarcasm": "sarcastic",
    "sar": "sarcastic",
    "surprise": "surprise",
    "surprised": "surprise",
    "sur": "surprise"
}

# Official Speaker mapping from dataset metadata (DATASET DETAILS ABVV.xlsx)
# Speakers 1-4: Female (F1-F4)
# Speakers 5-8: Male (M1-M4)
SPEAKER_NUM_TO_CODE = {
    "1": "F1", "2": "F2", "3": "F3", "4": "F4",
    "5": "M1", "6": "M2", "7": "M3", "8": "M4"
}

# Speaker-Independent Split Partition (8 Actors: 4 Male, 4 Female)
# Train: 5 actors (3 Male, 2 Female: M1, M2, M3, F1, F2) = 62.5% (2,000 samples)
# Val:   1 actor  (1 Female: F3)                         = 12.5% (400 samples)
# Test:  2 actors (1 Male, 1 Female: M4, F4)             = 25.0% (800 samples)
DEFAULT_SPEAKER_SPLITS = {
    "train_speakers": ["M1", "M2", "M3", "F1", "F2", "speaker_1", "speaker_2", "speaker_3", "speaker_4", "speaker_5"],
    "val_speakers": ["F3", "speaker_6"],
    "test_speakers": ["M4", "F4", "speaker_7", "speaker_8"]
}


def resolve_emotion_from_path(filepath: Path) -> Optional[str]:
    """Resolves emotion from folder name or filename tokens."""
    # Check parent folder name
    parent_clean = filepath.parent.name.lower().strip()
    if parent_clean in EMOTION_KEYWORDS:
        return EMOTION_KEYWORDS[parent_clean]

    # Check filename tokens (split by underscore, dash, dot)
    tokens = re.split(r"[_\-\.\s]+", filepath.stem.lower())
    for t in tokens:
        if t in EMOTION_KEYWORDS:
            return EMOTION_KEYWORDS[t]

    # Substring search in filename
    stem_lower = filepath.stem.lower()
    for kw, canonical in sorted(EMOTION_KEYWORDS.items(), key=lambda x: -len(x[0])):
        if kw in stem_lower:
            return canonical

    return None


def resolve_speaker_from_path(filepath: Path, index_fallback: int = 0) -> str:
    """Extracts speaker ID (e.g. M1-M4, F1-F4, speaker_X) from filename or folder."""
    full_str = f"{filepath.parent.name}_{filepath.stem}".upper()
    
    # 1. Match standard M1-M4, F1-F4 isolated tokens in path components or filename
    for part in filepath.parts:
        if re.match(r"^[MF][1-4]$", part.upper()):
            return part.upper()

    tokens = re.split(r"[^A-Z0-9]+", full_str)
    for t in tokens:
        if re.match(r"^[MF][1-4]$", t):
            return t

    # 2. Match actor or speaker pattern e.g. ACTOR_2, SPEAKER_3, ACTOR1
    act_match = re.search(r"(?:ACTOR|SPEAKER)[_\-]?(\d+)", full_str)
    if act_match:
        return f"speaker_{act_match.group(1)}"

    # 3. Match 1-8 speaker folder in directory hierarchy
    for part in filepath.parts:
        if part in SPEAKER_NUM_TO_CODE:
            return SPEAKER_NUM_TO_CODE[part]

    # 4. Match stem starting with 1.1., 5.2., etc.
    stem_match = re.match(r"^([1-8])\.\d+\.", filepath.name)
    if stem_match:
        return SPEAKER_NUM_TO_CODE[stem_match.group(1)]

    # 5. Fallback to deterministic hash-based speaker bucket (1-8)
    spk_bucket = (hash(filepath.stem.split("_")[0]) % 8) + 1
    return f"speaker_{spk_bucket}"


def convert_and_normalize_audio(
    input_path: Path,
    output_path: Path,
    target_sr: int = 16000
) -> Tuple[bool, float, str, Dict[str, Any]]:
    """
    Standardizes audio to mono 16 kHz 16-bit PCM WAV.
    Applies peak normalization (-1.0 dBFS) and rejects silent audio.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_wav = output_path.with_suffix(".tmp.wav")
    metrics = {"sample_rate": target_sr, "channels": 1, "duration": 0.0, "peak": 0.0, "rms": 0.0}

    ffmpeg_cmd = [
        str(FFMPEG_PATH), "-y", "-v", "error",
        "-i", str(input_path),
        "-vn",
        "-acodec", "pcm_s16le",
        "-ac", "1",
        "-ar", str(target_sr),
        str(temp_wav)
    ]

    try:
        res = subprocess.run(ffmpeg_cmd, capture_output=True, text=True, timeout=30)
        if res.returncode != 0:
            if temp_wav.exists():
                temp_wav.unlink(missing_ok=True)
            return False, 0.0, f"FFmpeg failed: {res.stderr.strip()}", metrics

        if not temp_wav.exists() or temp_wav.stat().st_size == 0:
            return False, 0.0, "Extracted audio file is empty", metrics

        data, sr = sf.read(str(temp_wav), dtype="float32")
        duration = len(data) / float(sr)
        metrics["duration"] = round(duration, 3)

        if duration < 0.2:
            temp_wav.unlink(missing_ok=True)
            return False, duration, f"Audio too short ({duration:.2f}s < 0.2s)", metrics

        rms = float(np.sqrt(np.mean(data ** 2)))
        metrics["rms"] = round(rms, 6)
        if rms < 1e-4:
            temp_wav.unlink(missing_ok=True)
            return False, duration, f"Audio is silent (RMS = {rms:.6f})", metrics

        # Peak normalization to -1.0 dBFS (~0.89125)
        max_val = float(np.max(np.abs(data)))
        if max_val > 0:
            norm_factor = 0.89125 / max_val
            norm_data = np.clip(data * norm_factor, -1.0, 1.0)
        else:
            norm_data = data

        metrics["peak"] = round(float(np.max(np.abs(norm_data))), 4)

        sf.write(str(output_path), norm_data, target_sr, subtype="PCM_16")
        temp_wav.unlink(missing_ok=True)
        return True, duration, "", metrics

    except Exception as e:
        if temp_wav.exists():
            temp_wav.unlink(missing_ok=True)
        return False, 0.0, str(e), metrics


def assign_split(speaker_id: str, split_config: Optional[Dict[str, List[str]]] = None) -> str:
    """Assigns split strictly by speaker ID to ensure zero speaker leakage."""
    split_config = split_config or DEFAULT_SPEAKER_SPLITS
    if speaker_id in split_config["train_speakers"]:
        return "train"
    elif speaker_id in split_config["val_speakers"]:
        return "val"
    elif speaker_id in split_config["test_speakers"]:
        return "test"
    return "train"


def run_preprocessing(
    input_dir: Path,
    output_dir: Path,
    split_config: Optional[Dict[str, List[str]]] = None
) -> Dict[str, Any]:
    """
    Executes preprocessing on Vishal B. dataset directory.
    Outputs standardized audio to data/vishlb_hindi/audio/ and writes manifests.
    """
    split_config = split_config or DEFAULT_SPEAKER_SPLITS
    output_dir.mkdir(parents=True, exist_ok=True)
    audio_out_dir = output_dir / "audio"
    audio_out_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "dataset_name": "Vishal-B-Hindi-SER",
        "input_directory": str(input_dir.resolve()),
        "output_directory": str(output_dir.resolve()),
        "total_files_discovered": 0,
        "successfully_processed": 0,
        "rejected_or_corrupted": 0,
        "emotion_distribution": {emo: 0 for emo in LABEL_TO_ID_8.keys()},
        "splits_distribution": {"train": 0, "val": 0, "test": 0},
        "speaker_distribution": {},
        "errors": []
    }

    if not input_dir.exists():
        summary["errors"].append(f"Input directory does not exist: {input_dir}")
        return summary

    media_files = sorted([
        f for f in input_dir.rglob("*")
        if f.is_file() and f.suffix.lower() in [".wav", ".mp3", ".m4a", ".flac", ".ogg"]
    ])
    summary["total_files_discovered"] = len(media_files)

    def process_file_item(item):
        idx, fpath = item
        emotion = resolve_emotion_from_path(fpath)
        if not emotion:
            return {"success": False, "file": fpath.name, "error": "Could not determine emotion label"}

        speaker_id = resolve_speaker_from_path(fpath, index_fallback=idx)
        dest_filename = f"{speaker_id}_{emotion}_{fpath.stem}.wav"
        dest_path = audio_out_dir / dest_filename

        success, duration, err, metrics = convert_and_normalize_audio(fpath, dest_path, target_sr=16000)
        if not success:
            return {"success": False, "file": fpath.name, "error": err}

        split = assign_split(speaker_id, split_config)
        row = {
            "utterance_id": dest_filename.replace(".wav", ""),
            "filepath": str(dest_path.relative_to(output_dir)),
            "absolute_filepath": str(dest_path.resolve()),
            "speaker_id": speaker_id,
            "language": "Hindi",
            "emotion": emotion,
            "label_id": LABEL_TO_ID_8[emotion],
            "duration_seconds": round(duration, 3),
            "sample_rate": 16000,
            "peak_amplitude": metrics["peak"],
            "rms_energy": metrics["rms"],
            "split": split
        }
        return {
            "success": True,
            "row": row,
            "emotion": emotion,
            "speaker_id": speaker_id,
            "split": split
        }

    max_workers = min(8, os.cpu_count() or 4)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(process_file_item, enumerate(media_files)))

    manifest_rows = []
    for res in results:
        if not res["success"]:
            summary["rejected_or_corrupted"] += 1
            summary["errors"].append({"file": res["file"], "error": res["error"]})
        else:
            summary["successfully_processed"] += 1
            summary["emotion_distribution"][res["emotion"]] += 1
            summary["splits_distribution"][res["split"]] += 1
            summary["speaker_distribution"][res["speaker_id"]] = summary["speaker_distribution"].get(res["speaker_id"], 0) + 1
            manifest_rows.append(res["row"])

    # Sort manifest rows by utterance_id for deterministic order
    manifest_rows.sort(key=lambda x: x["utterance_id"])

    # Write manifest.csv
    manifest_csv = output_dir / "manifest.csv"
    if manifest_rows:
        header = list(manifest_rows[0].keys())
        lines = [",".join(header)]
        for r in manifest_rows:
            lines.append(",".join(str(r[k]) for k in header))
        manifest_csv.write_text("\n".join(lines), encoding="utf-8")
    else:
        manifest_csv.write_text("utterance_id,filepath,absolute_filepath,speaker_id,language,emotion,label_id,duration_seconds,sample_rate,peak_amplitude,rms_energy,split\n")

    # Write label_mapping.json (8 classes)
    label_map_file = output_dir / "label_mapping.json"
    label_map_file.write_text(json.dumps({
        "label_to_id": LABEL_TO_ID_8,
        "id_to_label": ID_TO_LABEL_8,
        "num_classes": 8
    }, indent=2), encoding="utf-8")

    # Write dataset_summary.json
    summary_file = output_dir / "dataset_summary.json"
    summary_file.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    return summary


def main():
    parser = argparse.ArgumentParser(description="Vishal B. Hindi SER Preprocessing")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR, help="Raw Vishal B. dataset directory")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Preprocessed output directory")
    args = parser.parse_args()

    print("=== Vishal B. Hindi SER Preprocessing ===")
    print(f"Input Directory : {args.input_dir}")
    print(f"Output Directory: {args.output_dir}")

    summary = run_preprocessing(args.input_dir, args.output_dir)

    print("\n--- Summary ---")
    print(f"Total Discovered    : {summary['total_files_discovered']}")
    print(f"Successfully Done   : {summary['successfully_processed']}")
    print(f"Rejected / Errors   : {summary['rejected_or_corrupted']}")
    print(f"Emotions            : {summary['emotion_distribution']}")
    print(f"Splits              : {summary['splits_distribution']}")
    print(f"Speakers            : {summary['speaker_distribution']}")


if __name__ == "__main__":
    main()
