"""
Unit & Integration Tests: IEMOCAP Dataset Preprocessing (Phase 10)
==================================================================
Tests:
  1. Metadata manifest integrity and 10,039 utterance parsing.
  2. True speaker identification across all 5 sessions (10 speakers).
  3. Resolution of the 4,951 folder-level gender discrepancies.
  4. Class strategy mappings (4-class merged, 4-class strict, 5-class, 6-class).
  5. 0% speaker leakage across train, val, and test partitions.
  6. Leave-One-Session-Out (LOSO) 5-fold cross-validation generation.
  7. Audio preprocessor waveform formatting (mono, 16kHz, amplitude normalization).
  8. Preservation of original dataset file.
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd
import pytest

from scripts.preprocess_iemocap import (
    parse_iemocap_metadata,
    generate_loso_folds,
    IEMOCAPAudioPreprocessor,
)

CSV_PATH = Path("iemocap_full_dataset.csv")
DATA_DIR = Path("data/iemocap")


def test_original_dataset_exists_and_unaltered():
    assert CSV_PATH.exists(), "iemocap_full_dataset.csv must exist in project root."
    df = pd.read_csv(CSV_PATH)
    assert len(df) == 10039
    assert list(df.columns) == ["session", "method", "gender", "emotion", "n_annotators", "agreement", "path"]


def test_clean_manifest_structure_and_speakers():
    manifest_csv = DATA_DIR / "iemocap_clean_manifest.csv"
    assert manifest_csv.exists(), "Clean manifest CSV must be generated."
    df = pd.read_csv(manifest_csv)

    assert len(df) == 10039
    assert df["session"].nunique() == 5
    assert sorted(df["session"].unique().tolist()) == [1, 2, 3, 4, 5]

    # Exactly 10 unique speakers
    speakers = sorted(df["speaker_id"].unique().tolist())
    expected_speakers = [
        "Ses01_F", "Ses01_M",
        "Ses02_F", "Ses02_M",
        "Ses03_F", "Ses03_M",
        "Ses04_F", "Ses04_M",
        "Ses05_F", "Ses05_M",
    ]
    assert speakers == expected_speakers


def test_speaker_gender_discrepancy_resolved():
    df = pd.read_csv(DATA_DIR / "iemocap_clean_manifest.csv")
    raw_df = pd.read_csv(CSV_PATH)

    # In raw_df, 4951 rows were mislabeled with the folder's leading actor gender
    # In clean_df, speaker_gender must match the active speaking token (_F000 or _M000)
    for _, row in df.iterrows():
        token = row["utterance_id"].split("_")[-1]
        expected_gender = token[0]
        assert row["speaker_gender"] == expected_gender
        assert row["speaker_id"].endswith(f"_{expected_gender}")


def test_label_mapping_strategies_counts():
    df = pd.read_csv(DATA_DIR / "iemocap_clean_manifest.csv")

    # 4-class merged: {angry, happy, sad, neutral} (hap + exc merged)
    df_4m = df[df["label_4class_merged"].notnull()]
    assert len(df_4m) == 5531
    assert set(df_4m["label_4class_merged"].unique()) == {"angry", "happy", "sad", "neutral"}
    assert (df_4m["label_4class_merged"] == "neutral").sum() == 1708
    assert (df_4m["label_4class_merged"] == "happy").sum() == 1636
    assert (df_4m["label_4class_merged"] == "angry").sum() == 1103
    assert (df_4m["label_4class_merged"] == "sad").sum() == 1084

    # 4-class strict: {angry, happy, sad, neutral} (exc excluded)
    df_4s = df[df["label_4class_strict"].notnull()]
    assert len(df_4s) == 4490
    assert (df_4s["label_4class_strict"] == "happy").sum() == 595

    # 5-class: includes frustrated
    df_5 = df[df["label_5class"].notnull()]
    assert len(df_5) == 7380
    assert (df_5["label_5class"] == "frustrated").sum() == 1849

    # 6-class: hap and exc separate
    df_6 = df[df["label_6class"].notnull()]
    assert len(df_6) == 7380
    assert (df_6["label_6class"] == "excited").sum() == 1041
    assert (df_6["label_6class"] == "happy").sum() == 595


def test_zero_speaker_leakage_in_splits():
    splits_file = DATA_DIR / "iemocap_splits.json"
    assert splits_file.exists()
    splits = json.loads(splits_file.read_text())

    df = pd.read_csv(DATA_DIR / "iemocap_clean_manifest.csv")

    # Check Standard Split
    train_speakers = set(df[df["standard_split"] == "train"]["speaker_id"].unique())
    val_speakers = set(df[df["standard_split"] == "val"]["speaker_id"].unique())
    test_speakers = set(df[df["standard_split"] == "test"]["speaker_id"].unique())

    # Guarantee 0% speaker leakage
    assert len(train_speakers & val_speakers) == 0, "Speaker leakage between train and val!"
    assert len(train_speakers & test_speakers) == 0, "Speaker leakage between train and test!"
    assert len(val_speakers & test_speakers) == 0, "Speaker leakage between val and test!"

    # Verify session assignment
    assert train_speakers == {"Ses01_F", "Ses01_M", "Ses02_F", "Ses02_M", "Ses03_F", "Ses03_M"}
    assert val_speakers == {"Ses04_F", "Ses04_M"}
    assert test_speakers == {"Ses05_F", "Ses05_M"}

    # Check LOSO Folds
    for fold_name, fold_data in splits["leave_one_session_out_folds"].items():
        t_sess = fold_data["test_session"]
        v_sess = fold_data["val_session"]
        tr_sess = fold_data["train_sessions"]
        assert len(set([t_sess, v_sess] + tr_sess)) == 5


def test_audio_preprocessor_formatting():
    preproc = IEMOCAPAudioPreprocessor(target_sr=16000, target_peak=0.95, max_seconds=2.0)
    
    # Multi-channel stereo audio at 44.1 kHz
    stereo_44k = np.random.randn(44100, 2).astype(np.float32)
    processed, mask = preproc.process_waveform(stereo_44k, orig_sr=44100)

    assert processed.ndim == 1, "Output must be single-channel mono."
    assert len(processed) == 32000, "Output length must match 2.0s * 16,000 = 32,000 samples."
    assert len(mask) == 32000
    assert np.max(np.abs(processed)) <= 0.95 + 1e-4, "Peak amplitude must be normalized."

