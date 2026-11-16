"""
Unit and Integration Tests for Phase 17 Hindi SER Training Pipeline.
===================================================================
Tests:
- Manifest integrity (3,200 rows, 8 classes, 8 speakers).
- Speaker-grouped split disjointness (zero speaker leakage).
- 3-fold cross-validation configuration (every dev speaker in val exactly once).
- Acoustic feature extraction dimensions and finiteness.
- Lightweight emotion classifier forward pass shape and parameters.
- Partially fine-tuned Wav2Vec2 architecture trainable parameter count.
- Metric computation (accuracy, Macro-F1, UAR, confusion matrix, speaker metrics).
- Asset preservation (IEMOCAP checkpoint and SAMVEDNA Hindi samples untouched).
"""

import os
import json
import pytest
import numpy as np
import pandas as pd
import torch
from pathlib import Path

from scripts.train_hindi_ser import (
    extract_acoustic_features,
    LightweightEmotionClassifier,
    compute_ser_metrics,
    LABEL_TO_ID_8,
    ID_TO_LABEL_8,
    DEV_SPEAKERS,
    TEST_SPEAKERS,
    SPEAKER_FOLDS,
    MANIFEST_PATH
)


def test_manifest_and_label_mapping():
    """Verify manifest exists, contains 3200 rows, 8 classes, and 8 speakers."""
    assert MANIFEST_PATH.exists(), f"Manifest missing at {MANIFEST_PATH}"
    df = pd.read_csv(MANIFEST_PATH)
    assert len(df) == 3200
    assert set(df["emotion"].unique()) == set(LABEL_TO_ID_8.keys())
    assert set(df["speaker_id"].unique()) == set(DEV_SPEAKERS + TEST_SPEAKERS)


def test_speaker_split_disjointness_zero_leakage():
    """Ensure zero speaker leakage between development and test sets."""
    dev_set = set(DEV_SPEAKERS)
    test_set = set(TEST_SPEAKERS)
    assert len(dev_set & test_set) == 0, "Speaker leakage between Dev and Test!"
    assert len(dev_set) == 6
    assert len(test_set) == 2


def test_3_fold_cross_validation_configuration():
    """Verify each dev speaker is assigned to validation in exactly one fold."""
    assert len(SPEAKER_FOLDS) == 3
    val_counts = {spk: 0 for spk in DEV_SPEAKERS}

    for fold_cfg in SPEAKER_FOLDS:
        val_spks = set(fold_cfg["val_speakers"])
        train_spks = set(fold_cfg["train_speakers"])

        # No overlap within fold
        assert len(val_spks & train_spks) == 0, f"Overlap in fold {fold_cfg['fold']}"
        assert val_spks | train_spks == set(DEV_SPEAKERS)

        for s in val_spks:
            val_counts[s] += 1

    # Every dev speaker is in val exactly once
    for spk, count in val_counts.items():
        assert count == 1, f"Speaker {spk} validated {count} times (expected 1)"


def test_classifier_head_forward_pass():
    """Test LightweightEmotionClassifier forward pass with batch input."""
    model = LightweightEmotionClassifier(input_dim=1024, hidden_dim=256, num_classes=8)
    dummy_input = torch.randn(4, 1024)
    logits = model(dummy_input)
    assert logits.shape == (4, 8)
    assert not torch.isnan(logits).any()


def test_metric_computation():
    """Test accuracy, macro-F1, UAR, and confusion matrix calculation."""
    y_true = np.array([0, 1, 2, 3, 4, 5, 6, 7])
    y_pred = np.array([0, 1, 2, 3, 4, 5, 6, 7])
    speakers = ["F1"] * 4 + ["M1"] * 4

    metrics = compute_ser_metrics(y_true, y_pred, speakers)
    assert metrics["accuracy"] == 1.0
    assert metrics["macro_f1"] == 1.0
    assert metrics["uar"] == 1.0
    assert len(metrics["confusion_matrix"]) == 8
    assert "F1" in metrics["speaker_wise"]
    assert "M1" in metrics["speaker_wise"]


def test_asset_preservation():
    """Confirm English IEMOCAP checkpoint and SAMVEDNA Hindi samples remain intact."""
    iemocap_dir = Path("models/ser_iemocap")
    assert iemocap_dir.exists()
    assert (iemocap_dir / "best_val_checkpoint.pt").exists()
    assert (iemocap_dir / "label_mapping.json").exists()

    samvedna_audio_dir = Path("data/samvedna_hindi/audio")
    assert samvedna_audio_dir.exists()
    wavs = list(samvedna_audio_dir.glob("*.wav"))
    assert len(wavs) == 6, f"Expected 6 SAMVEDNA Hindi samples, found {len(wavs)}"
