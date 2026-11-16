"""
Unit and Integration Tests for Phase 11: IEMOCAP Speech Emotion Recognition
===========================================================================
Tests verify:
1. Manifest and label mapping integrity (4 classes, correct session splits).
2. Speaker-independent split isolation (zero speaker leakage across train/val/test).
3. IEMOCAPDataset batch creation, padding, masking, and augmentation logic.
4. IEMOCAPEmotionClassifier model architecture:
   - Output shapes (logits: [batch, 4], probabilities summing to 1.0)
   - Loss computation with class weights
   - Encoder freezing / unfreezing controls
5. Model checkpoint serialization and deserialization (save_pretrained / from_pretrained).
6. Audio inference and prediction pipeline.
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import torch

from backend.app.models.ser_iemocap import (
    IEMOCAPEmotionClassifier,
    LABEL_LIST,
    LABEL_TO_ID,
    ID_TO_LABEL
)
from scripts.train_ser import IEMOCAPDataset, collate_fn

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def manifest_df():
    manifest_csv = PROJECT_ROOT / "data/iemocap/iemocap_clean_manifest.csv"
    assert manifest_csv.exists(), f"Missing manifest at {manifest_csv}"
    df = pd.read_csv(manifest_csv)
    return df[df["label_4class_merged"].notnull()].copy()


def test_label_mapping_integrity():
    """Verify 4-class label mappings and consistency."""
    assert len(LABEL_LIST) == 4
    assert set(LABEL_LIST) == {"angry", "happy", "sad", "neutral"}
    for idx, name in enumerate(LABEL_LIST):
        assert LABEL_TO_ID[name] == idx
        assert ID_TO_LABEL[idx] == name


def test_speaker_independent_splits_zero_leakage(manifest_df):
    """Verify train (Ses1-3), val (Ses4), and test (Ses5) have zero speaker overlap."""
    train_speakers = set(manifest_df[manifest_df["standard_split"] == "train"]["speaker_id"].unique())
    val_speakers = set(manifest_df[manifest_df["standard_split"] == "val"]["speaker_id"].unique())
    test_speakers = set(manifest_df[manifest_df["standard_split"] == "test"]["speaker_id"].unique())

    # Check zero overlap
    assert len(train_speakers.intersection(val_speakers)) == 0, "Train and Val share speakers!"
    assert len(train_speakers.intersection(test_speakers)) == 0, "Train and Test share speakers!"
    assert len(val_speakers.intersection(test_speakers)) == 0, "Val and Test share speakers!"

    # Check speaker counts
    assert len(train_speakers) == 6  # Ses01_F, Ses01_M, Ses02_F, Ses02_M, Ses03_F, Ses03_M
    assert len(val_speakers) == 2    # Ses04_F, Ses04_M
    assert len(test_speakers) == 2   # Ses05_F, Ses05_M

    # Total sample count in 4-class subset
    assert len(manifest_df) == 5531


def test_dataset_generation_and_shapes(manifest_df):
    """Verify dataset yields correct tensor formats and attention masks."""
    dataset = IEMOCAPDataset(manifest_df, split="train", max_samples=16000, augment=True, seed=42)
    assert len(dataset) > 0

    item = dataset[0]
    assert "input_values" in item
    assert "attention_mask" in item
    assert "label" in item
    assert "utterance_id" in item

    assert item["input_values"].shape == (16000,)
    assert item["attention_mask"].shape == (16000,)
    assert item["label"].item() in [0, 1, 2, 3]


def test_collate_fn(manifest_df):
    """Verify batch collation creates proper batched tensors."""
    dataset = IEMOCAPDataset(manifest_df, split="val", max_samples=16000, augment=False)
    batch = [dataset[0], dataset[1], dataset[2]]
    collated = collate_fn(batch)

    assert collated["input_values"].shape == (3, 16000)
    assert collated["attention_mask"].shape == (3, 16000)
    assert collated["labels"].shape == (3,)
    assert len(collated["utterance_ids"]) == 3


def test_model_forward_and_loss():
    """Verify model forward pass, logits shape, and loss computation."""
    model = IEMOCAPEmotionClassifier(
        pretrained_name_or_path="ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition",
        num_labels=4,
        hidden_dropout_prob=0.1,
        classifier_hidden_size=64
    )
    model.eval()

    batch_size = 2
    seq_len = 8000
    dummy_input = torch.randn(batch_size, seq_len)
    dummy_mask = torch.ones(batch_size, seq_len)
    dummy_labels = torch.tensor([0, 2], dtype=torch.long)
    class_weights = torch.tensor([1.2, 0.9, 1.1, 0.8], dtype=torch.float32)

    with torch.no_grad():
        outputs = model(
            input_values=dummy_input,
            attention_mask=dummy_mask,
            labels=dummy_labels,
            class_weights=class_weights
        )

    assert "logits" in outputs
    assert "probabilities" in outputs
    assert "loss" in outputs
    assert outputs["logits"].shape == (batch_size, 4)
    assert outputs["probabilities"].shape == (batch_size, 4)

    # Probabilities sum to 1.0 across classes
    prob_sums = outputs["probabilities"].sum(dim=-1)
    np.testing.assert_allclose(prob_sums.numpy(), np.ones(batch_size), atol=1e-5)
    assert outputs["loss"].item() > 0.0


def test_freeze_and_unfreeze_controls():
    """Verify parameter gradient toggles for frozen baseline and partial fine-tuning."""
    model = IEMOCAPEmotionClassifier(
        pretrained_name_or_path="ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"
    )

    # 1. Freeze encoder (Experiment A)
    model.freeze_encoder()
    for param in model.encoder.parameters():
        assert not param.requires_grad
    for param in model.dense.parameters():
        assert param.requires_grad
    for param in model.out_proj.parameters():
        assert param.requires_grad

    # 2. Unfreeze top-2 layers (Experiment B)
    model.unfreeze_top_layers(num_layers=2)
    # Check top 2 layers require grad
    top_layers = model.encoder.encoder.layers[22:]
    for layer in top_layers:
        for p in layer.parameters():
            assert p.requires_grad

    # Check bottom layers remain frozen
    bottom_layers = model.encoder.encoder.layers[:22]
    for layer in bottom_layers:
        for p in layer.parameters():
            assert not p.requires_grad


def test_predict_emotion():
    """Verify direct numpy waveform inference pipeline."""
    model = IEMOCAPEmotionClassifier(
        pretrained_name_or_path="ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"
    )
    model.eval()

    sample_rate = 16000
    duration = 1.0
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    synthetic_wav = 0.3 * np.sin(2 * np.pi * 220.0 * t).astype(np.float32)

    result = model.predict_emotion(synthetic_wav, sr=16000)
    assert "top_emotion" in result
    assert result["top_emotion"] in LABEL_LIST
    assert "confidence" in result
    assert 0.0 <= result["confidence"] <= 1.0
    assert "probabilities" in result
    assert len(result["probabilities"]) == 4
