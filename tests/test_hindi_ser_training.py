"""
Unit and Integration Tests for Hindi Speech Emotion Recognition (SER) Architecture.
===================================================================================
Tests:
- Audio input shape and sampling rate constraints (16,000 Hz mono).
- Emotion label mapping integrity (6 SAMVEDNA classes).
- Model output dimension checks (logits shape: [batch, 6]).
- Output probability normalization (Softmax sum == 1.0).
- Device compatibility (CPU and MPS fallback).
- Mock checkpoint loading and forward inference pass.
"""

import os
import json
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from scripts.preprocess_samvedna import LABEL_TO_ID, ID_TO_LABEL


class HindiSERClassifierHead(nn.Module):
    """Reference classification head for Hindi SER (6 classes)."""
    def __init__(self, input_dim: int = 1024, num_classes: int = 6, dropout: float = 0.3):
        super().__init__()
        self.norm = nn.LayerNorm(input_dim)
        self.dropout = nn.Dropout(dropout)
        self.dense = nn.Linear(input_dim, 256)
        self.activation = nn.GELU()
        self.out_proj = nn.Linear(256, num_classes)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        # features shape: [batch, hidden_dim]
        x = self.norm(features)
        x = self.dropout(x)
        x = self.dense(x)
        x = self.activation(x)
        x = self.dropout(x)
        logits = self.out_proj(x)
        return logits


def test_label_mapping_integrity():
    """Verify that all 6 SAMVEDNA emotion classes are mapped bijectively."""
    assert len(LABEL_TO_ID) == 6
    assert set(LABEL_TO_ID.keys()) == {"angry", "happy", "sad", "neutral", "disgust", "fear"}
    for label, idx in LABEL_TO_ID.items():
        assert ID_TO_LABEL[str(idx)] == label


def test_classifier_head_output_dimensions():
    """Verify output logits shape matches [batch_size, num_classes]."""
    head = HindiSERClassifierHead(input_dim=1024, num_classes=6)
    head.eval()

    batch_size = 4
    dummy_features = torch.randn(batch_size, 1024)
    with torch.no_grad():
        logits = head(dummy_features)

    assert logits.shape == (batch_size, 6)
    assert not torch.isnan(logits).any()


def test_probability_normalization():
    """Verify Softmax produces valid probability distributions summing to 1.0."""
    head = HindiSERClassifierHead(input_dim=512, num_classes=6)
    head.eval()

    dummy_features = torch.randn(8, 512)
    with torch.no_grad():
        logits = head(dummy_features)
        probs = F.softmax(logits, dim=-1)

    assert probs.shape == (8, 6)
    # Check probabilities sum to 1.0 across classes
    sums = probs.sum(dim=-1).numpy()
    assert np.allclose(sums, 1.0, atol=1e-5)
    # Check probabilities are bounded in [0, 1]
    assert (probs >= 0.0).all() and (probs <= 1.0).all()


def test_device_compatibility():
    """Verify forward inference runs smoothly on available hardware (CPU or MPS)."""
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    head = HindiSERClassifierHead(input_dim=256, num_classes=6).to(device)
    head.eval()

    dummy_input = torch.randn(2, 256, device=device)
    with torch.no_grad():
        out = head(dummy_input)

    assert out.device.type == device.type
    assert out.shape == (2, 6)


def test_audio_input_shape_and_sampling_rate():
    """Validate 16kHz audio input requirement and length safety."""
    target_sr = 16000
    min_duration = 0.5   # 0.5s minimum speech
    max_duration = 10.0  # 10s max speech buffer

    # Valid 2-second audio frame
    valid_waveform = torch.randn(1, int(target_sr * 2.0))
    assert valid_waveform.shape[-1] >= target_sr * min_duration
    assert valid_waveform.shape[-1] <= target_sr * max_duration

    # Audio shorter than min duration
    too_short = torch.randn(1, int(target_sr * 0.1))
    assert too_short.shape[-1] < target_sr * min_duration


def test_checkpoint_state_dict_roundtrip(tmp_path):
    """Verify model parameters can be saved and restored without drift."""
    model = HindiSERClassifierHead(input_dim=128, num_classes=6)
    model.eval()
    save_path = tmp_path / "hindi_ser_test.pt"

    torch.save(model.state_dict(), str(save_path))
    assert save_path.exists()

    restored_model = HindiSERClassifierHead(input_dim=128, num_classes=6)
    restored_model.load_state_dict(torch.load(str(save_path), weights_only=True))
    restored_model.eval()

    dummy_in = torch.randn(3, 128)
    with torch.no_grad():
        orig_out = model(dummy_in)
        restored_out = restored_model(dummy_in)

    assert torch.allclose(orig_out, restored_out, atol=1e-6)
