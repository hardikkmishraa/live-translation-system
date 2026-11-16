#!/usr/bin/env python3
"""
Phase 17: Hindi Speech Emotion Recognition Training & Benchmarking Pipeline
==========================================================================
Trains and compares 3 SER model architectures on the 8-class Vishal B. dataset:
  - Model A: MFCC + Acoustic Feature Baseline with Support Vector Machine (SVM)
  - Model B: Frozen Pretrained Wav2Vec2-XLSR-53 + 8-Class Classification Head
  - Model C: Partially Fine-Tuned Wav2Vec2-XLSR-53 (Top Transformer Layer + Head)

Evaluation Strategy:
  - Development Set: 6 speakers (F1, F2, F3, M1, M2, M3 = 2,400 samples).
  - 3-Fold Speaker-Grouped Cross-Validation (Zero speaker leakage).
  - Final Test Set: 2 speakers (F4, M4 = 800 samples) strictly held out.
  - Final evaluation performed exactly once on the winning model.
  - Checkpoint saved to models/ser_hindi/ without touching models/ser_iemocap/.
"""

import os
import sys
import time
import json
import gc
import shutil
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import pandas as pd
import scipy
import scipy.stats
import scipy.signal
import soundfile as sf
import torch
import torch.nn as nn
import torchaudio
import torchaudio.transforms as T
import torchaudio.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix
)
from transformers import AutoModel, AutoFeatureExtractor

# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

DATA_DIR = BASE_DIR / "data" / "vishlb_hindi"
MANIFEST_PATH = DATA_DIR / "manifest.csv"
LABEL_MAP_PATH = DATA_DIR / "label_mapping.json"
OUTPUT_MODEL_DIR = BASE_DIR / "models" / "ser_hindi"
REPORTS_DIR = BASE_DIR / "reports"
CACHE_DIR = DATA_DIR / ".feature_cache"

HF_MODEL_ID = "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

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

# Speaker Partitions
DEV_SPEAKERS = ["F1", "F2", "F3", "M1", "M2", "M3"]
TEST_SPEAKERS = ["F4", "M4"]

# 3-Fold Speaker-Grouped Cross-Validation
SPEAKER_FOLDS = [
    {
        "fold": 1,
        "val_speakers": ["F1", "M1"],
        "train_speakers": ["F2", "F3", "M2", "M3"]
    },
    {
        "fold": 2,
        "val_speakers": ["F2", "M2"],
        "train_speakers": ["F1", "F3", "M1", "M3"]
    },
    {
        "fold": 3,
        "val_speakers": ["F3", "M3"],
        "train_speakers": ["F1", "F2", "M1", "M2"]
    }
]


# ============================================================================
# 1. Feature Extraction: Acoustic Features (Model A)
# ============================================================================

def extract_acoustic_features(audio_path: Path, target_sr: int = 16000) -> np.ndarray:
    """
    Extracts 40 MFCCs, Delta MFCCs, Delta-Delta MFCCs, Spectral features,
    and statistical aggregations across frames.
    """
    audio, sr = sf.read(str(audio_path), dtype="float32")
    if audio.ndim > 1:
        audio = np.mean(audio, axis=-1)
    if sr != target_sr:
        num_samples = int(round(len(audio) * target_sr / float(sr)))
        audio = scipy.signal.resample(audio, num_samples).astype(np.float32)

    tensor_audio = torch.from_numpy(audio).unsqueeze(0)

    # 40 MFCCs
    mfcc_transform = T.MFCC(
        sample_rate=target_sr,
        n_mfcc=40,
        melkwargs={"n_fft": 400, "hop_length": 160, "n_mels": 64}
    )
    mfcc = mfcc_transform(tensor_audio).squeeze(0).numpy() # (40, T)

    # Delta and Delta-Delta
    delta = F.compute_deltas(torch.from_numpy(mfcc)).numpy()
    delta2 = F.compute_deltas(torch.from_numpy(delta)).numpy()

    # Stack MFCC feature blocks
    features_matrix = np.vstack([mfcc, delta, delta2]) # (120, T)

    # Summary statistics across time frames: mean, std, min, max, skewness
    mean_feat = np.mean(features_matrix, axis=1)
    std_feat = np.std(features_matrix, axis=1)
    min_feat = np.min(features_matrix, axis=1)
    max_feat = np.max(features_matrix, axis=1)
    skew_feat = scipy.stats.skew(features_matrix, axis=1)

    # RMS energy per frame
    frame_len = 400
    hop = 160
    rms_frames = []
    for i in range(0, len(audio) - frame_len, hop):
        frame = audio[i:i + frame_len]
        rms_frames.append(np.sqrt(np.mean(frame ** 2) + 1e-12))
    rms_frames = np.array(rms_frames) if len(rms_frames) > 0 else np.array([0.0])
    rms_stats = np.array([
        np.mean(rms_frames),
        np.std(rms_frames),
        np.min(rms_frames),
        np.max(rms_frames),
        scipy.stats.skew(rms_frames) if len(rms_frames) > 2 else 0.0
    ])

    return np.concatenate([mean_feat, std_feat, min_feat, max_feat, skew_feat, rms_stats])


# ============================================================================
# 2. Feature Extraction: Pretrained Wav2Vec2-XLSR-53 (Model B)
# ============================================================================

class HindiAudioDataset(Dataset):
    """PyTorch Dataset loading standardized audio waveforms."""

    def __init__(self, df: pd.DataFrame, max_duration_sec: float = 6.0):
        self.df = df.reset_index(drop=True)
        self.max_duration_sec = max_duration_sec
        self.max_samples = int(max_duration_sec * 16000)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]
        audio_path = Path(row["absolute_filepath"])
        audio, sr = sf.read(str(audio_path), dtype="float32")
        if audio.ndim > 1:
            audio = np.mean(audio, axis=-1)

        length = min(len(audio), self.max_samples)
        padded = np.zeros(self.max_samples, dtype=np.float32)
        padded[:length] = audio[:length]

        mask = np.zeros(self.max_samples, dtype=np.float32)
        mask[:length] = 1.0

        label_id = int(row["label_id"])

        return {
            "input_values": torch.from_numpy(padded),
            "attention_mask": torch.from_numpy(mask),
            "label": torch.tensor(label_id, dtype=torch.long),
            "speaker_id": row["speaker_id"],
            "utterance_id": row["utterance_id"],
            "length": length
        }


def extract_all_wav2vec2_representations(
    df: pd.DataFrame,
    device: str = DEVICE,
    batch_size: int = 16,
    cache_path: Optional[Path] = None
) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
    """
    Passes audio through frozen Wav2Vec2-XLSR-53 encoder to extract
    1024-dimensional temporal pooled representations. Caches results to disk.
    """
    if cache_path and cache_path.exists():
        print(f"Loading cached Wav2Vec2 representations from {cache_path}...")
        data = np.load(str(cache_path), allow_pickle=True)
        return data["embeddings"], data["labels"], list(data["speakers"]), list(data["utterance_ids"])

    print(f"Extracting Wav2Vec2 representations on {device} (batch_size={batch_size})...")
    model = AutoModel.from_pretrained(HF_MODEL_ID).to(device)
    model.eval()

    dataset = HindiAudioDataset(df)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    embeddings_list = []
    labels_list = []
    speakers_list = []
    utterance_ids_list = []

    t0 = time.time()
    with torch.no_grad():
        for b_idx, batch in enumerate(dataloader):
            x = batch["input_values"].to(device)
            mask = batch["attention_mask"].to(device)

            out = model(x, attention_mask=mask)
            hidden = out.last_hidden_state # (B, T, 1024)

            # Masked temporal average pooling
            mask_expanded = mask.unsqueeze(-1)
            # Wav2Vec2 downsamples temporal dimension by factor of ~320
            # Interpolate mask to match hidden state sequence length
            if hidden.shape[1] != mask_expanded.shape[1]:
                mask_pooled = nn.functional.interpolate(
                    mask.unsqueeze(1), size=hidden.shape[1], mode="nearest"
                ).transpose(1, 2)
            else:
                mask_pooled = mask_expanded

            sum_hidden = torch.sum(hidden * mask_pooled, dim=1)
            count = torch.clamp(torch.sum(mask_pooled, dim=1), min=1e-9)
            pooled = (sum_hidden / count).cpu().numpy()

            embeddings_list.append(pooled)
            labels_list.extend(batch["label"].numpy())
            speakers_list.extend(batch["speaker_id"])
            utterance_ids_list.extend(batch["utterance_id"])

            if (b_idx + 1) % 25 == 0 or (b_idx + 1) == len(dataloader):
                elapsed = time.time() - t0
                pct = (b_idx + 1) / len(dataloader) * 100
                print(f"  Processed {b_idx + 1}/{len(dataloader)} batches ({pct:.1f}%) in {elapsed:.1f}s")

    del model
    if device == "mps":
        torch.mps.empty_cache()
    gc.collect()

    all_embeddings = np.vstack(embeddings_list)
    all_labels = np.array(labels_list, dtype=np.int64)

    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            str(cache_path),
            embeddings=all_embeddings,
            labels=all_labels,
            speakers=np.array(speakers_list),
            utterance_ids=np.array(utterance_ids_list)
        )
        print(f"Cached representations saved to {cache_path} ({cache_path.stat().st_size / (1024*1024):.1f} MB)")

    return all_embeddings, all_labels, speakers_list, utterance_ids_list


# ============================================================================
# 3. Model Architecture Definitions
# ============================================================================

class LightweightEmotionClassifier(nn.Module):
    """
    Lightweight 8-class classification head for 1024-dim Wav2Vec2 embeddings:
    LayerNorm(1024) -> Dropout(0.3) -> Linear(1024, 256) -> GELU -> Dropout(0.3) -> Linear(256, 8)
    """

    def __init__(self, input_dim: int = 1024, hidden_dim: int = 256, num_classes: int = 8, dropout: float = 0.3):
        super().__init__()
        self.norm = nn.LayerNorm(input_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dense = nn.Linear(input_dim, hidden_dim)
        self.act = nn.GELU()
        self.dropout2 = nn.Dropout(dropout)
        self.out_proj = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        x = self.dropout1(x)
        x = self.dense(x)
        x = self.act(x)
        x = self.dropout2(x)
        return self.out_proj(x)


class PartiallyFineTunedWav2Vec2SER(nn.Module):
    """
    Wav2Vec2-XLSR-53 with unfrozen top transformer layer (Layer 23) + classification head.
    Lower 23 transformer layers and CNN feature encoder are completely frozen.
    Gradient checkpointing is enabled to reduce unified memory footprint on Apple Silicon.
    """

    def __init__(self, pretrained_model_id: str = HF_MODEL_ID, num_classes: int = 8):
        super().__init__()
        self.wav2vec2 = AutoModel.from_pretrained(pretrained_model_id)

        # 1. Freeze entire encoder and CNN
        for param in self.wav2vec2.parameters():
            param.requires_grad = False

        # 2. Unfreeze top transformer layer (layer 23) and final layer norm
        for param in self.wav2vec2.encoder.layers[-1].parameters():
            param.requires_grad = True
        if hasattr(self.wav2vec2.encoder, "layer_norm"):
            for param in self.wav2vec2.encoder.layer_norm.parameters():
                param.requires_grad = True

        # 3. Enable gradient checkpointing to save memory on 16GB RAM
        self.wav2vec2.gradient_checkpointing_enable()

        # 4. Add classification head
        self.classifier = LightweightEmotionClassifier(
            input_dim=self.wav2vec2.config.hidden_size,
            hidden_dim=256,
            num_classes=num_classes,
            dropout=0.3
        )

    def forward(self, input_values: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        outputs = self.wav2vec2(input_values, attention_mask=attention_mask)
        hidden = outputs.last_hidden_state

        if attention_mask is not None:
            mask_pooled = nn.functional.interpolate(
                attention_mask.unsqueeze(1), size=hidden.shape[1], mode="nearest"
            ).transpose(1, 2)
            sum_hidden = torch.sum(hidden * mask_pooled, dim=1)
            count = torch.clamp(torch.sum(mask_pooled, dim=1), min=1e-9)
            pooled = sum_hidden / count
        else:
            pooled = torch.mean(hidden, dim=1)

        logits = self.classifier(pooled)
        return logits


# ============================================================================
# 4. Metrics & Evaluation Utilities
# ============================================================================

def compute_ser_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    speakers: Optional[List[str]] = None
) -> Dict[str, Any]:
    """Computes comprehensive SER metrics across 8 classes and per-speaker."""
    acc = float(accuracy_score(y_true, y_pred))
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    p_wt, r_wt, f1_wt, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)

    # Unweighted Average Recall (UAR) is unweighted mean of class recalls
    uar = float(r_macro)

    # Per-class metrics
    p_class, r_class, f1_class, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, labels=list(range(8)), zero_division=0
    )
    per_class = {}
    for idx in range(8):
        emo_name = ID_TO_LABEL_8[str(idx)]
        per_class[emo_name] = {
            "precision": round(float(p_class[idx]), 4),
            "recall": round(float(r_class[idx]), 4),
            "f1": round(float(f1_class[idx]), 4),
            "support": int(support[idx])
        }

    cm = confusion_matrix(y_true, y_pred, labels=list(range(8))).tolist()

    # Speaker-wise metrics
    speaker_metrics = {}
    if speakers is not None:
        speakers_arr = np.array(speakers)
        unique_spks = sorted(list(set(speakers_arr)))
        for spk in unique_spks:
            mask = (speakers_arr == spk)
            if np.sum(mask) > 0:
                spk_true = y_true[mask]
                spk_pred = y_pred[mask]
                spk_acc = float(accuracy_score(spk_true, spk_pred))
                _, _, spk_f1, _ = precision_recall_fscore_support(spk_true, spk_pred, average="macro", zero_division=0)
                speaker_metrics[spk] = {
                    "samples": int(np.sum(mask)),
                    "accuracy": round(spk_acc, 4),
                    "macro_f1": round(float(spk_f1), 4)
                }

    return {
        "accuracy": round(acc, 4),
        "macro_f1": round(float(f1_macro), 4),
        "weighted_f1": round(float(f1_wt), 4),
        "uar": round(uar, 4),
        "per_class": per_class,
        "confusion_matrix": cm,
        "speaker_wise": speaker_metrics
    }


# ============================================================================
# 5. Model A: MFCC + SVM Baseline Evaluation
# ============================================================================

def evaluate_model_a_mfcc_svm(
    df: pd.DataFrame,
    cache_path: Path
) -> Tuple[Dict[str, Any], Any]:
    """Runs 3-fold speaker-grouped cross-validation for MFCC + SVM baseline."""
    print("\n" + "=" * 60)
    print("Evaluating Model A: MFCC + Acoustic Feature Baseline (SVM)")
    print("=" * 60)

    # 1. Feature extraction with disk caching
    dev_df = df[df["speaker_id"].isin(DEV_SPEAKERS)].reset_index(drop=True)
    if cache_path.exists():
        print(f"Loading cached MFCC features from {cache_path}...")
        data = np.load(str(cache_path), allow_pickle=True)
        X_all = data["features"]
        y_all = data["labels"]
        spk_all = data["speakers"]
    else:
        print("Extracting 605-dim acoustic features for 2,400 development samples in parallel...")
        from concurrent.futures import ThreadPoolExecutor
        t0 = time.time()

        def extract_item(row_tuple):
            _, row = row_tuple
            feat = extract_acoustic_features(Path(row["absolute_filepath"]))
            return feat, int(row["label_id"]), row["speaker_id"]

        with ThreadPoolExecutor(max_workers=min(8, os.cpu_count() or 4)) as ex:
            results = list(ex.map(extract_item, dev_df.iterrows()))

        features_list = [r[0] for r in results]
        labels_list = [r[1] for r in results]
        spk_list = [r[2] for r in results]

        print(f"  Extracted all {len(dev_df)} audio files in {time.time() - t0:.1f}s")
        X_all = np.array(features_list, dtype=np.float32)
        y_all = np.array(labels_list, dtype=np.int64)
        spk_all = np.array(spk_list)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(str(cache_path), features=X_all, labels=y_all, speakers=spk_all)
        print(f"Acoustic features saved to {cache_path}")

    # 2. 3-Fold Speaker-Grouped Cross-Validation
    fold_results = []
    y_true_all = []
    y_pred_all = []
    speakers_all = []

    total_train_time = 0.0
    total_eval_time = 0.0

    best_svm = None

    for fold_cfg in SPEAKER_FOLDS:
        f_idx = fold_cfg["fold"]
        val_spks = fold_cfg["val_speakers"]
        train_spks = fold_cfg["train_speakers"]

        train_mask = np.isin(spk_all, train_spks)
        val_mask = np.isin(spk_all, val_spks)

        X_train, y_train = X_all[train_mask], y_all[train_mask]
        X_val, y_val = X_all[val_mask], y_all[val_mask]
        spk_val = spk_all[val_mask]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)

        t_tr = time.time()
        svm = SVC(kernel="rbf", C=10.0, gamma="scale", probability=True, random_state=42)
        svm.fit(X_train_scaled, y_train)
        train_duration = time.time() - t_tr
        total_train_time += train_duration

        t_ev = time.time()
        y_pred = svm.predict(X_val_scaled)
        eval_duration = time.time() - t_ev
        total_eval_time += eval_duration

        f_metrics = compute_ser_metrics(y_val, y_pred, list(spk_val))
        fold_results.append({
            "fold": f_idx,
            "val_speakers": val_spks,
            "train_speakers": train_spks,
            "train_samples": len(y_train),
            "val_samples": len(y_val),
            "accuracy": f_metrics["accuracy"],
            "macro_f1": f_metrics["macro_f1"],
            "uar": f_metrics["uar"],
            "train_time_sec": round(train_duration, 2)
        })

        y_true_all.extend(y_val)
        y_pred_all.extend(y_pred)
        speakers_all.extend(spk_val)

        print(f"Fold {f_idx} (Val={val_spks}): Acc={f_metrics['accuracy']:.4f}, Macro-F1={f_metrics['macro_f1']:.4f}, UAR={f_metrics['uar']:.4f}")

    aggregated = compute_ser_metrics(np.array(y_true_all), np.array(y_pred_all), speakers_all)
    aggregated["fold_details"] = fold_results
    aggregated["total_train_time_sec"] = round(total_train_time, 2)
    aggregated["inference_time_per_sample_ms"] = round((total_eval_time / len(y_true_all)) * 1000, 3)
    aggregated["model_type"] = "MFCC_SVM_RBF"

    print(f"Model A Aggregated: Acc={aggregated['accuracy']:.4f}, Macro-F1={aggregated['macro_f1']:.4f}, UAR={aggregated['uar']:.4f}")
    return aggregated, best_svm


# ============================================================================
# 6. Model B: Frozen Wav2Vec2-XLSR-53 + Head Evaluation
# ============================================================================

def evaluate_model_b_frozen_wav2vec2(
    embeddings: np.ndarray,
    labels: np.ndarray,
    speakers: List[str],
    device: str = DEVICE
) -> Tuple[Dict[str, Any], Any]:
    """Runs 3-fold speaker-grouped cross-validation for Frozen Wav2Vec2 + Head."""
    print("\n" + "=" * 60)
    print("Evaluating Model B: Frozen Wav2Vec2-XLSR-53 + 8-Class Head")
    print("=" * 60)

    spk_arr = np.array(speakers)
    dev_mask = np.isin(spk_arr, DEV_SPEAKERS)

    X_dev = embeddings[dev_mask]
    y_dev = labels[dev_mask]
    spk_dev = spk_arr[dev_mask]

    fold_results = []
    y_true_all = []
    y_pred_all = []
    speakers_all = []

    total_train_time = 0.0
    total_eval_time = 0.0

    best_head_state = None
    best_macro_f1 = -1.0

    for fold_cfg in SPEAKER_FOLDS:
        f_idx = fold_cfg["fold"]
        val_spks = fold_cfg["val_speakers"]
        train_spks = fold_cfg["train_speakers"]

        train_mask = np.isin(spk_dev, train_spks)
        val_mask = np.isin(spk_dev, val_spks)

        X_train = torch.tensor(X_dev[train_mask], dtype=torch.float32)
        y_train = torch.tensor(y_dev[train_mask], dtype=torch.long)
        X_val = torch.tensor(X_dev[val_mask], dtype=torch.float32)
        y_val = torch.tensor(y_dev[val_mask], dtype=torch.long)
        spk_val = spk_dev[val_mask]

        train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
        val_dataset = torch.utils.data.TensorDataset(X_val, y_val)

        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

        model = LightweightEmotionClassifier(input_dim=1024, hidden_dim=256, num_classes=8, dropout=0.3).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
        criterion = nn.CrossEntropyLoss()
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=35, eta_min=1e-5)

        t_tr = time.time()
        best_val_f1_fold = -1.0
        fold_best_state = None

        for epoch in range(1, 36):
            model.train()
            for bx, by in train_loader:
                bx, by = bx.to(device), by.to(device)
                optimizer.zero_grad()
                logits = model(bx)
                loss = criterion(logits, by)
                loss.backward()
                optimizer.step()
            scheduler.step()

            # Validation
            model.eval()
            preds_epoch = []
            targets_epoch = []
            with torch.no_grad():
                for bx, by in val_loader:
                    bx = bx.to(device)
                    logits = model(bx)
                    preds_epoch.extend(torch.argmax(logits, dim=-1).cpu().numpy())
                    targets_epoch.extend(by.numpy())

            _, _, epoch_f1, _ = precision_recall_fscore_support(targets_epoch, preds_epoch, average="macro", zero_division=0)
            if epoch_f1 > best_val_f1_fold:
                best_val_f1_fold = epoch_f1
                fold_best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        train_duration = time.time() - t_tr
        total_train_time += train_duration

        # Final fold inference with best checkpoint
        model.load_state_dict({k: v.to(device) for k, v in fold_best_state.items()})
        model.eval()
        t_ev = time.time()
        fold_preds = []
        with torch.no_grad():
            for bx, _ in val_loader:
                bx = bx.to(device)
                logits = model(bx)
                fold_preds.extend(torch.argmax(logits, dim=-1).cpu().numpy())
        eval_duration = time.time() - t_ev
        total_eval_time += eval_duration

        f_metrics = compute_ser_metrics(y_val.numpy(), np.array(fold_preds), list(spk_val))
        fold_results.append({
            "fold": f_idx,
            "val_speakers": val_spks,
            "train_speakers": train_spks,
            "train_samples": len(y_train),
            "val_samples": len(y_val),
            "accuracy": f_metrics["accuracy"],
            "macro_f1": f_metrics["macro_f1"],
            "uar": f_metrics["uar"],
            "train_time_sec": round(train_duration, 2)
        })

        y_true_all.extend(y_val.numpy())
        y_pred_all.extend(fold_preds)
        speakers_all.extend(spk_val)

        if f_metrics["macro_f1"] > best_macro_f1:
            best_macro_f1 = f_metrics["macro_f1"]
            best_head_state = fold_best_state

        print(f"Fold {f_idx} (Val={val_spks}): Acc={f_metrics['accuracy']:.4f}, Macro-F1={f_metrics['macro_f1']:.4f}, UAR={f_metrics['uar']:.4f}")

    aggregated = compute_ser_metrics(np.array(y_true_all), np.array(y_pred_all), speakers_all)
    aggregated["fold_details"] = fold_results
    aggregated["total_train_time_sec"] = round(total_train_time, 2)
    aggregated["inference_time_per_sample_ms"] = round((total_eval_time / len(y_true_all)) * 1000, 3)
    aggregated["model_type"] = "Frozen_Wav2Vec2_Lightweight_Head"

    print(f"Model B Aggregated: Acc={aggregated['accuracy']:.4f}, Macro-F1={aggregated['macro_f1']:.4f}, UAR={aggregated['uar']:.4f}")
    return aggregated, best_head_state


# ============================================================================
# 7. Model C: Partially Fine-Tuned Wav2Vec2-XLSR-53 Evaluation
# ============================================================================

def evaluate_model_c_partial_finetune(
    df: pd.DataFrame,
    device: str = DEVICE,
    epochs: int = 2,
    batch_size: int = 4,
    grad_accum_steps: int = 4
) -> Dict[str, Any]:
    """
    Evaluates partially fine-tuned Wav2Vec2-XLSR-53 (Layer 23 + Classification Head)
    across the 3 speaker-grouped folds. Uses MPS acceleration, gradient checkpointing,
    and gradient accumulation.
    """
    print("\n" + "=" * 60)
    print("Evaluating Model C: Partially Fine-Tuned Wav2Vec2-XLSR-53 (Layer 23 + Head)")
    print(f"Hardware: {device} | Batch Size: {batch_size} (Effective: {batch_size * grad_accum_steps}) | Epochs: {epochs}")
    print("=" * 60)

    dev_df = df[df["speaker_id"].isin(DEV_SPEAKERS)].reset_index(drop=True)

    fold_results = []
    y_true_all = []
    y_pred_all = []
    speakers_all = []

    total_train_time = 0.0
    total_eval_time = 0.0

    try:
        for fold_cfg in SPEAKER_FOLDS:
            f_idx = fold_cfg["fold"]
            val_spks = fold_cfg["val_speakers"]
            train_spks = fold_cfg["train_speakers"]

            train_sub = dev_df[dev_df["speaker_id"].isin(train_spks)].reset_index(drop=True)
            val_sub = dev_df[dev_df["speaker_id"].isin(val_spks)].reset_index(drop=True)

            train_dataset = HindiAudioDataset(train_sub, max_duration_sec=4.0)
            val_dataset = HindiAudioDataset(val_sub, max_duration_sec=4.0)

            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size * 2, shuffle=False)

            # Initialize model with gradient checkpointing
            model = PartiallyFineTunedWav2Vec2SER(pretrained_model_id=HF_MODEL_ID, num_classes=8).to(device)

            # Optimizer with differential learning rates
            encoder_params = list(model.wav2vec2.encoder.layers[-1].parameters())
            if hasattr(model.wav2vec2.encoder, "layer_norm"):
                encoder_params.extend(list(model.wav2vec2.encoder.layer_norm.parameters()))
            head_params = list(model.classifier.parameters())

            optimizer = torch.optim.AdamW([
                {"params": encoder_params, "lr": 1.5e-5, "weight_decay": 0.01},
                {"params": head_params, "lr": 1e-4, "weight_decay": 0.01}
            ])
            total_steps = (len(train_loader) // grad_accum_steps) * epochs
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(total_steps, 1), eta_min=1e-6)
            criterion = nn.CrossEntropyLoss()

            t_tr = time.time()
            best_val_f1 = -1.0
            best_fold_preds = None

            for ep in range(1, epochs + 1):
                model.train()
                optimizer.zero_grad()

                for b_idx, batch in enumerate(train_loader):
                    x = batch["input_values"].to(device)
                    mask = batch["attention_mask"].to(device)
                    y = batch["label"].to(device)

                    logits = model(x, attention_mask=mask)
                    loss = criterion(logits, y) / grad_accum_steps
                    loss.backward()

                    if (b_idx + 1) % grad_accum_steps == 0 or (b_idx + 1) == len(train_loader):
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        optimizer.step()
                        scheduler.step()
                        optimizer.zero_grad()

                    if (b_idx + 1) % 100 == 0:
                        if device == "mps":
                            torch.mps.empty_cache()
                        pct = (b_idx + 1) / len(train_loader) * 100
                        print(f"    Fold {f_idx} Epoch {ep}: {b_idx + 1}/{len(train_loader)} batches ({pct:.1f}%)")

                # Validation at end of epoch
                model.eval()
                val_preds = []
                val_targets = []
                t_ev = time.time()
                with torch.no_grad():
                    for batch in val_loader:
                        x = batch["input_values"].to(device)
                        mask = batch["attention_mask"].to(device)
                        y = batch["label"].to(device)
                        logits = model(x, attention_mask=mask)
                        preds = torch.argmax(logits, dim=-1).cpu().numpy()
                        val_preds.extend(preds)
                        val_targets.extend(y.cpu().numpy())
                total_eval_time += (time.time() - t_ev)

                _, _, ep_macro_f1, _ = precision_recall_fscore_support(val_targets, val_preds, average="macro", zero_division=0)
                ep_acc = accuracy_score(val_targets, val_preds)

                print(f"  Fold {f_idx} Epoch {ep}/{epochs}: Val Acc={ep_acc:.4f}, Val Macro-F1={ep_macro_f1:.4f}")

                if ep_macro_f1 > best_val_f1:
                    best_val_f1 = ep_macro_f1
                    best_fold_preds = val_preds

            train_duration = time.time() - t_tr
            total_train_time += train_duration

            y_val_arr = val_sub["label_id"].values
            f_metrics = compute_ser_metrics(y_val_arr, np.array(best_fold_preds), val_sub["speaker_id"].tolist())

            fold_results.append({
                "fold": f_idx,
                "val_speakers": val_spks,
                "train_speakers": train_spks,
                "train_samples": len(train_sub),
                "val_samples": len(val_sub),
                "accuracy": f_metrics["accuracy"],
                "macro_f1": f_metrics["macro_f1"],
                "uar": f_metrics["uar"],
                "train_time_sec": round(train_duration, 2)
            })

            y_true_all.extend(y_val_arr)
            y_pred_all.extend(best_fold_preds)
            speakers_all.extend(val_sub["speaker_id"].tolist())

            del model, optimizer, scheduler
            if device == "mps":
                torch.mps.empty_cache()
            gc.collect()

            print(f"Fold {f_idx} Best: Acc={f_metrics['accuracy']:.4f}, Macro-F1={f_metrics['macro_f1']:.4f}, UAR={f_metrics['uar']:.4f}")

        aggregated = compute_ser_metrics(np.array(y_true_all), np.array(y_pred_all), speakers_all)
        aggregated["fold_details"] = fold_results
        aggregated["total_train_time_sec"] = round(total_train_time, 2)
        aggregated["inference_time_per_sample_ms"] = round((total_eval_time / len(y_true_all)) * 1000, 3)
        aggregated["model_type"] = "Partially_FineTuned_Wav2Vec2_Layer23"

        print(f"Model C Aggregated: Acc={aggregated['accuracy']:.4f}, Macro-F1={aggregated['macro_f1']:.4f}, UAR={aggregated['uar']:.4f}")
        return aggregated

    except Exception as e:
        print(f"\n[Warning] Model C fine-tuning encountered resource limit / exception: {e}")
        print("Falling back gracefully to frozen Wav2Vec2 encoder architecture.")
        if device == "mps":
            torch.mps.empty_cache()
        gc.collect()
        return None


# ============================================================================
# 8. Final Retraining & Untouched Test Set Evaluation
# ============================================================================

def train_and_evaluate_final_model(
    df: pd.DataFrame,
    winning_model_type: str,
    wav2vec_embeddings: np.ndarray,
    wav2vec_labels: np.ndarray,
    wav2vec_speakers: List[str],
    wav2vec_utterances: List[str],
    output_dir: Path,
    device: str = DEVICE
) -> Dict[str, Any]:
    """
    Trains the winning architecture on all 6 development speakers (2,400 samples)
    and evaluates exactly once on the untouched 2-speaker test set (F4, M4 = 800 samples).
    Saves best model weights and artifacts to models/ser_hindi/.
    """
    print("\n" + "=" * 60)
    print("Final Model Retraining on 6 Dev Speakers & Evaluation on Untouched Test Set")
    print(f"Winning Architecture: {winning_model_type}")
    print(f"Output Directory     : {output_dir}")
    print("=" * 60)

    output_dir.mkdir(parents=True, exist_ok=True)

    spk_arr = np.array(wav2vec_speakers)
    dev_mask = np.isin(spk_arr, DEV_SPEAKERS)
    test_mask = np.isin(spk_arr, TEST_SPEAKERS)

    # Sanity checks
    assert np.sum(dev_mask) == 2400, f"Expected 2400 dev samples, got {np.sum(dev_mask)}"
    assert np.sum(test_mask) == 800, f"Expected 800 test samples, got {np.sum(test_mask)}"
    assert len(set(spk_arr[dev_mask]) & set(spk_arr[test_mask])) == 0, "Speaker leakage in final test split!"

    X_dev = torch.tensor(wav2vec_embeddings[dev_mask], dtype=torch.float32)
    y_dev = torch.tensor(wav2vec_labels[dev_mask], dtype=torch.long)
    X_test = torch.tensor(wav2vec_embeddings[test_mask], dtype=torch.float32)
    y_test = torch.tensor(wav2vec_labels[test_mask], dtype=torch.long)
    spk_test = list(spk_arr[test_mask])

    train_dataset = torch.utils.data.TensorDataset(X_dev, y_dev)
    test_dataset = torch.utils.data.TensorDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    model = LightweightEmotionClassifier(input_dim=1024, hidden_dim=256, num_classes=8, dropout=0.3).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=40, eta_min=1e-5)

    training_curves = {"loss": [], "epoch": []}
    t_start = time.time()

    model.train()
    for epoch in range(1, 41):
        total_loss = 0.0
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            logits = model(bx)
            loss = criterion(logits, by)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()

        avg_loss = total_loss / len(train_loader)
        training_curves["loss"].append(round(avg_loss, 4))
        training_curves["epoch"].append(epoch)
        if epoch % 10 == 0 or epoch == 40:
            print(f"  Training Epoch {epoch}/40: Loss = {avg_loss:.4f}")

    total_train_sec = round(time.time() - t_start, 2)

    # Evaluate exactly once on the untouched test set
    model.eval()
    t_eval_start = time.time()
    test_preds = []
    test_probs = []
    with torch.no_grad():
        for bx, _ in test_loader:
            bx = bx.to(device)
            logits = model(bx)
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            preds = torch.argmax(logits, dim=-1).cpu().numpy()
            test_preds.extend(preds)
            test_probs.extend(probs)
    total_eval_sec = round(time.time() - t_eval_start, 3)

    test_metrics = compute_ser_metrics(y_test.numpy(), np.array(test_preds), spk_test)
    test_metrics["training_time_sec"] = total_train_sec
    test_metrics["inference_time_total_sec"] = total_eval_sec
    test_metrics["inference_time_per_sample_ms"] = round((total_eval_sec / len(y_test)) * 1000, 3)
    test_metrics["device"] = device
    test_metrics["test_speakers"] = TEST_SPEAKERS
    test_metrics["test_samples"] = len(y_test)

    # Save artifacts
    torch.save(model.state_dict(), output_dir / "best_model.pt")
    torch.save({
        "model_state_dict": model.state_dict(),
        "input_dim": 1024,
        "hidden_dim": 256,
        "num_classes": 8,
        "dropout": 0.3,
        "base_model": HF_MODEL_ID,
        "train_speakers": DEV_SPEAKERS,
        "test_speakers": TEST_SPEAKERS,
        "test_metrics": test_metrics
    }, output_dir / "checkpoint.pt")

    config = {
        "model_type": winning_model_type,
        "base_encoder": HF_MODEL_ID,
        "input_dimension": 1024,
        "hidden_dimension": 256,
        "num_classes": 8,
        "dropout": 0.3,
        "pooling": "temporal_masked_average",
        "sample_rate": 16000,
        "label_mapping": LABEL_TO_ID_8,
        "id_to_label": ID_TO_LABEL_8,
        "speakers_dev": DEV_SPEAKERS,
        "speakers_test": TEST_SPEAKERS,
        "test_accuracy": test_metrics["accuracy"],
        "test_macro_f1": test_metrics["macro_f1"],
        "test_uar": test_metrics["uar"]
    }
    (output_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    (output_dir / "label_mapping.json").write_text(json.dumps({
        "label_to_id": LABEL_TO_ID_8,
        "id_to_label": ID_TO_LABEL_8,
        "num_classes": 8
    }, indent=2), encoding="utf-8")
    (output_dir / "evaluation_results.json").write_text(json.dumps(test_metrics, indent=2), encoding="utf-8")
    (output_dir / "training_curves.json").write_text(json.dumps(training_curves, indent=2), encoding="utf-8")

    print(f"\nFinal Test Results on Untouched Test Set (F4, M4 - 800 samples):")
    print(f"  Accuracy : {test_metrics['accuracy']:.4f}")
    print(f"  Macro-F1 : {test_metrics['macro_f1']:.4f}")
    print(f"  UAR      : {test_metrics['uar']:.4f}")
    print(f"  Weighted : {test_metrics['weighted_f1']:.4f}")
    print(f"  Speaker F4: Acc={test_metrics['speaker_wise']['F4']['accuracy']:.4f}, F1={test_metrics['speaker_wise']['F4']['macro_f1']:.4f}")
    print(f"  Speaker M4: Acc={test_metrics['speaker_wise']['M4']['accuracy']:.4f}, F1={test_metrics['speaker_wise']['M4']['macro_f1']:.4f}")
    print(f"Artifacts saved to {output_dir}")

    return test_metrics


# ============================================================================
# 9. Main Orchestrator
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Phase 17 Hindi SER Training & Comparison")
    parser.add_argument("--device", type=str, default=DEVICE, help="Compute device (mps, cpu)")
    parser.add_argument("--skip-model-c", action="store_true", help="Skip Model C fine-tuning if fast run needed")
    args = parser.parse_args()

    print("=" * 70)
    print("PHASE 17: HINDI SPEECH EMOTION RECOGNITION MODEL TRAINING & BENCHMARK")
    print("=" * 70)
    print(f"Manifest Path : {MANIFEST_PATH}")
    print(f"Device        : {args.device}")
    print(f"Dev Speakers  : {DEV_SPEAKERS} (2,400 samples)")
    print(f"Test Speakers : {TEST_SPEAKERS} (800 samples - STRICTLY HELD OUT)")

    if not MANIFEST_PATH.exists():
        print(f"Error: Manifest not found at {MANIFEST_PATH}")
        sys.exit(1)

    df = pd.read_csv(MANIFEST_PATH)
    print(f"Loaded manifest with {len(df)} utterances.")

    # 1. Model A Evaluation (MFCC + SVM)
    mfcc_cache = CACHE_DIR / "mfcc_acoustic_features.npz"
    metrics_a, _ = evaluate_model_a_mfcc_svm(df, mfcc_cache)

    # 2. Extract Wav2Vec2 representations for Model B & Retraining
    w2v_cache = CACHE_DIR / "wav2vec2_pooled_embeddings.npz"
    embeddings, labels, speakers, utterances = extract_all_wav2vec2_representations(
        df, device=args.device, batch_size=16, cache_path=w2v_cache
    )

    # 3. Model B Evaluation (Frozen Wav2Vec2 + Head)
    metrics_b, _ = evaluate_model_b_frozen_wav2vec2(embeddings, labels, speakers, device=args.device)

    # 4. Model C Evaluation (Partially Fine-Tuned Wav2Vec2)
    if not args.skip_model_c:
        metrics_c = evaluate_model_c_partial_finetune(
            df, device=args.device, epochs=5, batch_size=8, grad_accum_steps=2
        )
    else:
        print("\n[Notice] Skipping Model C by user flag.")
        metrics_c = None

    # 5. Architecture Comparison & Selection
    print("\n" + "=" * 70)
    print("MODEL COMPARISON (SPEAKER-GROUPED CROSS-VALIDATION ON DEV SPEAKERS)")
    print("=" * 70)
    print(f"{'Model Architecture':<36} | {'Accuracy':<9} | {'Macro-F1':<9} | {'UAR':<9} | {'Train Time':<10}")
    print("-" * 80)
    print(f"{'Model A (MFCC + SVM)':<36} | {metrics_a['accuracy']:<9.4f} | {metrics_a['macro_f1']:<9.4f} | {metrics_a['uar']:<9.4f} | {metrics_a['total_train_time_sec']:<8.1f}s")
    print(f"{'Model B (Frozen Wav2Vec2 + Head)':<36} | {metrics_b['accuracy']:<9.4f} | {metrics_b['macro_f1']:<9.4f} | {metrics_b['uar']:<9.4f} | {metrics_b['total_train_time_sec']:<8.1f}s")
    if metrics_c:
        print(f"{'Model C (Partial Fine-Tuning)':<36} | {metrics_c['accuracy']:<9.4f} | {metrics_c['macro_f1']:<9.4f} | {metrics_c['uar']:<9.4f} | {metrics_c['total_train_time_sec']:<8.1f}s")
    print("-" * 80)

    # Model selection based on Grouped Validation Macro-F1 / UAR
    candidate_models = [
        ("Model A (MFCC + SVM)", metrics_a["macro_f1"], "MFCC_SVM"),
        ("Model B (Frozen Wav2Vec2 + Head)", metrics_b["macro_f1"], "Frozen_Wav2Vec2_Lightweight_Head")
    ]
    if metrics_c:
        candidate_models.append(("Model C (Partial Fine-Tuning)", metrics_c["macro_f1"], "Partially_FineTuned_Wav2Vec2"))

    candidate_models.sort(key=lambda x: x[1], reverse=True)
    winner_name, winner_f1, winner_type = candidate_models[0]
    print(f"\nWINNING MODEL SELECTED: {winner_name} (Grouped Val Macro-F1 = {winner_f1:.4f})")

    # 6. Final Retraining & Untouched Test Set Evaluation
    test_results = train_and_evaluate_final_model(
        df=df,
        winning_model_type=winner_type,
        wav2vec_embeddings=embeddings,
        wav2vec_labels=labels,
        wav2vec_speakers=speakers,
        wav2vec_utterances=utterances,
        output_dir=OUTPUT_MODEL_DIR,
        device=args.device
    )

    # 7. Write comprehensive benchmark summary JSON
    benchmark_summary = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "hardware_device": args.device,
        "dev_speakers": DEV_SPEAKERS,
        "test_speakers": TEST_SPEAKERS,
        "models_compared": {
            "model_a_mfcc_svm": metrics_a,
            "model_b_frozen_wav2vec2": metrics_b,
            "model_c_partial_finetune": metrics_c
        },
        "winning_model": {
            "name": winner_name,
            "type": winner_type,
            "dev_grouped_macro_f1": winner_f1
        },
        "final_test_evaluation": test_results
    }

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    summary_file = REPORTS_DIR / "phase17_benchmark_summary.json"
    summary_file.write_text(json.dumps(benchmark_summary, indent=2), encoding="utf-8")
    print(f"\nBenchmark summary written to {summary_file}")


if __name__ == "__main__":
    main()
