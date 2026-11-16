"""
IEMOCAP Speech Emotion Recognition Evaluation Pipeline (Phase 11)
=================================================================
Evaluates the final selected model checkpoint strictly on the untouched
Session 5 test set (1,241 samples, speakers Ses05_F and Ses05_M).

Calculates:
  - Accuracy
  - Macro & Weighted Precision, Recall, F1
  - Per-class Precision, Recall, F1
  - 4x4 Confusion Matrix
  - Full Sklearn Classification Report
Saves all evaluation metrics to models/ser_iemocap/test_evaluation_results.json
"""

import sys
import json
import argparse
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.models.ser_iemocap import (
    IEMOCAPEmotionClassifier,
    LABEL_LIST,
    LABEL_TO_ID,
    ID_TO_LABEL
)
from scripts.train_ser import IEMOCAPDataset, collate_fn


def evaluate_test_set(
    model_dir: Path,
    manifest_csv: Path,
    device: str = "mps" if torch.backends.mps.is_available() else "cpu",
    batch_size: int = 16
) -> Dict[str, Any]:
    print("=" * 70)
    print("PHASE 11: UNTOUCHED SESSION 5 TEST SET EVALUATION")
    print(f"Model Directory : {model_dir}")
    print(f"Device          : {device}")
    print("=" * 70)

    # 1. Load Model
    model = IEMOCAPEmotionClassifier.from_pretrained(model_dir, device=device)
    model.eval()

    # 2. Load Manifest & Filter Test Set (Session 5)
    df = pd.read_csv(manifest_csv)
    df_4class = df[df["label_4class_merged"].notnull()].copy()
    test_df = df_4class[df_4class["standard_split"] == "test"].reset_index(drop=True)

    test_speakers = sorted(test_df["speaker_id"].unique().tolist())
    test_sessions = sorted(test_df["session"].unique().tolist())
    print(f"Test Set Specifications:")
    print(f"  Sessions   : {test_sessions} (Untouched Session 5)")
    print(f"  Speakers   : {test_speakers} (Zero overlap with train/val)")
    print(f"  Sample Count: {len(test_df):,} utterances\n")

    test_dataset = IEMOCAPDataset(df_4class, split="test", augment=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    # 3. Inference Loop
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for batch in test_loader:
            inputs = batch["input_values"].to(device)
            mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_values=inputs, attention_mask=mask)
            logits = outputs["logits"]
            probs = outputs["probabilities"]

            preds = torch.argmax(logits, dim=-1)
            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())
            all_probs.extend(probs.cpu().numpy().tolist())

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)

    # 4. Compute Metrics
    acc = accuracy_score(y_true, y_pred)
    macro_prec, macro_rec, macro_f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    weighted_prec, weighted_rec, weighted_f1, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)

    # Per-Class Metrics
    per_class_prec, per_class_rec, per_class_f1, per_class_support = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1, 2, 3], average=None, zero_division=0
    )

    per_class_metrics = {}
    for i, name in enumerate(LABEL_LIST):
        per_class_metrics[name] = {
            "precision": round(float(per_class_prec[i]), 4),
            "recall": round(float(per_class_rec[i]), 4),
            "f1_score": round(float(per_class_f1[i]), 4),
            "support": int(per_class_support[i])
        }

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3])
    cm_dict = {
        LABEL_LIST[i]: {LABEL_LIST[j]: int(cm[i, j]) for j in range(4)}
        for i in range(4)
    }

    report_str = classification_report(y_true, y_pred, target_names=LABEL_LIST, digits=4)
    print("CLASSIFICATION REPORT (SESSION 5 TEST SET):")
    print("-" * 70)
    print(report_str)

    print("CONFUSION MATRIX:")
    print("-" * 70)
    print(f"{'Actual / Pred':<15} | {'Angry':<8} | {'Happy':<8} | {'Sad':<8} | {'Neutral':<8}")
    print("-" * 60)
    for i, row_name in enumerate(LABEL_LIST):
        print(f"{row_name:<15} | {cm[i, 0]:<8} | {cm[i, 1]:<8} | {cm[i, 2]:<8} | {cm[i, 3]:<8}")

    results = {
        "evaluation_split": "test (Session 5 untouched)",
        "test_sample_count": len(test_df),
        "test_sessions": test_sessions,
        "test_speakers": test_speakers,
        "overall_metrics": {
            "accuracy": round(float(acc), 4),
            "macro_precision": round(float(macro_prec), 4),
            "macro_recall": round(float(macro_rec), 4),
            "macro_f1": round(float(macro_f1), 4),
            "weighted_f1": round(float(weighted_f1), 4)
        },
        "per_class_metrics": per_class_metrics,
        "confusion_matrix": cm_dict,
        "raw_confusion_matrix": cm.tolist()
    }

    # Save Results
    results_path = model_dir / "test_evaluation_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved test evaluation metrics to: {results_path}")
    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate 4-class IEMOCAP SER Model (Phase 11)")
    parser.add_argument("--model_dir", type=str, default="models/ser_iemocap", help="Model checkpoint directory")
    parser.add_argument("--device", type=str, default="mps" if torch.backends.mps.is_available() else "cpu")
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()

    model_dir = PROJECT_ROOT / args.model_dir
    manifest_csv = PROJECT_ROOT / "data/iemocap/iemocap_clean_manifest.csv"

    evaluate_test_set(
        model_dir=model_dir,
        manifest_csv=manifest_csv,
        device=args.device,
        batch_size=args.batch_size
    )


if __name__ == "__main__":
    main()

