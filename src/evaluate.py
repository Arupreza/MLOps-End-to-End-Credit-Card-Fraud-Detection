#!/usr/bin/env python3
"""
Evaluation script for CreditCardGRU with full MLflow logging.

Usage:
  python src/evaluate.py

Auto-discovers:
  - CSV:  Data/processed/creditcard_processed_test.csv (fallback: _val, _train)
  - Model: models/best_fraud_detection_model.pth (fallback: any .pth in models/)
  - HParams: models/model_info.json (fallback: optuna_study/study_results.json)

Logs to MLflow:
  - metrics (argmax metrics + optional threshold-sweep metrics)
  - confusion matrix, ROC, PR plots
  - predictions.csv, metrics.json, classification_report.txt
  - tags linking to source training run if available
"""

import os
import json
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score,
    roc_curve, average_precision_score, precision_recall_curve
)

import matplotlib.pyplot as plt
import seaborn as sns

import mlflow
from mlflow import MlflowClient


# ----------------------------
# Model (must match training)
# ----------------------------
class CreditCardGRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout=0.5):
        super(CreditCardGRU, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes

        self.input_projection = nn.Linear(input_size, hidden_size)
        nn.init.xavier_uniform_(self.input_projection.weight)

        self.gru = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False
        )
        for name, param in self.gru.named_parameters():
            if 'weight' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)

        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc2 = nn.Linear(hidden_size // 2, hidden_size // 4)
        self.fc3 = nn.Linear(hidden_size // 4, num_classes)

        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)

        self.relu = nn.ReLU()
        self.ln1 = nn.LayerNorm(hidden_size // 2)
        self.ln2 = nn.LayerNorm(hidden_size // 4)

    def forward(self, x):
        # x: (N, seq_len, input_size); seq_len=1 in this pipeline
        batch_size = x.size(0)
        x = self.input_projection(x)
        x = self.relu(x)

        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        out, _ = self.gru(x, h0)
        last = out[:, -1, :]

        z = self.fc1(last); z = self.ln1(z); z = self.relu(z); z = self.dropout(z)
        z = self.fc2(z); z = self.ln2(z); z = self.relu(z); z = self.dropout(z)
        z = self.fc3(z)  # logits
        return z


# ----------------------------
# Helpers
# ----------------------------
def _first_existing(*paths):
    for p in paths:
        if p and os.path.exists(p):
            return p
    return None


def resolve_defaults():
    """Resolve default paths from your repo layout."""
    # CSVs: prefer test → val → train
    test_csv = _first_existing(
        "Data/processed/creditcard_processed_test.csv",
        "Data/processed/creditcard_processed_val.csv",
        "Data/processed/creditcard_processed_train.csv",
    )
    if not test_csv:
        raise FileNotFoundError("Could not find any dataset at Data/processed/creditcard_processed_{test,val,train}.csv")

    # Model checkpoint
    model_path = _first_existing("models/best_fraud_detection_model.pth")
    if not model_path and os.path.isdir("models"):
        for name in os.listdir("models"):
            if name.endswith(".pth"):
                model_path = os.path.join("models", name)
                break
    if not model_path:
        raise FileNotFoundError("No model .pth found under models/. Expected models/best_fraud_detection_model.pth")

    # Metadata / hparams
    meta_json = _first_existing(
        "models/model_info.json",                         # **present in your tree**
        "models/credit_card_gru_fraud_detection_fixed_best.json",
    )
    study_json = _first_existing("optuna_study/study_results.json")

    return test_csv, model_path, meta_json, study_json


def parse_hparams_from_meta(meta):
    """Accept {'best_params': {...}}, {'params': {...}}, or flat maps."""
    hp = meta.get("best_params") or meta.get("params") or meta
    keymap = {
        "hidden_size": ["hidden_size", "hidden_dim", "hsz"],
        "num_layers":  ["num_layers", "n_layers", "layers"],
        "dropout":     ["dropout", "dropout_rate", "p_drop"],
        "num_classes": ["num_classes", "n_classes"],
        "experiment_name": ["experiment_name", "exp_name"]
    }
    out = {}
    for k, alts in keymap.items():
        for a in alts:
            if a in hp:
                out[k] = hp[a]
                break
        # allow some keys to come from the top-level meta
        if k not in out and k in meta:
            out[k] = meta[k]
    return out


def load_best_hparams(meta_json, study_json):
    """
    Load best hyperparameters.
    Priority:
        1) models/model_info.json (preferred)
        2) optuna_study/study_results.json (best_params)
    """
    if meta_json:
        with open(meta_json, "r") as f:
            meta = json.load(f)
        out = parse_hparams_from_meta(meta)
        if {"hidden_size", "num_layers", "dropout"} <= set(out.keys()):
            return {
                "hidden_size": int(out["hidden_size"]),
                "num_layers": int(out["num_layers"]),
                "dropout": float(out["dropout"]),
                "num_classes": int(out.get("num_classes", 2)),
                "experiment_name": out.get("experiment_name", "fraud_detection_evaluation"),
                "source_run_id": meta.get("mlflow_run_id") or out.get("mlflow_run_id")
            }

    if study_json:
        with open(study_json, "r") as f:
            sr = json.load(f)
        bp = sr.get("best_params")
        if bp and {"hidden_size", "num_layers", "dropout"} <= set(bp.keys()):
            return {
                "hidden_size": int(bp["hidden_size"]),
                "num_layers": int(bp["num_layers"]),
                "dropout": float(bp["dropout"]),
                "num_classes": int(bp.get("num_classes", 2)),
                "experiment_name": "fraud_detection_evaluation",
                "source_run_id": None
            }

    raise RuntimeError(
        "Could not find best hyperparameters. Ensure models/model_info.json or optuna_study/study_results.json exists."
    )


def ensure_out_dir(base="reports/eval"):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(base, ts)
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def softmax(x):
    return torch.softmax(x, dim=1)


def log_plot_to_mlflow(fig, artifact_file):
    """Log a Matplotlib figure to MLflow and also save it to the local out_dir."""
    mlflow.log_figure(fig, artifact_file)


# ----------------------------
# Evaluation core (with MLflow)
# ----------------------------
def main():
    # 0) Configure MLflow tracking
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "file:./mlruns")
    mlflow.set_tracking_uri(tracking_uri)

    # 1) Discover inputs & hyperparameters
    test_csv, model_path, meta_json, study_json = resolve_defaults()
    hp = load_best_hparams(meta_json, study_json)
    exp_name = hp.get("experiment_name") or "fraud_detection_evaluation"

    # 2) Prepare output dir
    out_dir = ensure_out_dir()

    # 3) Data
    df = pd.read_csv(test_csv)
    label_col = "Class" if "Class" in df.columns else df.columns[-1]
    y_true = df[label_col].astype(int).values
    X = df.drop(columns=[label_col]).values.astype(np.float32)
    input_size = X.shape[1]

    X_tensor = torch.from_numpy(X).unsqueeze(1)  # (N, 1, input_size)
    y_tensor = torch.from_numpy(y_true).long()
    loader = DataLoader(TensorDataset(X_tensor, y_tensor), batch_size=2048, shuffle=False, num_workers=0)

    # 4) Model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CreditCardGRU(
        input_size=input_size,
        hidden_size=hp["hidden_size"],
        num_layers=hp["num_layers"],
        num_classes=hp["num_classes"],
        dropout=hp["dropout"]
    ).to(device)

    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    # 5) Inference
    all_logits, all_preds = [], []
    with torch.no_grad():
        for xb, _ in loader:
            xb = xb.to(device)
            logits = model(xb)
            all_logits.append(logits.cpu())
            all_preds.append(logits.argmax(dim=1).cpu())

    logits = torch.cat(all_logits, dim=0)           # (N, C)
    y_pred = torch.cat(all_preds, dim=0).numpy()    # (N,)
    probs = softmax(logits).numpy()
    pos_proba = probs[:, 1] if hp["num_classes"] == 2 else None

    # 6) Metrics (argmax operating point)
    metrics_argmax = {
        "accuracy":            float(accuracy_score(y_true, y_pred)),
        "precision_weighted":  float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "recall_weighted":     float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1_weighted":         float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "precision_macro":     float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro":        float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro":            float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "support_pos":         int((y_true == 1).sum()),
        "support_neg":         int((y_true == 0).sum()),
    }

    roc_auc = ap = None
    if hp["num_classes"] == 2 and len(np.unique(y_true)) == 2 and pos_proba is not None:
        try:
            roc_auc = float(roc_auc_score(y_true, pos_proba))
        except Exception:
            roc_auc = None
        try:
            ap = float(average_precision_score(y_true, pos_proba))
        except Exception:
            ap = None
        if roc_auc is not None:
            metrics_argmax["roc_auc"] = roc_auc
        if ap is not None:
            metrics_argmax["average_precision"] = ap

    cls_report = classification_report(y_true, y_pred, digits=4, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(hp["num_classes"])))

    # 7) Optional: threshold sweep (binary) to maximize F1
    metrics_thresh = {}
    best_thresh = None
    if hp["num_classes"] == 2 and pos_proba is not None and len(np.unique(y_true)) == 2:
        thresholds = np.linspace(0.01, 0.99, 99)
        rows = []
        for t in thresholds:
            pred_t = (pos_proba >= t).astype(int)
            rows.append({
                "threshold": t,
                "f1_weighted": f1_score(y_true, pred_t, average="weighted", zero_division=0),
                "precision": precision_score(y_true, pred_t, zero_division=0),
                "recall": recall_score(y_true, pred_t, zero_division=0),
                "accuracy": accuracy_score(y_true, pred_t),
            })
        sweep_df = pd.DataFrame(rows)
        best_row = sweep_df.iloc[sweep_df["f1_weighted"].idxmax()]
        best_thresh = float(best_row["threshold"])
        metrics_thresh = {
            "best_threshold": best_thresh,
            "f1_weighted_at_best_threshold": float(best_row["f1_weighted"]),
            "precision_at_best_threshold": float(best_row["precision"]),
            "recall_at_best_threshold": float(best_row["recall"]),
            "accuracy_at_best_threshold": float(best_row["accuracy"]),
        }
        # save sweep
        sweep_path = os.path.join(out_dir, "threshold_sweep.csv")
        sweep_df.to_csv(sweep_path, index=False)

    # 8) Save local artifacts
    # metrics.json packs both argmax and threshold metrics
    bundle = {
        "dataset": os.path.abspath(test_csv),
        "model_checkpoint": os.path.abspath(model_path),
        "device": device,
        "input_size": int(input_size),
        "hparams": {
            "hidden_size": int(hp["hidden_size"]),
            "num_layers": int(hp["num_layers"]),
            "dropout": float(hp["dropout"]),
            "num_classes": int(hp["num_classes"])
        },
        "metrics_argmax": metrics_argmax,
        "metrics_threshold_sweep": metrics_thresh
    }
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(bundle, f, indent=2)

    with open(os.path.join(out_dir, "classification_report.txt"), "w") as f:
        f.write(cls_report)

    pred_df = pd.DataFrame({"y_true": y_true, "y_pred": y_pred})
    if hp["num_classes"] == 2 and pos_proba is not None:
        pred_df["prob_pos"] = pos_proba
        if best_thresh is not None:
            pred_df["y_pred@best_t"] = (pos_proba >= best_thresh).astype(int)
    pred_csv_path = os.path.join(out_dir, "predictions.csv")
    pred_df.to_csv(pred_csv_path, index=False)

    # Confusion matrix (argmax)
    fig = plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cbar=True,
                xticklabels=list(range(hp["num_classes"])),
                yticklabels=list(range(hp["num_classes"])))
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix (argmax)")
    plt.tight_layout()
    cm_path = os.path.join(out_dir, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=200)
    plt.close(fig)

    # ROC / PR (binary)
    roc_path = pr_path = None
    if hp["num_classes"] == 2 and pos_proba is not None and len(np.unique(y_true)) == 2:
        fpr, tpr, _ = roc_curve(y_true, pos_proba)
        precision, recall, _ = precision_recall_curve(y_true, pos_proba)

        fig = plt.figure(figsize=(5, 4))
        plt.plot(fpr, tpr, lw=2)
        plt.plot([0, 1], [0, 1], linestyle="--", lw=1)
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC Curve (AUC={metrics_argmax.get('roc_auc', None)})")
        plt.tight_layout()
        roc_path = os.path.join(out_dir, "roc_curve.png")
        plt.savefig(roc_path, dpi=200)
        plt.close(fig)

        fig = plt.figure(figsize=(5, 4))
        plt.plot(recall, precision, lw=2)
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title(f"PR Curve (AP={metrics_argmax.get('average_precision', None)})")
        plt.tight_layout()
        pr_path = os.path.join(out_dir, "pr_curve.png")
        plt.savefig(pr_path, dpi=200)
        plt.close(fig)

    # 9) MLflow logging
    mlflow.set_experiment(exp_name)
    with mlflow.start_run(run_name=f"evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}") as run:
        # Link back to source training run if present
        if hp.get("source_run_id"):
            mlflow.set_tag("source_run_id", hp["source_run_id"])
        mlflow.set_tag("phase", "evaluation")
        mlflow.set_tag("dataset", os.path.basename(test_csv))

        # Params
        mlflow.log_params({
            "input_size": input_size,
            "hidden_size": hp["hidden_size"],
            "num_layers": hp["num_layers"],
            "dropout": hp["dropout"],
            "num_classes": hp["num_classes"],
            "device": device,
            "checkpoint": os.path.basename(model_path)
        })

        # Metrics
        mlflow.log_metrics(metrics_argmax)
        if metrics_thresh:
            mlflow.log_metrics({f"thresh_{k}": v for k, v in metrics_thresh.items()})

        # Artifacts
        mlflow.log_artifact(os.path.join(out_dir, "metrics.json"), artifact_path="evaluation")
        mlflow.log_artifact(os.path.join(out_dir, "classification_report.txt"), artifact_path="evaluation")
        mlflow.log_artifact(pred_csv_path, artifact_path="evaluation")
        mlflow.log_artifact(cm_path, artifact_path="evaluation")
        if roc_path:
            mlflow.log_artifact(roc_path, artifact_path="evaluation")
        if pr_path:
            mlflow.log_artifact(pr_path, artifact_path="evaluation")
        if hp["num_classes"] == 2 and best_thresh is not None:
            mlflow.log_artifact(os.path.join(out_dir, "threshold_sweep.csv"), artifact_path="evaluation")

        # Print a short summary to console
        print("\n" + "=" * 66)
        print(" M L f l o w   E V A L U A T I O N   S U M M A R Y")
        print("=" * 66)
        print(f"Run ID:     {run.info.run_id}")
        print(f"Experiment: {exp_name}")
        print(f"Tracking:   {tracking_uri}")
        print(f"Dataset:    {test_csv}")
        print(f"Model:      {model_path}")
        print(f"Saved to:   {out_dir}\n")
        for k in [
            "accuracy", "precision_weighted", "recall_weighted", "f1_weighted",
            "precision_macro", "recall_macro", "f1_macro", "roc_auc", "average_precision"
        ]:
            if k in metrics_argmax:
                print(f"{k:>22}: {metrics_argmax[k]}")
        if best_thresh is not None:
            print(f"\nBest threshold by F1: {best_thresh:.3f}")
            for k in ["f1_weighted_at_best_threshold", "precision_at_best_threshold",
                    "recall_at_best_threshold", "accuracy_at_best_threshold"]:
                print(f"{k:>30}: {metrics_thresh[k]}")
        print("\nClassification Report (argmax):\n")
        print(cls_report)
        print("=" * 66 + "\n")


if __name__ == "__main__":
    main()
