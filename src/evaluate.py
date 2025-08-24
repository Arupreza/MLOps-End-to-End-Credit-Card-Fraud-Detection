#!/usr/bin/env python3
"""
Evaluation script for CreditCardGRU with full MLflow logging, feature-schema alignment,
BatchNorm/LayerNorm checkpoint compatibility, and positive-class auto-detection.

Run:
    python src/evaluate.py

Auto-discovers:
    - CSV:   Data/processed/creditcard_processed_test.csv (fallback: _val, _train)
    - Model: models/best_fraud_detection_model.pth (fallback: first .pth in models/)
    - Meta:  models/model_info.json (fallback: optuna_study/study_results.json)

Artifacts saved to reports/eval/<timestamp>/ and logged to MLflow (MLFLOW_TRACKING_URI or ./mlruns).
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


# =========================
# Model (BN/LN flexible)
# =========================
class CreditCardGRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout=0.5, norm_type="layer"):
        """
        norm_type: "batch" | "layer" | "none"
        """
        super(CreditCardGRU, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.norm_type = norm_type

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

        # Instantiate normalization modules with names matching checkpoints
        if norm_type == "batch":
            self.bn1 = nn.BatchNorm1d(hidden_size // 2)
            self.bn2 = nn.BatchNorm1d(hidden_size // 4)
        elif norm_type == "layer":
            self.ln1 = nn.LayerNorm(hidden_size // 2)
            self.ln2 = nn.LayerNorm(hidden_size // 4)
        else:
            self.n1 = nn.Identity()
            self.n2 = nn.Identity()

    def forward(self, x):
        # x: (N, seq_len, input_size); seq_len=1 in this pipeline
        batch_size = x.size(0)
        x = self.input_projection(x)
        x = self.relu(x)

        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=x.device)
        out, _ = self.gru(x, h0)
        last = out[:, -1, :]

        z = self.fc1(last)
        if self.norm_type == "batch":
            z = self.bn1(z)
        elif self.norm_type == "layer":
            z = self.ln1(z)
        else:
            z = self.n1(z)
        z = self.relu(z)
        z = self.dropout(z)

        z = self.fc2(z)
        if self.norm_type == "batch":
            z = self.bn2(z)
        elif self.norm_type == "layer":
            z = self.ln2(z)
        else:
            z = self.n2(z)
        z = self.relu(z)
        z = self.dropout(z)

        z = self.fc3(z)  # logits
        return z


# =========================
# Confusion matrix (requested style)
# =========================
def plot_confusion_matrix_with_percentages_bin(y_true, y_pred, labels, save_path=None, show=False):
    """
    Draw a confusion matrix heatmap with BOTH raw counts and row-wise percentages.
    Colors specific tick labels if they exist:
      - "Normal Transection" -> green
      - "Suspicion  Transection" -> red  (note the double space, as requested)
    """
    cm = confusion_matrix(y_true, y_pred)

    # Row-wise percentages (% of actual class predicted as each class)
    row_sums = cm.sum(axis=1, keepdims=True)
    # Avoid division by zero
    row_sums[row_sums == 0] = 1
    cm_percentages = cm / row_sums * 100.0

    fig, ax = plt.subplots(figsize=(8, 7))

    # Base heatmap (no annotations from seaborn; we place our own)
    sns.heatmap(cm, annot=False, cmap="Pastel2", cbar=False, ax=ax)

    # Overlay counts + percentages
    n_rows, n_cols = cm.shape
    for i in range(n_rows):
        for j in range(n_cols):
            ax.text(
                j + 0.5, i + 0.50,
                f"{cm[i, j]}",
                ha="center", va="center",
                fontsize=25, fontweight="bold", color="black",
            )
            ax.text(
                j + 0.5, i + 0.80,
                f"{cm_percentages[i, j]:.2f}%",
                ha="center", va="center",
                fontsize=18, fontweight="bold", color="black",
            )

    ax.set_xlabel("Predicted Class", fontweight="bold", fontsize=19, color="darkblue")
    ax.set_ylabel("Actual Class", fontweight="bold", fontsize=19, color="darkblue")
    ax.set_title("Confusion Matrix", fontsize=16, fontweight="bold")

    # Ticks in the middle of each cell
    ax.set_xticks(np.arange(len(labels)) + 0.5)
    ax.set_yticks(np.arange(len(labels)) + 0.5)
    ax.set_xticklabels(labels, fontweight="bold", fontsize=16, rotation=90)
    ax.set_yticklabels(labels, fontweight="bold", fontsize=16, rotation=0)

    # Optional coloring of specific labels, if present
    try:
        xt = ax.get_xticklabels()
        yt = ax.get_yticklabels()
        if "Normal Transection" in labels:
            idx = labels.index("Normal Transection")
            xt[idx].set_color("Green")
            yt[idx].set_color("Green")
        if "Suspicion  Transection" in labels:  # note the two spaces
            idx = labels.index("Suspicion  Transection")
            xt[idx].set_color("Red")
            yt[idx].set_color("Red")
    except Exception:
        pass

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
    elif show:
        plt.show()


# =========================
# Helpers
# =========================
def _first_existing(*paths):
    for p in paths:
        if p and os.path.exists(p):
            return p
    return None


def resolve_defaults():
    """Find test CSV, model checkpoint, and meta/study files from your layout."""
    test_csv = _first_existing(
        "Data/processed/creditcard_processed_test.csv",
        "Data/processed/creditcard_processed_val.csv",
        "Data/processed/creditcard_processed_train.csv",
    )
    if not test_csv:
        raise FileNotFoundError("Could not find Data/processed/creditcard_processed_{test,val,train}.csv")

    model_path = _first_existing("models/best_fraud_detection_model.pth")
    if not model_path and os.path.isdir("models"):
        for name in os.listdir("models"):
            if name.endswith(".pth"):
                model_path = os.path.join("models", name)
                break
    if not model_path:
        raise FileNotFoundError("No model .pth found under models/")

    meta_json = _first_existing(
        "models/model_info.json",
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
        "experiment_name": ["experiment_name", "exp_name"],
        "mlflow_run_id": ["mlflow_run_id", "run_id"]
    }
    out = {}
    for k, alts in keymap.items():
        for a in alts:
            if a in hp:
                out[k] = hp[a]
                break
        # also allow top-level keys
        if k not in out and k in meta:
            out[k] = meta[k]

    # Bubble through optional schema info
    if "feature_names" in meta:
        out["feature_names"] = meta["feature_names"]
    if "label_col" in meta:
        out["label_col"] = meta["label_col"]

    return out


def load_best_hparams(meta_json, study_json):
    """
    Load best hyperparameters.
    Priority:
        1) models/model_info.json
        2) optuna_study/study_results.json
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
                "source_run_id": out.get("mlflow_run_id"),
                "feature_names": out.get("feature_names"),
                "label_col": out.get("label_col")
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
                "source_run_id": None,
                "feature_names": None,
                "label_col": None
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


def _coerce_state_dict(obj):
    """
    Accept common checkpoint formats:
        - pure state_dict (key->tensor)
        - {'state_dict': ...} or {'model_state_dict': ...}
        - full nn.Module saved -> extract .state_dict()
    """
    if isinstance(obj, nn.Module):
        return obj.state_dict()
    if isinstance(obj, dict):
        if "state_dict" in obj and isinstance(obj["state_dict"], dict):
            return obj["state_dict"]
        if "model_state_dict" in obj and isinstance(obj["model_state_dict"], dict):
            return obj["model_state_dict"]
        # Heuristic: treat as state_dict if values look like tensors
        if all(isinstance(v, (torch.Tensor, np.ndarray, float, int, bool)) or hasattr(v, "shape") for v in obj.values()):
            return obj
    raise ValueError("Unrecognized checkpoint format; expected a state_dict or a dict with 'state_dict'.")


def _detect_norm_type_from_state(state_dict):
    keys = list(state_dict.keys())
    if any(k.startswith("bn1.") or k.startswith("bn2.") for k in keys):
        return "batch"
    if any(k.startswith("ln1.") or k.startswith("ln2.") for k in keys):
        return "layer"
    return "none"


# =========================
# Main Evaluation
# =========================
def main():
    # Configure MLflow (use env URI or local)
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "file:./mlruns")
    mlflow.set_tracking_uri(tracking_uri)

    # Discover inputs / hparams
    test_csv, model_path, meta_json, study_json = resolve_defaults()
    hp = load_best_hparams(meta_json, study_json)
    exp_name = hp.get("experiment_name") or "fraud_detection_evaluation"
    mlflow.set_experiment(exp_name)

    out_dir = ensure_out_dir()

    # --- Data ---
    df = pd.read_csv(test_csv)

    # Use saved label name if provided
    label_col = hp.get("label_col") or ("Class" if "Class" in df.columns else df.columns[-1])

    # Align features to training order if available
    feat_names = hp.get("feature_names")
    diag = {}  # diagnostics we will save later

    if feat_names:
        missing = [c for c in feat_names if c not in df.columns]
        extra = [c for c in df.columns if c not in (feat_names + [label_col])]
        diag["feature_mismatch"] = {"missing": missing, "extra": extra}
        if missing:
            raise ValueError(
                f"Missing features in test CSV: {missing}\n"
                f"Extra columns present: {extra}\n"
                f"Expected order (first 10): {feat_names[:10]}"
            )
        X = df[feat_names].values.astype(np.float32)
    else:
        # Fallback: best-effort (may cause silent degradation if order differs)
        X = df.drop(columns=[label_col]).values.astype(np.float32)

    y_true = df[label_col].astype(int).values
    input_size = X.shape[1]

    X_tensor = torch.from_numpy(X).unsqueeze(1)  # (N, 1, input_size)
    y_tensor = torch.from_numpy(y_true).long()
    loader = DataLoader(TensorDataset(X_tensor, y_tensor), batch_size=2048, shuffle=False, num_workers=0)

    # Checkpoint
    raw_obj = torch.load(model_path, map_location="cpu")
    state = _coerce_state_dict(raw_obj)
    norm_type = _detect_norm_type_from_state(state)

    # Model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CreditCardGRU(
        input_size=input_size,
        hidden_size=hp["hidden_size"],
        num_layers=hp["num_layers"],
        num_classes=hp["num_classes"],
        dropout=hp["dropout"],
        norm_type=norm_type
    ).to(device)

    # Load weights strictly (keys match because we created bn*/ln* accordingly)
    model.load_state_dict(state, strict=True)
    model.eval()

    # Inference
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

    # --- Auto-detect which column is "positive" (higher AUC) ---
    pos_index = 1
    roc_ok = hp["num_classes"] == 2 and len(np.unique(y_true)) == 2
    auc_col1 = auc_col0 = None
    if roc_ok:
        try:
            auc_col1 = float(roc_auc_score(y_true, probs[:, 1]))
        except Exception:
            pass
        try:
            auc_col0 = float(roc_auc_score(y_true, probs[:, 0]))
        except Exception:
            pass
        if auc_col0 is not None and (auc_col1 is None or auc_col0 > auc_col1):
            pos_index = 0

    pos_proba = probs[:, pos_index] if hp["num_classes"] == 2 else None
    diag["auc_col0"] = auc_col0
    diag["auc_col1"] = auc_col1
    diag["chosen_pos_index"] = pos_index

    # Metrics at argmax
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

    if hp["num_classes"] == 2 and len(np.unique(y_true)) == 2 and pos_proba is not None:
        try:
            metrics_argmax["roc_auc"] = float(roc_auc_score(y_true, pos_proba))
        except Exception:
            pass
        try:
            metrics_argmax["average_precision"] = float(average_precision_score(y_true, pos_proba))
        except Exception:
            pass

    cls_report = classification_report(y_true, y_pred, digits=4, zero_division=0)

    # Threshold sweep (binary) to maximize F1
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
        sweep_df.to_csv(os.path.join(out_dir, "threshold_sweep.csv"), index=False)

    # Save local artifacts
    bundle = {
        "dataset": os.path.abspath(test_csv),
        "model_checkpoint": os.path.abspath(model_path),
        "device": device,
        "input_size": int(input_size),
        "norm_type": norm_type,
        "hparams": {
            "hidden_size": int(hp["hidden_size"]),
            "num_layers": int(hp["num_layers"]),
            "dropout": float(hp["dropout"]),
            "num_classes": int(hp["num_classes"]),
            "label_col": label_col,
            "feature_names_used": feat_names if feat_names else "auto(drop-cols)"
        },
        "diagnostics": diag,
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

    # --- Confusion Matrix (requested custom heatmap) ---
    cm_labels = (
        ["Normal", "Suspicion"]  # keep your exact spellings
        if hp["num_classes"] == 2
        else [str(i) for i in range(hp["num_classes"])]
    )
    cm_path = os.path.join(out_dir, "confusion_matrix.png")
    plot_confusion_matrix_with_percentages_bin(
        y_true=y_true, y_pred=y_pred, labels=cm_labels, save_path=cm_path
    )

    # ROC / PR (binary)
    roc_path = pr_path = None
    if hp["num_classes"] == 2 and pos_proba is not None and len(np.unique(y_true)) == 2:
        fpr, tpr, _ = roc_curve(y_true, pos_proba)
        precision, recall, _ = precision_recall_curve(y_true, pos_proba)

        plt.figure(figsize=(5, 4))
        plt.plot(fpr, tpr, lw=2)
        plt.plot([0, 1], [0, 1], linestyle="--", lw=1)
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC Curve (AUC={metrics_argmax.get('roc_auc', None)})")
        plt.tight_layout()
        roc_path = os.path.join(out_dir, "roc_curve.png")
        plt.savefig(roc_path, dpi=200)
        plt.close()

        plt.figure(figsize=(5, 4))
        plt.plot(recall, precision, lw=2)
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title(f"PR Curve (AP={metrics_argmax.get('average_precision', None)})")
        plt.tight_layout()
        pr_path = os.path.join(out_dir, "pr_curve.png")
        plt.savefig(pr_path, dpi=200)
        plt.close()

    # MLflow logging
    with mlflow.start_run(run_name=f"evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}") as run:
        if hp.get("source_run_id"):
            mlflow.set_tag("source_run_id", hp["source_run_id"])
        mlflow.set_tag("phase", "evaluation")
        mlflow.set_tag("dataset", os.path.basename(test_csv))
        mlflow.set_tag("norm_type", norm_type)

        mlflow.log_params({
            "input_size": input_size,
            "hidden_size": hp["hidden_size"],
            "num_layers": hp["num_layers"],
            "dropout": hp["dropout"],
            "num_classes": hp["num_classes"],
            "device": device,
            "checkpoint": os.path.basename(model_path)
        })

        mlflow.log_metrics(metrics_argmax)
        if metrics_thresh:
            mlflow.log_metrics({f"thresh_{k}": v for k, v in metrics_thresh.items()})

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

        # Console summary
        print("\n" + "=" * 66)
        print(" M L f l o w   E V A L U A T I O N   S U M M A R Y")
        print("=" * 66)
        print(f"Run ID:     {run.info.run_id}")
        print(f"Experiment: {mlflow.get_experiment(run.info.experiment_id).name}")
        print(f"Tracking:   {mlflow.get_tracking_uri()}")
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
        print(classification_report(y_true, y_pred, digits=4, zero_division=0))
        print("=" * 66 + "\n")


if __name__ == "__main__":
    main()
