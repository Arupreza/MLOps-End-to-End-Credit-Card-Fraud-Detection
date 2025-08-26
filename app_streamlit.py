# app_streamlit.py
import os
import io
import json
from typing import List, Tuple

import numpy as np
import pandas as pd
import streamlit as st

import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_recall_fscore_support
)

# =========================
# Streamlit Page
# =========================
st.set_page_config(page_title="Credit-Fraud Eval (Model)", page_icon="💳", layout="centered")
st.title("💳 Fraud Evaluation on Kaggle Dataset — Model Predictions")

st.markdown("""
**Dataset overview**  
- **Name**: Credit Card Fraud Detection Dataset 2023 (Kaggle)  
- **Records**: ~550K anonymized transactions from European cardholders  
- **Features**: `V1–V28` (anonymized), `Amount`, (often `Time`/`id`), and **`Class`** (0=normal, 1=fraud).  
""")

# =========================
# Paths & Defaults
# =========================
# Default internal path; can be overridden by env var TEST_DATA_PATH
TEST_CSV_DEFAULT = "Data/processed/creditcard_processed_test.csv"
TEST_CSV = os.getenv("TEST_DATA_PATH", TEST_CSV_DEFAULT)

INFO_JSON = "models/model_info.json"
MODEL_PTH = "models/best_fraud_detection_model.pth"

# Optional fallback list if model_info lacks features
TARGET_COLS = ['V3','V4','V9','V10','V11','V12','V14','V16','V17','V18','V21','V22','Class']

# =========================
# Sidebar Controls
# =========================
st.subheader("Settings")
with st.sidebar:
    st.header("Data Source")
    uploaded = st.file_uploader(
        "Upload a CSV file",
        type=["csv"],
        accept_multiple_files=False,
        help="If none provided, the app uses TEST_DATA_PATH or the built-in default path."
    )
    use_default_if_missing = st.toggle("Use default dataset if no upload", value=True)
    st.caption(f"Default path: `{TEST_CSV}` (override via TEST_DATA_PATH)")

percent = st.slider("Use % of data (stratified by `Class`)", 1, 100, 20, 1)
run_eval = st.button("Run Evaluation")

# =========================
# Model (from your evaluate.py)
# =========================
class CreditCardGRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout=0.5, norm_type="layer"):
        super().__init__()
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
# Confusion matrix (your style)
# =========================
def plot_confusion_matrix_with_percentages_bin(y_true, y_pred, labels):
    cm = confusion_matrix(y_true, y_pred)
    row_sums = cm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    cm_pct = cm / row_sums * 100.0

    fig, ax = plt.subplots(figsize=(10, 10))
    sns.heatmap(cm, annot=False, cmap="Pastel2", cbar=False, ax=ax)

    n_rows, n_cols = cm.shape
    for i in range(n_rows):
        for j in range(n_cols):
            ax.text(j + 0.5, i + 0.50, f"{cm[i, j]}",
                    ha="center", va="center", fontsize=25, fontweight="bold", color="black")
            ax.text(j + 0.5, i + 0.80, f"{cm_pct[i, j]:.2f}%",
                    ha="center", va="center", fontsize=18, fontweight="bold", color="black")

    ax.set_xlabel("Predicted Class", fontweight="bold", fontsize=19, color="darkblue")
    ax.set_ylabel("Actual Class",   fontweight="bold", fontsize=19, color="darkblue")
    ax.set_title("Confusion Matrix", fontsize=16, fontweight="bold")

    ax.set_xticks(np.arange(len(labels)) + 0.5)
    ax.set_yticks(np.arange(len(labels)) + 0.5)
    ax.set_xticklabels(labels, fontweight="bold", fontsize=16, rotation=90)
    ax.set_yticklabels(labels, fontweight="bold", fontsize=16, rotation=0)

    try:
        xt = ax.get_xticklabels(); yt = ax.get_yticklabels()
        if "Normal Transection" in labels:
            idx = labels.index("Normal Transection")
            xt[idx].set_color("green"); yt[idx].set_color("green")
        if "Suspicion  Transection" in labels:
            idx = labels.index("Suspicion  Transection")
            xt[idx].set_color("red"); yt[idx].set_color("red")
    except Exception:
        pass

    plt.tight_layout()
    return fig

# =========================
# Cached loaders
# =========================
@st.cache_data(show_spinner=False)
def _load_csv_from_path(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

@st.cache_data(show_spinner=False)
def _load_csv_from_bytes(b: bytes) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(b))

# =========================
# Meta / checkpoint helpers
# =========================
def load_model_info(path: str) -> Tuple[List[str], int, int, dict]:
    """Load feature_names & hyperparams; force input_size = len(feature_names)."""
    if not os.path.exists(path):
        st.error(f"Missing {path}. Create it with your prep script.")
        st.stop()
    with open(path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    feature_names = meta.get("feature_names")
    if not feature_names:
        feature_names = [c for c in TARGET_COLS if c != "Class"]

    if not feature_names:
        st.error("model_info.json has no feature_names and no fallback available.")
        st.stop()

    num_classes = int(meta.get("num_classes", 2))
    input_size = len(feature_names)  # <-- critical fix: do NOT trust meta['input_size']

    hp = meta.get("best_params", {}) or {}
    hidden_size = int(hp.get("hidden_size", 256))
    num_layers  = int(hp.get("num_layers", 1))
    dropout     = float(hp.get("dropout", 0.0))

    return feature_names, input_size, num_classes, {
        "hidden_size": hidden_size,
        "num_layers": num_layers,
        "dropout": dropout
    }

def _coerce_state_dict(obj):
    """Accept state_dict, {'state_dict': ...}, {'model_state_dict': ...}, or full module."""
    if isinstance(obj, nn.Module):
        return obj.state_dict()
    if isinstance(obj, dict):
        if "state_dict" in obj and isinstance(obj["state_dict"], dict):
            return obj["state_dict"]
        if "model_state_dict" in obj and isinstance(obj["model_state_dict"], dict):
            return obj["model_state_dict"]
        # heuristic: looks like a state_dict
        if all(hasattr(v, "shape") or isinstance(v, (int, float, bool)) for v in obj.values()):
            return obj
    raise ValueError("Unrecognized checkpoint format; expected a state_dict or a dict with 'state_dict'.")

def _detect_norm_type_from_state(state_dict):
    keys = list(state_dict.keys())
    if any(k.startswith("bn1.") or k.startswith("bn2.") for k in keys):
        return "batch"
    if any(k.startswith("ln1.") or k.startswith("ln2.") for k in keys):
        return "layer"
    return "none"

def stratified_fraction(df: pd.DataFrame, frac: float, label_col: str = "Class", random_state: int = 42):
    return (
        df.groupby(label_col, group_keys=False)
          .apply(lambda g: g.sample(frac=frac, random_state=random_state) if len(g) > 0 else g)
          .reset_index(drop=True)
    )

# =========================
# Main
# =========================
if run_eval:
    # --- Load data (uploaded > default path)
    csv_source = None
    if uploaded is not None:
        try:
            df = _load_csv_from_bytes(uploaded.read())
            csv_source = f"uploaded:{uploaded.name}"
        except Exception as e:
            st.error(f"Failed to read uploaded CSV: {e}")
            st.stop()
    else:
        if use_default_if_missing:
            if not os.path.exists(TEST_CSV):
                st.error(f"CSV not found: {TEST_CSV}\nUpload a CSV or set TEST_DATA_PATH.")
                st.stop()
            df = _load_csv_from_path(TEST_CSV)
            csv_source = TEST_CSV
        else:
            st.info("Please upload a CSV (toggle is off for default).")
            st.stop()

    if "Class" not in df.columns:
        st.error("Ground-truth label column `Class` not found in the CSV.")
        st.stop()

    # --- Load meta + checkpoint
    feats, input_size, num_classes, hp = load_model_info(INFO_JSON)

    if not os.path.exists(MODEL_PTH):
        st.error(f"Model checkpoint not found: {MODEL_PTH}")
        st.stop()
    raw_obj = torch.load(MODEL_PTH, map_location="cpu")
    state = _coerce_state_dict(raw_obj)
    norm_type = _detect_norm_type_from_state(state)

    # --- Derive trained input size from checkpoint; override if mismatched
    trained_input_size = state.get("input_projection.weight", torch.empty(0, 0)).shape[1] \
                         if "input_projection.weight" in state else input_size

    if trained_input_size != input_size:
        st.warning(
            f"model_info feature count ({input_size}) != checkpoint trained input size ({trained_input_size}). "
            f"Overriding to {trained_input_size}."
        )
        input_size = trained_input_size

    if len(feats) != input_size:
        st.error(
            f"Feature count ({len(feats)}) does not match checkpoint's trained input size ({input_size}).\n"
            f"Update model_info.json 'feature_names' to exactly {input_size} training columns."
        )
        st.stop()

    # --- Subsample (stratified)
    frac = percent / 100.0
    df_sub = stratified_fraction(df, frac, label_col="Class", random_state=42)

    # --- Build X (preserve training feature order)
    missing = [c for c in feats if c not in df_sub.columns]
    if missing:
        st.error(f"Your CSV lacks required feature columns: {missing}")
        st.stop()
    X = df_sub[feats].values.astype(np.float32)
    y_true = df_sub["Class"].astype(int).values

    # --- Tensor dataset (seq_len=1)
    X_tensor = torch.from_numpy(X).unsqueeze(1)  # (N, 1, input_size)
    y_tensor = torch.from_numpy(y_true).long()
    loader = DataLoader(TensorDataset(X_tensor, y_tensor), batch_size=2048, shuffle=False, num_workers=0)

    # --- Model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CreditCardGRU(
        input_size=input_size,
        hidden_size=hp["hidden_size"],
        num_layers=hp["num_layers"],
        num_classes=num_classes,
        dropout=hp["dropout"],
        norm_type=norm_type
    ).to(device)
    model.load_state_dict(state, strict=True)
    model.eval()

    # --- Inference
    all_logits, all_preds = [], []
    with torch.no_grad():
        for xb, _ in loader:
            xb = xb.to(device)
            logits = model(xb)
            all_logits.append(logits.cpu())
            all_preds.append(torch.argmax(logits, dim=1).cpu())

    logits = torch.cat(all_logits, dim=0).numpy()      # (N, C)
    y_pred = torch.cat(all_preds, dim=0).numpy()       # (N,)

    # =========================
    # Parameters panel
    # =========================
    st.subheader("Parameters")
    c1, c2 = st.columns(2)
    with c1:
        st.write({
            "CSV": csv_source,
            "Rows used": f"{len(df_sub)} / {len(df)} ({percent}%)",
            "Model": MODEL_PTH,
            "Meta": INFO_JSON,
            "Input size": input_size,
            "Hidden size": hp["hidden_size"],
            "Num layers": hp["num_layers"],
            "Dropout": hp["dropout"],
            "Num classes": num_classes,
            "Norm type": norm_type
        })
    with c2:
        st.write("Feature order (first 12):", feats[:12] + (["…"] if len(feats) > 12 else []))
        st.write("Class counts:", dict(df_sub["Class"].value_counts().sort_index()))

    # =========================
    # Results
    # =========================
    st.subheader("Results")
    st.markdown("**Confusion Matrix**")
    fig = plot_confusion_matrix_with_percentages_bin(
        y_true, y_pred, labels=["Normal Transection", "Suspicion  Transection"]
    )
    st.pyplot(fig)

    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)

    st.markdown("**Metrics**")
    st.write(
        f"- Accuracy: **{acc:.4f}**  \n"
        f"- Precision(1): **{prec:.4f}**  \n"
        f"- Recall(1): **{rec:.4f}**  \n"
        f"- F1(1): **{f1:.4f}**"
    )

    st.markdown("**Classification Report**")
    report = classification_report(y_true, y_pred, digits=4, zero_division=0)
    st.code(report, language="text")

else:
    st.info("Upload a CSV (or rely on the default path), adjust the percentage, and click **Run Evaluation** to run the model and see results.")