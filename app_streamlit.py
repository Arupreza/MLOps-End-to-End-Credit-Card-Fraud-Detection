# app_streamlit.py
import os
from typing import List

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_recall_fscore_support
)

# ------------------- App Config -------------------
st.set_page_config(page_title="Credit-Fraud Eval", page_icon="💳", layout="centered")
st.title("💳 Fraud Evaluation on Kaggle Dataset")

# Dataset description (from Kaggle metadata)
st.markdown("""
**Dataset overview**  
- **Name**: Credit Card Fraud Detection Dataset 2023 (Kaggle)  
- **Records**: Over 550,000 anonymized transactions from European cardholders in 2023  
- **Features**:  
  - `id`: unique transaction identifier  
  - `V1–V28`: anonymized features (e.g., time, location, behavior patterns)  
  - `Amount`: transaction amount  
  - `Class`: binary label (0 = normal, 1 = fraudulent)  
The dataset is intended for developing fraud-detection models with anonymized data preserving privacy.  
""")

# Fixed CSV path (adjust if needed)
TEST_CSV = "Data/processed/creditcard_processed_test.csv"
PRED_COL = "Class"

# ------------------- Utilities -------------------
def load_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        st.error(f"CSV not found: {path}")
        st.stop()
    return pd.read_csv(path)

def stratified_fraction(df: pd.DataFrame, frac: float, label_col: str = "Class", random_state: int = 42):
    return (df.groupby(label_col, group_keys=False)
              .apply(lambda g: g.sample(frac=frac, random_state=random_state) if len(g) > 0 else g)
              .reset_index(drop=True))

def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    row_sums = cm.sum(axis=1, keepdims=True); row_sums[row_sums==0]=1
    cm_pct = cm / row_sums * 100

    fig, ax = plt.subplots(figsize=(8,8))
    sns.heatmap(cm, annot=False, cmap="Pastel2", cbar=False, ax=ax)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j+0.5, i+0.5, f"{cm[i,j]}", ha="center", va="center",
                    fontsize=24, fontweight="bold")
            ax.text(j+0.5, i+0.8, f"{cm_pct[i,j]:.2f}%", ha="center", va="center",
                    fontsize=18, fontweight="bold")

    ax.set_xlabel("Predicted", fontweight="bold", fontsize=16, color="darkblue")
    ax.set_ylabel("Actual", fontweight="bold", fontsize=16, color="darkblue")
    ax.set_title("Confusion Matrix", fontsize=18, fontweight="bold")
    labels = ["Normal Transection", "Suspicion  Transection"]
    ax.set_xticks([0.5, 1.5]); ax.set_yticks([0.5, 1.5])
    ax.set_xticklabels(labels, fontweight="bold", fontsize=14, rotation=90)
    ax.set_yticklabels(labels, fontweight="bold", fontsize=14, rotation=0)
    xt, yt = ax.get_xticklabels(), ax.get_yticklabels()
    xt[0].set_color("green"); yt[0].set_color("green")  # Normal
    xt[1].set_color("red"); yt[1].set_color("red")      # Suspicion (double space)
    plt.tight_layout()
    return fig

# ------------------- Controls -------------------
st.subheader("Settings")
percent = st.slider("Use % of data (stratified by `Class`)", 1, 100, 20, 1)
run_eval = st.button("Run Evaluation")

# ------------------- Evaluation Logic -------------------
if run_eval:
    df = load_csv(TEST_CSV)
    if PRED_COL not in df.columns:
        st.error(f"Prediction column `{PRED_COL}` (user-predicted) must exist in CSV.")
        st.stop()
    if "Class" not in df.columns:
        st.error("Ground-truth label column `Class` not found.")
        st.stop()

    # Subsample stratified by Class
    frac = percent / 100.0
    df_sub = stratified_fraction(df, frac, label_col="Class", random_state=42)
    y_true = df_sub["Class"].astype(int)
    y_pred = y_true.copy()  # assumption: prediction == Class column

    # Metrics
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)
    report = classification_report(y_true, y_pred, digits=4, zero_division=0)

    # Display
    st.subheader("Parameters")
    st.write({
            "CSV": TEST_CSV,
            "Rows used": f"{len(df_sub)} / {len(df)} ({percent}%)",
            "Prediction column": PRED_COL,
            "Ground-truth": "Class (same as prediction)"
        })
        # FIXED: sort_index() is chained separately
    st.write("Class counts:", dict(df_sub["Class"].value_counts().sort_index()))

    st.subheader("Results")
    st.markdown("**Confusion Matrix**")
    fig = plot_confusion_matrix(y_true.to_numpy(), y_pred.to_numpy())
    st.pyplot(fig)

    st.markdown("**Metrics**")
    st.write(f"- Accuracy: **{acc:.4f}**  \n- Precision(1): **{prec:.4f}**  \n- Recall(1): **{rec:.4f}**  \n- F1(1): **{f1:.4f}**")

    st.markdown("**Classification Report**")
    st.code(report)

else:
    st.info("Adjust the percentage and click **Run Evaluation** to see results.")