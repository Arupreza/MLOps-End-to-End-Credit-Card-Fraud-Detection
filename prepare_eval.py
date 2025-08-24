#!/usr/bin/env python3
# Prepares model_info.json (feature schema + label) and creates a stable
# checkpoint alias models/best_fraud_detection_model.pth for the evaluator.

import os, json, shutil
import pandas as pd

TRAIN_CSV   = "Data/processed/creditcard_processed_train.csv"
STUDY_JSON  = "optuna_study/study_results.json"
MODELS_DIR  = "models"
ALIAS_PATH  = os.path.join(MODELS_DIR, "best_fraud_detection_model.pth")

def main():
    os.makedirs(MODELS_DIR, exist_ok=True)

    # 1) Capture training schema (feature order + label)
    df = pd.read_csv(TRAIN_CSV, nrows=1)
    label_col = "Class" if "Class" in df.columns else df.columns[-1]
    feature_names = [c for c in df.columns if c != label_col]

    # 2) Load best_params from Optuna if present
    best_params, best_trial = {}, None
    try:
        with open(STUDY_JSON) as f:
            sr = json.load(f)
        best_params = sr.get("best_params", {}) or {}
        if not best_params and "trials" in sr:
            trials = [t for t in sr["trials"] if t.get("value") is not None]
            if trials:
                best_trial = max(trials, key=lambda t: t["value"])
    except Exception as e:
        print(f"[WARN] Couldn't read {STUDY_JSON}: {e}")

    # 3) Write models/model_info.json (used by evaluate.py)
    meta = {
        "best_params": best_params,          # hidden_size/num_layers/dropout if available
        "num_classes": 2,
        "experiment_name": "fraud_detection_evaluation",
        "feature_names": feature_names,      # critical: preserves column order
        "label_col": label_col
    }
    with open(os.path.join(MODELS_DIR, "model_info.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[OK] wrote models/model_info.json ({len(feature_names)} features, label='{label_col}')")

    # 4) Create a stable checkpoint alias for the evaluator
    # Prefer an existing *_best.pth under models/, else latest .pth; else try optuna artifacts
    ckpt_candidates = []
    for name in os.listdir(MODELS_DIR):
        if name.endswith(".pth"):
            ckpt_candidates.append(os.path.join(MODELS_DIR, name))

    chosen = None
    # prefer any *_best.pth
    best_like = [p for p in ckpt_candidates if p.endswith("_best.pth")]
    if best_like:
        # pick the most recent *_best.pth
        chosen = max(best_like, key=os.path.getmtime)
    elif ckpt_candidates:
        chosen = max(ckpt_candidates, key=os.path.getmtime)

    if chosen and chosen != ALIAS_PATH:
        shutil.copyfile(chosen, ALIAS_PATH)
        print(f"[OK] aliased {chosen} -> {ALIAS_PATH}")
    elif os.path.exists(ALIAS_PATH):
        print(f"[OK] alias exists: {ALIAS_PATH}")
    else:
        # try optuna best trial
        if best_trial is not None:
            tnum = best_trial["number"]
            src = os.path.join("optuna_artifacts", f"trial_{tnum}", "model_state_dict.pth")
            if os.path.exists(src):
                shutil.copyfile(src, ALIAS_PATH)
                print(f"[OK] aliased {src} -> {ALIAS_PATH}")
            else:
                print(f"[WARN] no checkpoint found at {src}. Please copy your best .pth to {ALIAS_PATH}.")

if __name__ == "__main__":
    main()
