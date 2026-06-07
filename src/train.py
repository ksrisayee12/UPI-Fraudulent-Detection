#!/usr/bin/env python3
"""
train.py -- Train CatBoost fraud model on final 14 features.

Produces:
 - models/fraud_model.cbm
 - models/fraud_model_metadata.json
 - models/feature_importances.csv

Usage:
    python src/train.py
"""

import os
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    classification_report,
    confusion_matrix,
)
from catboost import CatBoostClassifier

# --------------------------
# CONFIG
# --------------------------
DATA_PATH = "synthetic_training_realistic.csv"
MODEL_DIR = "models"
MODEL_FILENAME = "fraud_modelN.cbm"
METADATA_FILENAME = "fraud_model_metadataN.json"
FEATURE_IMP_FILENAME = "feature_importancesN.csv"
RANDOM_SEED = 42

FEATURES = [
    "transaction_amount",
    "amount_zscore",
    "hour_of_day",
    "is_night",
    "day_of_week",
    "device_change_flag",
    "location_risk_level",
    "location_velocity_flag",
    "sender_bank_code",
    "receiver_bank_code",
    "velocity_1h",
    "failed_attempts_24h",
    "fraud_history_flag",
    "is_new_beneficiary",
]

TARGET = "fraud_label"

CAT_PARAMS = {
    "iterations": 1500,
    "depth": 6,
    "learning_rate": 0.03,
    "loss_function": "Logloss",
    "eval_metric": "AUC",
    "random_seed": RANDOM_SEED,
    "l2_leaf_reg": 3,
    "verbose": 200,
}

EARLY_STOPPING_ROUNDS = 150
MAX_CLASS_WEIGHT = 1000


# --------------------------
# HELPERS
# --------------------------

def safe_mkdir(path: str):
    Path(path).mkdir(parents=True, exist_ok=True)

def choose_class_weights(fraud_rate: float):
    if fraud_rate <= 0:
        return [1, 1]
    desired_ratio = (1.0 - fraud_rate) / max(1e-6, fraud_rate)
    weight_for_1 = float(np.clip(desired_ratio, 1.0, MAX_CLASS_WEIGHT))
    return [1.0, weight_for_1]

def print_conf_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    print("Confusion matrix:")
    print(f"TN: {tn}, FP: {fp}, FN: {fn}, TP: {tp}")
    return {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}


# --------------------------
# MAIN
# --------------------------
def main():
    np.random.seed(RANDOM_SEED)

    # 1) Load data
    if not Path(DATA_PATH).exists():
        raise SystemExit(f"ERROR: DATA_PATH not found: {DATA_PATH}")

    print("Loading data:", DATA_PATH)
    df = pd.read_csv(DATA_PATH)
    print("Raw data shape:", df.shape)

    missing_cols = [c for c in FEATURES + [TARGET] if c not in df.columns]
    if missing_cols:
        raise SystemExit(f"Missing required columns: {missing_cols}")

    if df[FEATURES].isna().any().any():
        raise SystemExit("ERROR: NaN values found in feature columns.")

    fraud_rate = float(df[TARGET].mean())
    print(f"Observed fraud rate: {fraud_rate:.6f} ({fraud_rate*100:.3f}%)")

    class_weights = choose_class_weights(fraud_rate)
    print("Using class_weights:", class_weights)

    # 2) Sequential split
    n = len(df)
    train_end = int(n * 0.7)
    X_train = df.loc[:train_end-1, FEATURES].reset_index(drop=True)
    y_train = df.loc[:train_end-1, TARGET].reset_index(drop=True)
    X_test = df.loc[train_end:, FEATURES].reset_index(drop=True)
    y_test = df.loc[train_end:, TARGET].reset_index(drop=True)

    print("Train shape:", X_train.shape, "Test shape:", X_test.shape)

    # 3) Train model
    model_params = CAT_PARAMS.copy()
    model_params["class_weights"] = class_weights

    model = CatBoostClassifier(**model_params)

    print("Starting training...")
    model.fit(
        X_train, y_train,
        eval_set=(X_test, y_test),
        use_best_model=True,
        early_stopping_rounds=EARLY_STOPPING_ROUNDS
    )

    # 4) Evaluation
    print("Predicting probabilities...")
    y_prob = model.predict_proba(X_test)[:, 1]

    auc = roc_auc_score(y_test, y_prob)
    pr_auc = average_precision_score(y_test, y_prob)

    print(f"AUC: {auc:.6f}")
    print(f"PR-AUC: {pr_auc:.6f}")

    # --------------------------
    # UPDATED THRESHOLD SECTION
    # --------------------------

    precision, recall, thresholds = precision_recall_curve(y_test, y_prob)

    # 1. Max-F1 threshold
    f1_scores = (2 * precision * recall) / np.maximum(precision + recall, 1e-12)
    max_f1_idx = np.nanargmax(f1_scores)
    f1_threshold = thresholds[max_f1_idx] if max_f1_idx < len(thresholds) else 0.5

    print(f"\n[THRESHOLD] Max-F1 threshold: {f1_threshold:.4f}")

    # 2. Block threshold (≥95% precision)
    block_threshold = 0.90
    high_precision_idxs = np.where(precision >= 0.95)[0]
    if len(high_precision_idxs) > 0:
        block_threshold = thresholds[max(high_precision_idxs)-1]
    print(f"[THRESHOLD] Block threshold (≥95% precision): {block_threshold:.4f}")

    # 3. Review threshold (top 5%)
    review_threshold = np.percentile(y_prob, 95)
    print(f"[THRESHOLD] Review threshold (top 5%): {review_threshold:.4f}")

    # 4. Monitor threshold (top 20%)
    monitor_threshold = np.percentile(y_prob, 80)
    print(f"[THRESHOLD] Monitor threshold (top 20%): {monitor_threshold:.4f}")

    thresholds_dict = {
        "max_f1": float(f1_threshold),
        "block": float(block_threshold),
        "review": float(review_threshold),
        "monitor": float(monitor_threshold)
    }

    # Evaluate using Max-F1 threshold
    y_pred = (y_prob >= f1_threshold).astype(int)

    print("\nClassification report (Max-F1 threshold):")
    print(classification_report(y_test, y_pred, digits=4))

    cm = print_conf_matrix(y_test, y_pred)

    # Precision@K
    def precision_at_k(y_true, y_score, k):
        idx = np.argsort(y_score)[::-1][:k]
        return float(np.mean(y_true.iloc[idx]))

    for k in [100, 500, 1000, 5000]:
        if k < len(y_test):
            print(f"Precision@{k}: {precision_at_k(y_test, y_prob, k):.4f}")

    # 5) Save model + metadata
    safe_mkdir(MODEL_DIR)
    model_path = os.path.join(MODEL_DIR, MODEL_FILENAME)
    metadata_path = os.path.join(MODEL_DIR, METADATA_FILENAME)
    featimp_path = os.path.join(MODEL_DIR, FEATURE_IMP_FILENAME)

    print("Saving model:", model_path)
    model.save_model(model_path)

    fi = model.get_feature_importance()
    feat_df = pd.DataFrame({"feature": FEATURES, "importance": fi})
    feat_df.sort_values("importance", ascending=False).to_csv(featimp_path, index=False)

    metadata = {
        "model_file": model_path,
        "features": FEATURES,
        "fraud_rate": fraud_rate,
        "class_weights": class_weights,
        "thresholds": thresholds_dict,
        "AUC": float(auc),
        "PR_AUC": float(pr_auc),
        "confusion_matrix": cm,
        "n_train": len(X_train),
        "n_test": len(X_test),
    }

    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print("Saved metadata:", metadata_path)
    print("Training completed successfully.")


if __name__ == "__main__":
    main()
