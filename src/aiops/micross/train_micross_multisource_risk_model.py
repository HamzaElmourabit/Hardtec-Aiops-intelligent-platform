
"""
MicroSS Multisource Incident Risk Prediction

Goal:
    Predict whether an incident will occur within the next 15 minutes.

Dataset:
    micross_multisource_incident_analysis.csv

Method:
    - Temporal split: 70% train / 15% validation / 15% test
    - Past/current observable features only
    - XGBoost classifier
    - Class imbalance handling
    - Validation threshold selection
    - Test evaluation with:
        * Precision
        * Recall
        * F1
        * PR-AUC
        * ROC-AUC
        * Confusion matrix
        * Classification report

Important:
    The +1 hour fault alignment is a working hypothesis.
    It is NOT claimed to be proven timezone synchronization.
"""

from pathlib import Path
import json
import warnings

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from xgboost import XGBClassifier


warnings.filterwarnings("ignore")


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(
    r"C:\Users\khadi\Desktop\hardtec-intelligent-ticketing"
)

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "micross"
    / "micross_multisource_incident_analysis.csv"
)

MODEL_DIR = (
    PROJECT_ROOT
    / "models"
    / "micross"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "micross"
)

MODEL_FILE = (
    MODEL_DIR
    / "micross_multisource_risk_xgboost.pkl"
)

FEATURES_FILE = (
    MODEL_DIR
    / "micross_multisource_risk_features.json"
)

VALIDATION_FILE = (
    OUTPUT_DIR
    / "micross_risk_validation_predictions.csv"
)

TEST_FILE = (
    OUTPUT_DIR
    / "micross_risk_test_predictions.csv"
)

METRICS_FILE = (
    OUTPUT_DIR
    / "micross_multisource_risk_metrics.json"
)

CONFUSION_FILE = (
    OUTPUT_DIR
    / "micross_multisource_risk_confusion_matrix.csv"
)


# ============================================================
# PARAMETERS
# ============================================================

TARGET = "incident_next_15m"

TRAIN_RATIO = 0.70
VALIDATION_RATIO = 0.15
TEST_RATIO = 0.15

RANDOM_STATE = 42

# Candidate thresholds evaluated on validation set.
THRESHOLDS = np.arange(
    0.10,
    0.91,
    0.05,
)

# We optimize F1 on validation.
# This avoids selecting the threshold using the test set.


# ============================================================
# HELPERS
# ============================================================


def print_section(title: str):
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def evaluate_predictions(
    y_true,
    probabilities,
    threshold,
):
    predictions = (
        probabilities >= threshold
    ).astype(int)

    precision = precision_score(
        y_true,
        predictions,
        zero_division=0,
    )

    recall = recall_score(
        y_true,
        predictions,
        zero_division=0,
    )

    f1 = f1_score(
        y_true,
        predictions,
        zero_division=0,
    )

    pr_auc = average_precision_score(
        y_true,
        probabilities,
    )

    try:
        roc_auc = roc_auc_score(
            y_true,
            probabilities,
        )
    except ValueError:
        roc_auc = float("nan")

    return {
        "threshold": float(threshold),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "pr_auc": float(pr_auc),
        "roc_auc": float(roc_auc),
        "predicted_positive": int(
            predictions.sum()
        ),
        "actual_positive": int(
            np.sum(y_true)
        ),
        "total": int(
            len(y_true)
        ),
    }


# ============================================================
# START
# ============================================================

print("=" * 70)
print("MICROSS MULTISOURCE INCIDENT RISK MODEL")
print("=" * 70)

print(f"[INFO] Target: {TARGET}")

print(
    f"[INFO] Input file:\n{INPUT_FILE}"
)


# ============================================================
# CHECK INPUT
# ============================================================

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"Input file not found:\n{INPUT_FILE}"
    )


MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# LOAD DATA
# ============================================================

print_section("LOADING DATA")

df = pd.read_csv(
    INPUT_FILE,
)

print(
    f"[INFO] Dataset shape: {df.shape}"
)

if "timestamp" not in df.columns:
    raise ValueError(
        "Column 'timestamp' is missing."
    )

if TARGET not in df.columns:
    raise ValueError(
        f"Target '{TARGET}' is missing."
    )


# ============================================================
# TIMESTAMP
# ============================================================

print("[INFO] Converting timestamp...")

df["timestamp"] = pd.to_datetime(
    df["timestamp"],
    errors="coerce",
)

df = df.dropna(
    subset=["timestamp"]
)

df = df.sort_values(
    "timestamp"
).reset_index(
    drop=True
)

print(
    "[INFO] Time range:"
)

print(
    f"       {df['timestamp'].min()}"
)

print(
    f"       {df['timestamp'].max()}"
)


# ============================================================
# TARGET
# ============================================================

print_section("TARGET ANALYSIS")

df[TARGET] = pd.to_numeric(
    df[TARGET],
    errors="coerce",
)

df = df.dropna(
    subset=[TARGET]
)

df[TARGET] = (
    df[TARGET]
    .astype(int)
)

# Ensure binary target.
unique_target = sorted(
    df[TARGET].unique().tolist()
)

print(
    f"[INFO] Target values: {unique_target}"
)

if not set(unique_target).issubset({0, 1}):
    raise ValueError(
        f"Target must be binary 0/1. "
        f"Found: {unique_target}"
    )

print(
    f"[INFO] Total rows: {len(df):,}"
)

print(
    f"[INFO] Positive rows: "
    f"{df[TARGET].sum():,}"
)

print(
    f"[INFO] Positive rate: "
    f"{df[TARGET].mean() * 100:.2f}%"
)


# ============================================================
# REMOVE LABEL / FUTURE INFORMATION
# ============================================================

print_section(
    "BUILDING MODEL FEATURES"
)

# Columns that must never be used as predictors.
EXCLUDED_COLUMNS = {
    "timestamp",

    # Current/future incident labels.
    "incident_now",
    "incident_next_5m",
    "incident_next_10m",
    "incident_next_15m",
    "incident_next_30m",

    # This is derived directly from future fault timestamps.
    "minutes_to_next_fault",

    # Potential descriptive anomaly fields generated from labels
    # or event correlation.
    "next_fault_timestamp",
    "fault_timestamp",
    "fault_start",
    "fault_end",

    # Target itself.
    TARGET,
}


candidate_features = [
    column
    for column in df.columns
    if column not in EXCLUDED_COLUMNS
]


print(
    f"[INFO] Candidate features: "
    f"{len(candidate_features)}"
)


# ============================================================
# NUMERIC FEATURES ONLY
# ============================================================

numeric_features = []

for column in candidate_features:

    if pd.api.types.is_numeric_dtype(
        df[column]
    ):
        numeric_features.append(
            column
        )

print(
    f"[INFO] Numeric features: "
    f"{len(numeric_features)}"
)


if not numeric_features:
    raise ValueError(
        "No numeric features available."
    )


# Remove columns that are entirely missing.
valid_features = []

for column in numeric_features:

    if df[column].notna().any():
        valid_features.append(
            column
        )

features = valid_features


print(
    f"[INFO] Final model features: "
    f"{len(features)}"
)

print("[INFO] Feature groups:")

metric_features = [
    c
    for c in features
    if c.startswith("metric_")
]

mean_features = [
    c
    for c in features
    if c.startswith("mean_")
]

count_features = [
    c
    for c in features
    if c.startswith("count_")
]

other_features = [
    c
    for c in features
    if c not in metric_features
    and c not in mean_features
    and c not in count_features
]

print(
    f"       Metric features: "
    f"{len(metric_features)}"
)

print(
    f"       Mean features:   "
    f"{len(mean_features)}"
)

print(
    f"       Count features:  "
    f"{len(count_features)}"
)

print(
    f"       Other features:  "
    f"{len(other_features)}"
)


# ============================================================
# FEATURE MATRIX
# ============================================================

X = df[features].copy()

y = df[TARGET].copy()

# Replace infinities.
X = X.replace(
    [np.inf, -np.inf],
    np.nan,
)

# XGBoost can handle NaN values natively.
# Therefore, we intentionally do NOT forward-fill
# or interpolate using future observations.


# ============================================================
# TEMPORAL SPLIT
# ============================================================

print_section(
    "TEMPORAL TRAIN / VALIDATION / TEST SPLIT"
)

n_rows = len(df)

train_end = int(
    n_rows * TRAIN_RATIO
)

validation_end = int(
    n_rows
    * (
        TRAIN_RATIO
        + VALIDATION_RATIO
    )
)

X_train = X.iloc[
    :train_end
].copy()

y_train = y.iloc[
    :train_end
].copy()

X_validation = X.iloc[
    train_end:validation_end
].copy()

y_validation = y.iloc[
    train_end:validation_end
].copy()

X_test = X.iloc[
    validation_end:
].copy()

y_test = y.iloc[
    validation_end:
].copy()


print(
    f"[INFO] Train rows: "
    f"{len(X_train):,}"
)

print(
    f"[INFO] Validation rows: "
    f"{len(X_validation):,}"
)

print(
    f"[INFO] Test rows: "
    f"{len(X_test):,}"
)


print()
print("[INFO] Train time:")
print(
    f"       {df['timestamp'].iloc[0]}"
)
print(
    f"       {df['timestamp'].iloc[train_end - 1]}"
)

print()
print("[INFO] Validation time:")
print(
    f"       {df['timestamp'].iloc[train_end]}"
)
print(
    f"       {df['timestamp'].iloc[validation_end - 1]}"
)

print()
print("[INFO] Test time:")
print(
    f"       {df['timestamp'].iloc[validation_end]}"
)
print(
    f"       {df['timestamp'].iloc[-1]}"
)


# ============================================================
# CLASS BALANCE
# ============================================================

print_section(
    "CLASS BALANCE"
)

train_positive = int(
    y_train.sum()
)

train_negative = int(
    len(y_train)
    - train_positive
)

if train_positive == 0:
    raise ValueError(
        "Training set contains no positive samples."
    )

scale_pos_weight = (
    train_negative
    / train_positive
)

print(
    f"[INFO] Train negative: "
    f"{train_negative:,}"
)

print(
    f"[INFO] Train positive: "
    f"{train_positive:,}"
)

print(
    f"[INFO] Positive rate: "
    f"{y_train.mean() * 100:.2f}%"
)

print(
    f"[INFO] scale_pos_weight: "
    f"{scale_pos_weight:.4f}"
)


# ============================================================
# MODEL
# ============================================================

print_section(
    "TRAINING XGBOOST"
)

model = XGBClassifier(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,

    objective="binary:logistic",
    eval_metric="logloss",

    scale_pos_weight=scale_pos_weight,

    random_state=RANDOM_STATE,
    n_jobs=-1,

    tree_method="hist",
)

print("[INFO] Model parameters:")
print(
    "       n_estimators      = 300"
)
print(
    "       max_depth         = 5"
)
print(
    "       learning_rate     = 0.05"
)
print(
    "       subsample         = 0.8"
)
print(
    "       colsample_bytree  = 0.8"
)
print(
    f"       scale_pos_weight  = "
    f"{scale_pos_weight:.4f}"
)

print()
print(
    "[INFO] Training..."
)

model.fit(
    X_train,
    y_train,
    eval_set=[
        (
            X_validation,
            y_validation,
        )
    ],
    verbose=False,
)

print(
    "[INFO] Training complete."
)


# ============================================================
# VALIDATION
# ============================================================

print_section(
    "VALIDATION THRESHOLD SELECTION"
)

validation_probabilities = (
    model.predict_proba(
        X_validation
    )[:, 1]
)

validation_results = []

for threshold in THRESHOLDS:

    result = evaluate_predictions(
        y_validation,
        validation_probabilities,
        threshold,
    )

    validation_results.append(
        result
    )


validation_results_df = pd.DataFrame(
    validation_results
)

best_row = (
    validation_results_df
    .sort_values(
        [
            "f1",
            "recall",
        ],
        ascending=False,
    )
    .iloc[0]
)

best_threshold = float(
    best_row["threshold"]
)

print(
    f"[INFO] Best validation threshold: "
    f"{best_threshold:.2f}"
)

print(
    f"[INFO] Validation precision: "
    f"{best_row['precision']:.4f}"
)

print(
    f"[INFO] Validation recall: "
    f"{best_row['recall']:.4f}"
)

print(
    f"[INFO] Validation F1: "
    f"{best_row['f1']:.4f}"
)

print(
    f"[INFO] Validation PR-AUC: "
    f"{best_row['pr_auc']:.4f}"
)

print(
    f"[INFO] Validation ROC-AUC: "
    f"{best_row['roc_auc']:.4f}"
)


# ============================================================
# SAVE VALIDATION PREDICTIONS
# ============================================================

validation_predictions = pd.DataFrame(
    {
        "timestamp": df[
            "timestamp"
        ].iloc[
            train_end:validation_end
        ].values,

        "actual_incident_next_15m":
            y_validation.values,

        "risk_probability":
            validation_probabilities,

        "predicted_incident":
            (
                validation_probabilities
                >= best_threshold
            ).astype(int),
    }
)

validation_predictions.to_csv(
    VALIDATION_FILE,
    index=False,
)

print(
    f"[INFO] Validation predictions saved:\n"
    f"{VALIDATION_FILE}"
)


# ============================================================
# TEST EVALUATION
# ============================================================

print_section(
    "FINAL TEST EVALUATION"
)

test_probabilities = (
    model.predict_proba(
        X_test
    )[:, 1]
)

test_predictions = (
    test_probabilities
    >= best_threshold
).astype(int)


test_precision = precision_score(
    y_test,
    test_predictions,
    zero_division=0,
)

test_recall = recall_score(
    y_test,
    test_predictions,
    zero_division=0,
)

test_f1 = f1_score(
    y_test,
    test_predictions,
    zero_division=0,
)

test_pr_auc = (
    average_precision_score(
        y_test,
        test_probabilities,
    )
)

try:
    test_roc_auc = roc_auc_score(
        y_test,
        test_probabilities,
    )
except ValueError:
    test_roc_auc = float("nan")


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test,
    test_predictions,
)

tn, fp, fn, tp = cm.ravel()

specificity = (
    tn / (tn + fp)
    if (tn + fp) > 0
    else 0.0
)

print()
print(
    "TEST RESULTS"
)

print(
    f"  Threshold:       {best_threshold:.2f}"
)

print(
    f"  Precision:        {test_precision:.4f}"
)

print(
    f"  Recall:           {test_recall:.4f}"
)

print(
    f"  F1:               {test_f1:.4f}"
)

print(
    f"  PR-AUC:           {test_pr_auc:.4f}"
)

print(
    f"  ROC-AUC:          {test_roc_auc:.4f}"
)

print(
    f"  Specificity:      {specificity:.4f}"
)

print()
print(
    "CONFUSION MATRIX"
)

print(
    f"  True Negative:    {tn:,}"
)

print(
    f"  False Positive:   {fp:,}"
)

print(
    f"  False Negative:   {fn:,}"
)

print(
    f"  True Positive:    {tp:,}"
)

print()
print(
    f"  Actual positives: "
    f"{int(y_test.sum()):,}"
)

print(
    f"  Predicted positives: "
    f"{int(test_predictions.sum()):,}"
)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print()
print(
    "CLASSIFICATION REPORT"
)

report = classification_report(
    y_test,
    test_predictions,
    target_names=[
        "No incident",
        "Incident",
    ],
    zero_division=0,
)

print(report)


# ============================================================
# SAVE TEST PREDICTIONS
# ============================================================

test_predictions_df = pd.DataFrame(
    {
        "timestamp": df[
            "timestamp"
        ].iloc[
            validation_end:
        ].values,

        "actual_incident_next_15m":
            y_test.values,

        "risk_probability":
            test_probabilities,

        "predicted_incident":
            test_predictions,
    }
)

test_predictions_df.to_csv(
    TEST_FILE,
    index=False,
)

print(
    f"[INFO] Test predictions saved:\n"
    f"{TEST_FILE}"
)


# ============================================================
# SAVE CONFUSION MATRIX
# ============================================================

confusion_df = pd.DataFrame(
    cm,
    index=[
        "actual_0",
        "actual_1",
    ],
    columns=[
        "predicted_0",
        "predicted_1",
    ],
)

confusion_df.to_csv(
    CONFUSION_FILE,
)

print(
    f"[INFO] Confusion matrix saved:\n"
    f"{CONFUSION_FILE}"
)


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

print_section(
    "TOP FEATURE IMPORTANCE"
)

importance_df = pd.DataFrame(
    {
        "feature": features,
        "importance": model.feature_importances_,
    }
)

importance_df = (
    importance_df
    .sort_values(
        "importance",
        ascending=False,
    )
)

print(
    importance_df.head(20).to_string(
        index=False
    )
)


IMPORTANCE_FILE = (
    OUTPUT_DIR
    / "micross_multisource_risk_feature_importance.csv"
)

importance_df.to_csv(
    IMPORTANCE_FILE,
    index=False,
)

print(
    f"[INFO] Feature importance saved:\n"
    f"{IMPORTANCE_FILE}"
)


# ============================================================
# SAVE MODEL
# ============================================================

print_section(
    "SAVING MODEL"
)

joblib.dump(
    model,
    MODEL_FILE,
)

print(
    f"[INFO] Model saved:\n"
    f"{MODEL_FILE}"
)


# ============================================================
# SAVE FEATURE LIST
# ============================================================

feature_metadata = {
    "target": TARGET,
    "n_features": len(features),
    "features": features,
    "train_ratio": TRAIN_RATIO,
    "validation_ratio": VALIDATION_RATIO,
    "test_ratio": TEST_RATIO,
    "best_validation_threshold": best_threshold,
    "random_state": RANDOM_STATE,
    "feature_policy": (
        "Numeric observable/current features only. "
        "Future-derived labels and minutes_to_next_fault "
        "were excluded."
    ),
}

with open(
    FEATURES_FILE,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        feature_metadata,
        f,
        indent=2,
    )

print(
    f"[INFO] Feature metadata saved:\n"
    f"{FEATURES_FILE}"
)


# ============================================================
# SAVE METRICS
# ============================================================

metrics = {
    "project": "HARDTEC MicroSS Multisource AIOps",
    "task": (
        "Predict incident within next 15 minutes"
    ),
    "target": TARGET,

    "dataset": {
        "rows": int(len(df)),
        "features": int(len(features)),
        "start": str(
            df["timestamp"].min()
        ),
        "end": str(
            df["timestamp"].max()
        ),
    },

    "split": {
        "train_rows": int(
            len(X_train)
        ),
        "validation_rows": int(
            len(X_validation)
        ),
        "test_rows": int(
            len(X_test)
        ),
        "method": (
            "chronological 70/15/15"
        ),
    },

    "class_balance": {
        "train_positive": train_positive,
        "train_negative": train_negative,
        "train_positive_rate": float(
            y_train.mean()
        ),
        "scale_pos_weight": float(
            scale_pos_weight
        ),
    },

    "validation": {
        "threshold": best_threshold,
        "precision": float(
            best_row["precision"]
        ),
        "recall": float(
            best_row["recall"]
        ),
        "f1": float(
            best_row["f1"]
        ),
        "pr_auc": float(
            best_row["pr_auc"]
        ),
        "roc_auc": float(
            best_row["roc_auc"]
        ),
    },

    "test": {
        "threshold": best_threshold,
        "precision": float(
            test_precision
        ),
        "recall": float(
            test_recall
        ),
        "f1": float(
            test_f1
        ),
        "pr_auc": float(
            test_pr_auc
        ),
        "roc_auc": float(
            test_roc_auc
        ),
        "specificity": float(
            specificity
        ),
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
        "actual_positive": int(
            y_test.sum()
        ),
        "predicted_positive": int(
            test_predictions.sum()
        ),
    },

    "alignment": {
        "fault_alignment": "+1 hour",
        "status": (
            "working hypothesis, "
            "not proven timezone synchronization"
        ),
    },

    "leakage_policy": [
        "chronological split",
        "future incident labels excluded",
        "minutes_to_next_fault excluded",
        "no random train/test split",
        "no future interpolation or forward-looking filling",
    ],
}

with open(
    METRICS_FILE,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        metrics,
        f,
        indent=2,
    )

print(
    f"[INFO] Metrics saved:\n"
    f"{METRICS_FILE}"
)


# ============================================================
# FINAL
# ============================================================

print()
print("=" * 70)
print(
    "MICROSS MULTISOURCE RISK MODEL COMPLETE"
)
print("=" * 70)

print(
    f"Target:                  {TARGET}"
)

print(
    f"Features:                {len(features)}"
)

print(
    f"Train rows:              {len(X_train):,}"
)

print(
    f"Validation rows:         {len(X_validation):,}"
)

print(
    f"Test rows:               {len(X_test):,}"
)

print(
    f"Selected threshold:      {best_threshold:.2f}"
)

print()
print(
    "FINAL TEST METRICS"
)

print(
    f"Precision:               {test_precision:.4f}"
)

print(
    f"Recall:                  {test_recall:.4f}"
)

print(
    f"F1:                      {test_f1:.4f}"
)

print(
    f"PR-AUC:                  {test_pr_auc:.4f}"
)

print(
    f"ROC-AUC:                 {test_roc_auc:.4f}"
)

print()
print(
    "Model:"
)

print(
    str(MODEL_FILE)
)

print("=" * 70)

