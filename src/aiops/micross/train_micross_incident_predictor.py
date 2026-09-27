
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from xgboost import XGBClassifier


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]

DATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "micross"
    / "micross_prediction_dataset.csv"
)

MODEL_DIR = PROJECT_ROOT / "models" / "micross"
RESULT_DIR = PROJECT_ROOT / "data" / "processed" / "micross"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "incident_next_15m"

RANDOM_STATE = 42

# Chronological split
TRAIN_RATIO = 0.70
VALID_RATIO = 0.15
TEST_RATIO = 0.15


# ============================================================
# DISPLAY
# ============================================================

print("=" * 70)
print("MICROSS - AIOPS INCIDENT PREDICTION")
print("=" * 70)

print()
print("Project root:")
print(PROJECT_ROOT)

print()
print("Dataset:")
print(DATA_FILE)


# ============================================================
# 1. LOAD DATA
# ============================================================

print()
print("[1] Loading dataset...")

df = pd.read_csv(DATA_FILE)

if TARGET not in df.columns:
    raise RuntimeError(
        f"Target column '{TARGET}' not found."
    )

if "timestamp" not in df.columns:
    raise RuntimeError(
        "Temporal column 'timestamp' not found."
    )

df["timestamp"] = pd.to_datetime(
    df["timestamp"],
    errors="coerce"
)

df = df.dropna(
    subset=["timestamp", "service", TARGET]
).copy()

df[TARGET] = df[TARGET].astype(int)

# Critical: chronological ordering
df = df.sort_values(
    "timestamp"
).reset_index(drop=True)

print(f"Rows: {len(df):,}")
print(f"Columns: {len(df.columns)}")
print(f"Services: {df['service'].nunique()}")

print(
    f"Time range: "
    f"{df['timestamp'].min()} -> "
    f"{df['timestamp'].max()}"
)


# ============================================================
# 2. TARGET ANALYSIS
# ============================================================

print()
print("[2] Target distribution...")

target_counts = df[TARGET].value_counts().sort_index()

print(target_counts)

positive_rate = df[TARGET].mean()

print(
    f"Positive rate: {positive_rate:.4%}"
)


# ============================================================
# 3. CREATE SAFE TIME FEATURES
# ============================================================

print()
print("[3] Creating temporal features...")

# Time features are derived only from the current timestamp.
# They do not use future information.

df["hour"] = df["timestamp"].dt.hour
df["minute"] = df["timestamp"].dt.minute

df["day_of_week"] = df["timestamp"].dt.dayofweek

df["is_weekend"] = (
    df["day_of_week"] >= 5
).astype(int)

# Cyclic representation of time
df["hour_sin"] = np.sin(
    2 * np.pi * df["hour"] / 24
)

df["hour_cos"] = np.cos(
    2 * np.pi * df["hour"] / 24
)

df["minute_sin"] = np.sin(
    2 * np.pi * df["minute"] / 60
)

df["minute_cos"] = np.cos(
    2 * np.pi * df["minute"] / 60
)


# ============================================================
# 4. REMOVE LEAKAGE
# ============================================================

print()
print("[4] Removing leakage columns...")

# These columns contain future information or direct incident
# information and MUST NOT be used as model features.

LEAKAGE_COLUMNS = [
    "incident_next_5m",
    "incident_next_10m",
    "incident_next_15m",
    "minutes_to_next_incident",
    "incident_active_now",
]

print()
print("Leakage columns removed:")

for col in LEAKAGE_COLUMNS:
    if col in df.columns:
        print(f"  - {col}")


# ============================================================
# 5. REMOVE IDENTIFIERS / NON-PREDICTIVE COLUMNS
# ============================================================

print()
print("[5] Removing identifiers and raw timestamp...")

DROP_COLUMNS = [
    TARGET,
    "timestamp",
]

for col in LEAKAGE_COLUMNS:
    if col in df.columns:
        DROP_COLUMNS.append(col)

for col in [
    "fault_id",
    "block_id",
    "window_start",
    "window_end",
    "aligned_fault_start",
    "aligned_fault_end",
    "fault_type",
    "fault_service",
]:
    if col in df.columns:
        DROP_COLUMNS.append(col)

DROP_COLUMNS = list(dict.fromkeys(DROP_COLUMNS))

X = df.drop(
    columns=DROP_COLUMNS,
    errors="ignore"
).copy()

y = df[TARGET].copy()


# ============================================================
# 6. FEATURE TYPES
# ============================================================

numeric_features = X.select_dtypes(
    include=["number", "bool"]
).columns.tolist()

categorical_features = X.select_dtypes(
    include=["object", "category"]
).columns.tolist()

print()
print("Numeric features:")
for col in numeric_features:
    print(f"  - {col}")

print()
print("Categorical features:")
for col in categorical_features:
    print(f"  - {col}")


# ============================================================
# 7. TEMPORAL SPLIT
# ============================================================

print()
print("[6] Chronological train / validation / test split...")

n = len(df)

train_end = int(
    n * TRAIN_RATIO
)

valid_end = int(
    n * (TRAIN_RATIO + VALID_RATIO)
)

train_idx = np.arange(
    0,
    train_end
)

valid_idx = np.arange(
    train_end,
    valid_end
)

test_idx = np.arange(
    valid_end,
    n
)

X_train = X.iloc[train_idx].copy()
y_train = y.iloc[train_idx].copy()

X_valid = X.iloc[valid_idx].copy()
y_valid = y.iloc[valid_idx].copy()

X_test = X.iloc[test_idx].copy()
y_test = y.iloc[test_idx].copy()


def describe_split(
    name,
    X_split,
    y_split
):
    print()
    print(name)
    print("-" * 50)

    print(
        f"Rows: {len(X_split):,}"
    )

    print(
        f"Start: "
        f"{df['timestamp'].iloc[X_split.index.min()]}"
    )

    print(
        f"End:   "
        f"{df['timestamp'].iloc[X_split.index.max()]}"
    )

    print(
        f"Positive incidents: "
        f"{int(y_split.sum()):,}"
    )

    print(
        f"Positive rate: "
        f"{y_split.mean():.4%}"
    )


describe_split(
    "TRAIN",
    X_train,
    y_train
)

describe_split(
    "VALIDATION",
    X_valid,
    y_valid
)

describe_split(
    "TEST",
    X_test,
    y_test
)


# ============================================================
# 8. PREPROCESSING
# ============================================================

print()
print("[7] Building preprocessing pipeline...")

numeric_transformer = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="median"
            )
        )
    ]
)

categorical_transformer = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="most_frequent"
            )
        ),
        (
            "onehot",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            )
        ),
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            numeric_transformer,
            numeric_features
        ),
        (
            "categorical",
            categorical_transformer,
            categorical_features
        ),
    ],
    remainder="drop"
)


# ============================================================
# 9. CLASS IMBALANCE
# ============================================================

print()
print("[8] Computing class imbalance...")

positive_count = int(
    y_train.sum()
)

negative_count = int(
    len(y_train) - positive_count
)

if positive_count == 0:
    raise RuntimeError(
        "Training set contains zero positive incidents."
    )

scale_pos_weight = (
    negative_count /
    positive_count
)

print(
    f"Negative samples: {negative_count:,}"
)

print(
    f"Positive samples: {positive_count:,}"
)

print(
    f"scale_pos_weight: "
    f"{scale_pos_weight:.2f}"
)


# ============================================================
# 10. XGBOOST
# ============================================================

print()
print("[9] Training XGBoost...")

xgb_model = XGBClassifier(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.05,

    subsample=0.85,
    colsample_bytree=0.85,

    min_child_weight=3,
    gamma=0.1,

    reg_alpha=0.1,
    reg_lambda=1.0,

    objective="binary:logistic",

    eval_metric="aucpr",

    scale_pos_weight=scale_pos_weight,

    random_state=RANDOM_STATE,

    n_jobs=-1,

    tree_method="hist",
)


pipeline = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "model",
            xgb_model
        ),
    ]
)

pipeline.fit(
    X_train,
    y_train
)


# ============================================================
# 11. VALIDATION THRESHOLD OPTIMIZATION
# ============================================================

print()
print("[10] Optimizing prediction threshold...")

valid_proba = pipeline.predict_proba(
    X_valid
)[:, 1]

thresholds = np.arange(
    0.05,
    0.96,
    0.05
)

threshold_results = []

best_threshold = 0.50
best_f1 = -1.0

for threshold in thresholds:

    valid_pred = (
        valid_proba >= threshold
    ).astype(int)

    precision = precision_score(
        y_valid,
        valid_pred,
        zero_division=0
    )

    recall = recall_score(
        y_valid,
        valid_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_valid,
        valid_pred,
        zero_division=0
    )

    threshold_results.append(
        {
            "threshold": float(threshold),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
        }
    )

    if f1 > best_f1:
        best_f1 = f1
        best_threshold = float(
            threshold
        )


print(
    f"Best validation threshold: "
    f"{best_threshold:.2f}"
)

print(
    f"Best validation F1: "
    f"{best_f1:.4f}"
)


# ============================================================
# 12. TEST PREDICTION
# ============================================================

print()
print("[11] Evaluating on TEST set...")

test_proba = pipeline.predict_proba(
    X_test
)[:, 1]

test_pred = (
    test_proba >= best_threshold
).astype(int)


# ============================================================
# 13. METRICS
# ============================================================

precision = precision_score(
    y_test,
    test_pred,
    zero_division=0
)

recall = recall_score(
    y_test,
    test_pred,
    zero_division=0
)

f1 = f1_score(
    y_test,
    test_pred,
    zero_division=0
)

pr_auc = average_precision_score(
    y_test,
    test_proba
)

try:
    roc_auc = roc_auc_score(
        y_test,
        test_proba
    )
except ValueError:
    roc_auc = float("nan")


cm = confusion_matrix(
    y_test,
    test_pred
)


# ============================================================
# 14. DISPLAY RESULTS
# ============================================================

print()
print("=" * 70)
print("TEST RESULTS")
print("=" * 70)

print(
    f"Precision : {precision:.4f}"
)

print(
    f"Recall    : {recall:.4f}"
)

print(
    f"F1        : {f1:.4f}"
)

print(
    f"PR-AUC    : {pr_auc:.4f}"
)

print(
    f"ROC-AUC   : {roc_auc:.4f}"
)

print()
print("Confusion matrix:")

print(cm)

print()
print("Classification report:")

print(
    classification_report(
        y_test,
        test_pred,
        digits=4,
        zero_division=0
    )
)


# ============================================================
# 15. SAVE TEST PREDICTIONS
# ============================================================

print()
print("[12] Saving test predictions...")

test_results = df.iloc[
    test_idx
].copy()

test_results[
    "predicted_probability"
] = test_proba

test_results[
    "predicted_incident"
] = test_pred

test_results[
    "prediction_threshold"
] = best_threshold

prediction_file = (
    RESULT_DIR
    / "micross_incident_predictions_15m.csv"
)

test_results.to_csv(
    prediction_file,
    index=False
)

print(
    "Saved:"
)

print(
    prediction_file
)


# ============================================================
# 16. SAVE MODEL
# ============================================================

print()
print("[13] Saving model...")

model_file = (
    MODEL_DIR
    / "micross_incident_predictor_15m.pkl"
)

joblib.dump(
    pipeline,
    model_file
)

print(
    f"Saved:\n{model_file}"
)


# ============================================================
# 17. SAVE THRESHOLD RESULTS
# ============================================================

threshold_file = (
    RESULT_DIR
    / "micross_prediction_thresholds_15m.csv"
)

pd.DataFrame(
    threshold_results
).to_csv(
    threshold_file,
    index=False
)


# ============================================================
# 18. SAVE METADATA
# ============================================================

metadata = {
    "project": "HARDTEC AIOps",
    "dataset": str(DATA_FILE),

    "target": TARGET,
    "horizon_minutes": 15,

    "random_state": RANDOM_STATE,

    "split": {
        "train_ratio": TRAIN_RATIO,
        "validation_ratio": VALID_RATIO,
        "test_ratio": TEST_RATIO,
        "strategy": "chronological"
    },

    "threshold": best_threshold,

    "class_balance": {
        "negative_train": negative_count,
        "positive_train": positive_count,
        "scale_pos_weight": scale_pos_weight
    },

    "metrics": {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "pr_auc": pr_auc,
        "roc_auc": roc_auc
    },

    "features": list(X.columns),

    "numeric_features": numeric_features,

    "categorical_features": categorical_features,

    "leakage_columns_removed": [
        c
        for c in LEAKAGE_COLUMNS
        if c in df.columns
    ],

    "identifier_columns_removed": [
        c
        for c in DROP_COLUMNS
        if c in [
            "fault_id",
            "block_id",
            "window_start",
            "window_end",
            "aligned_fault_start",
            "aligned_fault_end",
            "fault_type",
            "fault_service",
            "timestamp",
        ]
    ]
}

metadata_file = (
    MODEL_DIR
    / "micross_incident_predictor_15m_metadata.json"
)

with open(
    metadata_file,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        metadata,
        f,
        indent=2,
        ensure_ascii=False
    )


# ============================================================
# 19. FINAL SUMMARY
# ============================================================

print()
print("=" * 70)
print("MODEL CREATED SUCCESSFULLY")
print("=" * 70)

print()
print("Model:")
print(model_file)

print()
print("Metadata:")
print(metadata_file)

print()
print("Predictions:")
print(prediction_file)

print()
print("Threshold analysis:")
print(threshold_file)

print()
print("FINAL TEST METRICS")
print("-" * 40)

print(
    f"Precision : {precision:.4f}"
)

print(
    f"Recall    : {recall:.4f}"
)

print(
    f"F1        : {f1:.4f}"
)

print(
    f"PR-AUC    : {pr_auc:.4f}"
)

print(
    f"ROC-AUC   : {roc_auc:.4f}"
)

print()
print("=" * 70)
print("DONE")
print("=" * 70)

