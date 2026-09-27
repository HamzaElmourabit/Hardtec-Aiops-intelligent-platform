
"""
HARDTEC - MicroSS Multisource Risk Model V3.1
==============================================

V3.1 = V3 corrigée pour limiter l'utilisation mémoire.

Ajouts temporels :
    - lag 5m
    - lag 10m
    - lag 15m
    - delta 5m
    - delta 10m
    - delta 15m
    - pct_change 5m
    - pct_change 10m
    - pct_change 15m
    - rolling mean 15m
    - rolling std 15m

IMPORTANT :
    - aucune feature future
    - split temporel 70/15/15
    - labels futurs exclus des features
    - calculs en float32
    - construction des blocs hors DataFrame fragmenté
"""

from pathlib import Path
import gc
import json
import warnings

import joblib
import numpy as np
import pandas as pd

from xgboost import XGBClassifier

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    roc_auc_score,
    confusion_matrix,
)

warnings.filterwarnings("ignore")


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(
    r"C:\Users\khadi\Desktop\hardtec-intelligent-ticketing"
)

FEATURE_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "micross"
    / "micross_multisource_features_5m.csv"
)

LABEL_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "micross"
    / "micross_incident_labels_5m.csv"
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

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


MODEL_FILE = (
    MODEL_DIR
    / "micross_multisource_risk_xgboost_v3.pkl"
)

FEATURES_FILE = (
    MODEL_DIR
    / "micross_multisource_risk_features_v3.json"
)

METRICS_FILE = (
    OUTPUT_DIR
    / "micross_multisource_risk_v3_metrics.json"
)

IMPORTANCE_FILE = (
    OUTPUT_DIR
    / "micross_multisource_risk_v3_feature_importance.csv"
)

CONFUSION_FILE = (
    OUTPUT_DIR
    / "micross_multisource_risk_v3_confusion_matrix.csv"
)

VALIDATION_FILE = (
    OUTPUT_DIR
    / "micross_risk_v3_validation_predictions.csv"
)

TEST_FILE = (
    OUTPUT_DIR
    / "micross_risk_v3_test_predictions.csv"
)


TARGET = "incident_next_15m"

EPSILON = 1e-6

MAX_BASE_METRICS = 115


# ============================================================
# PRINT
# ============================================================

def header(title):

    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


# ============================================================
# SELECT BASE METRICS
# ============================================================

def select_base_metrics(df):

    excluded = {
        "timestamp",
        TARGET,
        "incident_now",
        "incident_next_5m",
        "incident_next_10m",
        "incident_next_15m",
        "incident_next_30m",
        "minutes_to_next_fault",
        "next_fault_timestamp",
        "fault_timestamp",
    }

    numeric = []

    for col in df.columns:

        if col in excluded:
            continue

        if not pd.api.types.is_numeric_dtype(
            df[col]
        ):
            continue

        numeric.append(col)

    # Priorité aux métriques agrégées mean
    mean_columns = [
        c
        for c in numeric
        if "__mean" in c.lower()
        or "_mean" in c.lower()
    ]

    other_columns = [
        c
        for c in numeric
        if c not in mean_columns
    ]

    selected = mean_columns[
        :MAX_BASE_METRICS
    ]

    if len(selected) < MAX_BASE_METRICS:

        needed = (
            MAX_BASE_METRICS
            - len(selected)
        )

        selected.extend(
            other_columns[:needed]
        )

    return selected


# ============================================================
# BUILD TEMPORAL FEATURES WITHOUT DATAFRAME FRAGMENTATION
# ============================================================

def build_temporal_matrix(
    df,
    base_columns,
):

    header(
        "BUILDING TEMPORAL FEATURES"
    )

    n_rows = len(df)

    blocks = []

    feature_names = []

    # --------------------------------------------------------
    # Base values
    # --------------------------------------------------------

    base_block = np.empty(
        (
            n_rows,
            len(base_columns),
        ),
        dtype=np.float32,
    )

    for j, col in enumerate(
        base_columns
    ):

        base_block[:, j] = (
            pd.to_numeric(
                df[col],
                errors="coerce",
            )
            .astype("float32")
            .to_numpy()
        )

    blocks.append(base_block)

    feature_names.extend(
        base_columns
    )

    print(
        f"[INFO] Base feature block: "
        f"{base_block.shape}"
    )

    del base_block
    gc.collect()

    # --------------------------------------------------------
    # Convert base columns to a matrix
    # --------------------------------------------------------

    base_matrix = np.column_stack(
        [
            pd.to_numeric(
                df[col],
                errors="coerce",
            )
            .astype("float32")
            .to_numpy()
            for col in base_columns
        ]
    )

    base_matrix = np.asarray(
        base_matrix,
        dtype=np.float32,
    )

    # --------------------------------------------------------
    # LAGS
    # --------------------------------------------------------

    for lag, suffix in [
        (1, "5m"),
        (2, "10m"),
        (3, "15m"),
    ]:

        print(
            f"[INFO] Building lag {suffix}..."
        )

        lag_block = np.roll(
            base_matrix,
            lag,
            axis=0,
        ).astype(
            np.float32,
            copy=False,
        )

        lag_block[
            :lag,
            :
        ] = np.nan

        blocks.append(
            lag_block.copy()
        )

        feature_names.extend(
            [
                f"{c}__lag_{suffix}"
                for c in base_columns
            ]
        )

        del lag_block
        gc.collect()

    # --------------------------------------------------------
    # DELTAS
    # --------------------------------------------------------

    print(
        "[INFO] Building deltas..."
    )

    delta_blocks = []

    for lag, suffix in [
        (1, "5m"),
        (2, "10m"),
        (3, "15m"),
    ]:

        lag_matrix = np.roll(
            base_matrix,
            lag,
            axis=0,
        ).astype(
            np.float32,
            copy=False,
        )

        lag_matrix[
            :lag,
            :
        ] = np.nan

        delta = (
            base_matrix
            - lag_matrix
        ).astype(
            np.float32
        )

        delta_blocks.append(
            delta
        )

        feature_names.extend(
            [
                f"{c}__delta_{suffix}"
                for c in base_columns
            ]
        )

        del lag_matrix
        gc.collect()

    blocks.extend(
        delta_blocks
    )

    del delta_blocks
    gc.collect()

    # --------------------------------------------------------
    # PERCENT CHANGES
    # --------------------------------------------------------

    print(
        "[INFO] Building percentage changes..."
    )

    for lag, suffix in [
        (1, "5m"),
        (2, "10m"),
        (3, "15m"),
    ]:

        lag_matrix = np.roll(
            base_matrix,
            lag,
            axis=0,
        ).astype(
            np.float32,
            copy=False,
        )

        lag_matrix[
            :lag,
            :
        ] = np.nan

        pct = (
            base_matrix
            - lag_matrix
        ) / (
            np.abs(lag_matrix)
            + EPSILON
        )

        pct = np.clip(
            pct,
            -10.0,
            10.0,
        ).astype(
            np.float32
        )

        pct[
            :lag,
            :
        ] = np.nan

        blocks.append(
            pct
        )

        feature_names.extend(
            [
                f"{c}__pct_change_{suffix}"
                for c in base_columns
            ]
        )

        del lag_matrix
        gc.collect()

    # --------------------------------------------------------
    # ROLLING FEATURES
    # --------------------------------------------------------

    print(
        "[INFO] Building rolling statistics..."
    )

    rolling_mean = np.empty_like(
        base_matrix,
        dtype=np.float32,
    )

    rolling_std = np.empty_like(
        base_matrix,
        dtype=np.float32,
    )

    rolling_mean.fill(
        np.nan
    )

    rolling_std.fill(
        np.nan
    )

    # Calcul manuel par colonne.
    # shift(1) garantit que la valeur actuelle
    # n'entre jamais dans le rolling.
    for j in range(
        base_matrix.shape[1]
    ):

        series = pd.Series(
            base_matrix[:, j],
            dtype="float32",
        )

        historical = series.shift(
            1
        )

        rolling_mean[:, j] = (
            historical
            .rolling(
                window=3,
                min_periods=1,
            )
            .mean()
            .to_numpy(
                dtype=np.float32
            )
        )

        rolling_std[:, j] = (
            historical
            .rolling(
                window=3,
                min_periods=2,
            )
            .std()
            .to_numpy(
                dtype=np.float32
            )
        )

    blocks.append(
        rolling_mean
    )

    feature_names.extend(
        [
            f"{c}__rolling_mean_15m"
            for c in base_columns
        ]
    )

    blocks.append(
        rolling_std
    )

    feature_names.extend(
        [
            f"{c}__rolling_std_15m"
            for c in base_columns
        ]
    )

    del rolling_mean
    del rolling_std
    gc.collect()

    # --------------------------------------------------------
    # Combine all blocks once
    # --------------------------------------------------------

    print(
        "[INFO] Consolidating feature matrix..."
    )

    X = np.column_stack(
        blocks
    ).astype(
        np.float32,
        copy=False,
    )

    del blocks
    del base_matrix
    gc.collect()

    print(
        f"[INFO] Final temporal matrix: "
        f"{X.shape}"
    )

    print(
        f"[INFO] Approx memory: "
        f"{X.nbytes / 1024**2:.1f} MB"
    )

    return X, feature_names


# ============================================================
# THRESHOLD
# ============================================================

def select_threshold(
    y_true,
    probabilities,
):

    candidates = np.arange(
        0.10,
        0.91,
        0.05,
    )

    best_threshold = 0.50
    best_f1 = -1

    rows = []

    for threshold in candidates:

        prediction = (
            probabilities
            >= threshold
        ).astype(int)

        precision = precision_score(
            y_true,
            prediction,
            zero_division=0,
        )

        recall = recall_score(
            y_true,
            prediction,
            zero_division=0,
        )

        f1 = f1_score(
            y_true,
            prediction,
            zero_division=0,
        )

        positive_rate = (
            prediction.mean()
        )

        rows.append(
            {
                "threshold": float(
                    threshold
                ),
                "precision": float(
                    precision
                ),
                "recall": float(
                    recall
                ),
                "f1": float(f1),
                "predicted_positive_rate":
                    float(
                        positive_rate
                    ),
            }
        )

        # Évite les seuils qui déclenchent
        # presque partout.
        if positive_rate > 0.85:
            continue

        if f1 > best_f1:

            best_f1 = f1
            best_threshold = float(
                threshold
            )

    return (
        best_threshold,
        pd.DataFrame(rows),
    )


# ============================================================
# EVALUATION
# ============================================================

def evaluate(
    y_true,
    probabilities,
    threshold,
):

    prediction = (
        probabilities
        >= threshold
    ).astype(int)

    precision = precision_score(
        y_true,
        prediction,
        zero_division=0,
    )

    recall = recall_score(
        y_true,
        prediction,
        zero_division=0,
    )

    f1 = f1_score(
        y_true,
        prediction,
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

        roc_auc = None

    tn, fp, fn, tp = (
        confusion_matrix(
            y_true,
            prediction,
            labels=[0, 1],
        )
        .ravel()
    )

    specificity = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else 0.0
    )

    return {
        "threshold": float(
            threshold
        ),
        "precision": float(
            precision
        ),
        "recall": float(
            recall
        ),
        "f1": float(
            f1
        ),
        "pr_auc": float(
            pr_auc
        ),
        "roc_auc": (
            float(roc_auc)
            if roc_auc is not None
            else None
        ),
        "specificity": float(
            specificity
        ),
        "predicted_positive_rate":
            float(
                prediction.mean()
            ),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


# ============================================================
# MAIN
# ============================================================

def main():

    header(
        "HARDTEC - MICROSS MULTISOURCE RISK MODEL V3.1"
    )

    print(
        "[INFO] Project root:"
    )

    print(
        PROJECT_ROOT
    )

    # ========================================================
    # LOAD
    # ========================================================

    header(
        "LOADING DATA"
    )

    features = pd.read_csv(
        FEATURE_FILE
    )

    labels = pd.read_csv(
        LABEL_FILE
    )

    print(
        f"Features shape: "
        f"{features.shape}"
    )

    print(
        f"Labels shape: "
        f"{labels.shape}"
    )

    # ========================================================
    # TIMESTAMP
    # ========================================================

    features[
        "timestamp"
    ] = pd.to_datetime(
        features[
            "timestamp"
        ],
        errors="coerce",
    )

    labels[
        "timestamp"
    ] = pd.to_datetime(
        labels[
            "timestamp"
        ],
        errors="coerce",
    )

    features = (
        features
        .dropna(
            subset=["timestamp"]
        )
        .sort_values(
            "timestamp"
        )
    )

    labels = (
        labels
        .dropna(
            subset=["timestamp"]
        )
        .sort_values(
            "timestamp"
        )
    )

    # ========================================================
    # MERGE
    # ========================================================

    header(
        "MERGING FEATURES + LABELS"
    )

    if TARGET in features.columns:

        df = features.copy()

    else:

        labels_small = labels[
            [
                "timestamp",
                TARGET,
            ]
        ].copy()

        df = features.merge(
            labels_small,
            on="timestamp",
            how="left",
        )

    df[TARGET] = (
        pd.to_numeric(
            df[TARGET],
            errors="coerce",
        )
        .fillna(0)
        .astype("int8")
    )

    df = (
        df
        .sort_values(
            "timestamp"
        )
        .drop_duplicates(
            subset=["timestamp"],
            keep="last",
        )
        .reset_index(drop=True)
    )

    print(
        f"Final dataset shape: "
        f"{df.shape}"
    )

    print(
        f"Positive rows: "
        f"{df[TARGET].sum():,}"
    )

    print(
        f"Positive rate: "
        f"{df[TARGET].mean() * 100:.2f}%"
    )

    # ========================================================
    # SELECT BASE METRICS
    # ========================================================

    header(
        "SELECTING BASE METRICS"
    )

    base_metrics = select_base_metrics(
        df
    )

    print(
        f"Base metrics selected: "
        f"{len(base_metrics)}"
    )

    # ========================================================
    # TEMPORAL MATRIX
    # ========================================================

    X_matrix, feature_names = (
        build_temporal_matrix(
            df,
            base_metrics,
        )
    )

    y = df[
        TARGET
    ].to_numpy(
        dtype=np.int8
    )

    timestamps = df[
        "timestamp"
    ].to_numpy()

    print()
    print(
        f"Training matrix: "
        f"{X_matrix.shape}"
    )

    # ========================================================
    # REMOVE INF
    # ========================================================

    X_matrix[
        ~np.isfinite(X_matrix)
    ] = np.nan

    # ========================================================
    # TEMPORAL SPLIT
    # ========================================================

    header(
        "TEMPORAL TRAIN / VALIDATION / TEST SPLIT"
    )

    n = len(X_matrix)

    train_end = int(
        n * 0.70
    )

    val_end = int(
        n * 0.85
    )

    X_train = X_matrix[
        :train_end
    ]

    y_train = y[
        :train_end
    ]

    X_val = X_matrix[
        train_end:val_end
    ]

    y_val = y[
        train_end:val_end
    ]

    X_test = X_matrix[
        val_end:
    ]

    y_test = y[
        val_end:
    ]

    ts_val = timestamps[
        train_end:val_end
    ]

    ts_test = timestamps[
        val_end:
    ]

    print(
        f"Train: "
        f"{len(X_train):,}"
    )

    print(
        f"Validation: "
        f"{len(X_val):,}"
    )

    print(
        f"Test: "
        f"{len(X_test):,}"
    )

    print()
    print(
        f"Train positive rate: "
        f"{y_train.mean() * 100:.2f}%"
    )

    print(
        f"Validation positive rate: "
        f"{y_val.mean() * 100:.2f}%"
    )

    print(
        f"Test positive rate: "
        f"{y_test.mean() * 100:.2f}%"
    )

    # ========================================================
    # CLASS WEIGHT
    # ========================================================

    positives = y_train.sum()

    negatives = (
        len(y_train)
        - positives
    )

    scale_pos_weight = (
        negatives / positives
    )

    print()
    print(
        f"scale_pos_weight: "
        f"{scale_pos_weight:.4f}"
    )

    # ========================================================
    # FREE MEMORY
    # ========================================================

    del X_matrix
    gc.collect()

    # ========================================================
    # XGBOOST
    # ========================================================

    header(
        "TRAINING XGBOOST V3"
    )

    print(
        f"Training features: "
        f"{len(feature_names):,}"
    )

    model = XGBClassifier(
        n_estimators=350,
        max_depth=5,
        learning_rate=0.04,
        subsample=0.80,
        colsample_bytree=0.70,
        min_child_weight=5,
        gamma=0.10,
        reg_alpha=0.10,
        reg_lambda=1.50,
        scale_pos_weight=scale_pos_weight,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        random_state=42,
        n_jobs=4,
    )

    model.fit(
        X_train,
        y_train,
        eval_set=[
            (
                X_val,
                y_val,
            )
        ],
        verbose=False,
    )

    print(
        "[INFO] XGBoost training completed."
    )

    # ========================================================
    # VALIDATION
    # ========================================================

    header(
        "VALIDATION"
    )

    val_probabilities = (
        model
        .predict_proba(
            X_val
        )[:, 1]
    )

    threshold, threshold_table = (
        select_threshold(
            y_val,
            val_probabilities,
        )
    )

    validation_metrics = evaluate(
        y_val,
        val_probabilities,
        threshold,
    )

    print(
        f"Selected threshold: "
        f"{threshold:.2f}"
    )

    print(
        f"Validation precision: "
        f"{validation_metrics['precision']:.4f}"
    )

    print(
        f"Validation recall: "
        f"{validation_metrics['recall']:.4f}"
    )

    print(
        f"Validation F1: "
        f"{validation_metrics['f1']:.4f}"
    )

    print(
        f"Validation PR-AUC: "
        f"{validation_metrics['pr_auc']:.4f}"
    )

    print(
        f"Validation ROC-AUC: "
        f"{validation_metrics['roc_auc']:.4f}"
    )

    print(
        f"Validation specificity: "
        f"{validation_metrics['specificity']:.4f}"
    )

    print(
        f"Validation predicted positive rate: "
        f"{validation_metrics['predicted_positive_rate']:.4f}"
    )

    # ========================================================
    # TEST
    # ========================================================

    header(
        "FINAL TEST"
    )

    test_probabilities = (
        model
        .predict_proba(
            X_test
        )[:, 1]
    )

    test_metrics = evaluate(
        y_test,
        test_probabilities,
        threshold,
    )

    print(
        f"Test threshold: "
        f"{threshold:.2f}"
    )

    print(
        f"Test precision: "
        f"{test_metrics['precision']:.4f}"
    )

    print(
        f"Test recall: "
        f"{test_metrics['recall']:.4f}"
    )

    print(
        f"Test F1: "
        f"{test_metrics['f1']:.4f}"
    )

    print(
        f"Test PR-AUC: "
        f"{test_metrics['pr_auc']:.4f}"
    )

    print(
        f"Test ROC-AUC: "
        f"{test_metrics['roc_auc']:.4f}"
    )

    print(
        f"Test specificity: "
        f"{test_metrics['specificity']:.4f}"
    )

    print(
        f"Test predicted positive rate: "
        f"{test_metrics['predicted_positive_rate']:.4f}"
    )

    print()
    print(
        "Confusion matrix:"
    )

    print(
        f"TN = {test_metrics['tn']}"
    )

    print(
        f"FP = {test_metrics['fp']}"
    )

    print(
        f"FN = {test_metrics['fn']}"
    )

    print(
        f"TP = {test_metrics['tp']}"
    )

    # ========================================================
    # SAVE MODEL
    # ========================================================

    header(
        "SAVING OUTPUTS"
    )

    joblib.dump(
        model,
        MODEL_FILE,
    )

    # ========================================================
    # FEATURE METADATA
    # ========================================================

    feature_metadata = {
        "version": "V3.1",
        "target": TARGET,
        "base_metrics": base_metrics,
        "n_base_metrics": len(
            base_metrics
        ),
        "n_features": len(
            feature_names
        ),
        "features": feature_names,
        "threshold": float(
            threshold
        ),
        "future_leakage": False,
        "temporal_split": {
            "train": 0.70,
            "validation": 0.15,
            "test": 0.15,
        },
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
            ensure_ascii=False,
        )

    # ========================================================
    # METRICS JSON
    # ========================================================

    metrics_output = {
        "version": "V3.1",
        "dataset_rows": int(n),
        "positive_rows": int(
            y.sum()
        ),
        "positive_rate": float(
            y.mean()
        ),
        "base_metrics": int(
            len(base_metrics)
        ),
        "training_features": int(
            len(feature_names)
        ),
        "scale_pos_weight": float(
            scale_pos_weight
        ),
        "validation": validation_metrics,
        "test": test_metrics,
        "threshold": float(
            threshold
        ),
        "temporal_split": {
            "train_rows": int(
                len(X_train)
            ),
            "validation_rows": int(
                len(X_val)
            ),
            "test_rows": int(
                len(X_test)
            ),
        },
        "feature_engineering": {
            "lags": [
                "5m",
                "10m",
                "15m",
            ],
            "deltas": [
                "5m",
                "10m",
                "15m",
            ],
            "pct_changes": [
                "5m",
                "10m",
                "15m",
            ],
            "rolling_window": "15m",
            "rolling_statistics": [
                "mean",
                "std",
            ],
            "future_leakage": False,
        },
    }

    with open(
        METRICS_FILE,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            metrics_output,
            f,
            indent=2,
            ensure_ascii=False,
        )

    # ========================================================
    # FEATURE IMPORTANCE
    # ========================================================

    importance_df = pd.DataFrame(
        {
            "feature": feature_names,
            "importance":
                model.feature_importances_,
        }
    ).sort_values(
        "importance",
        ascending=False,
    )

    importance_df.to_csv(
        IMPORTANCE_FILE,
        index=False,
        encoding="utf-8",
    )

    print()
    print(
        "Top 20 features:"
    )

    print(
        importance_df
        .head(20)
        .to_string(
            index=False
        )
    )

    # ========================================================
    # CONFUSION MATRIX
    # ========================================================

    confusion_df = pd.DataFrame(
        [
            [
                test_metrics["tn"],
                test_metrics["fp"],
            ],
            [
                test_metrics["fn"],
                test_metrics["tp"],
            ],
        ],
        index=[
            "Actual 0",
            "Actual 1",
        ],
        columns=[
            "Predicted 0",
            "Predicted 1",
        ],
    )

    confusion_df.to_csv(
        CONFUSION_FILE,
        encoding="utf-8",
    )

    # ========================================================
    # VALIDATION PREDICTIONS
    # ========================================================

    validation_predictions = (
        pd.DataFrame(
            {
                "timestamp": ts_val,
                "actual": y_val,
                "risk_probability":
                    val_probabilities,
                "prediction": (
                    val_probabilities
                    >= threshold
                ).astype(int),
            }
        )
    )

    validation_predictions.to_csv(
        VALIDATION_FILE,
        index=False,
        encoding="utf-8",
    )

    # ========================================================
    # TEST PREDICTIONS
    # ========================================================

    test_predictions = (
        pd.DataFrame(
            {
                "timestamp": ts_test,
                "actual": y_test,
                "risk_probability":
                    test_probabilities,
                "prediction": (
                    test_probabilities
                    >= threshold
                ).astype(int),
            }
        )
    )

    test_predictions.to_csv(
        TEST_FILE,
        index=False,
        encoding="utf-8",
    )

    # ========================================================
    # CLEAN
    # ========================================================

    del X_train
    del X_val
    del X_test
    del model

    gc.collect()

    # ========================================================
    # FINAL
    # ========================================================

    header(
        "MICROSS MULTISOURCE V3.1 COMPLETE"
    )

    print(
        f"Base metrics: "
        f"{len(base_metrics):,}"
    )

    print(
        f"Temporal/model features: "
        f"{len(feature_names):,}"
    )

    print(
        f"Dataset rows: "
        f"{n:,}"
    )

    print()
    print(
        "TEST RESULTS"
    )

    print(
        f"Precision: "
        f"{test_metrics['precision']:.4f}"
    )

    print(
        f"Recall: "
        f"{test_metrics['recall']:.4f}"
    )

    print(
        f"F1: "
        f"{test_metrics['f1']:.4f}"
    )

    print(
        f"PR-AUC: "
        f"{test_metrics['pr_auc']:.4f}"
    )

    print(
        f"ROC-AUC: "
        f"{test_metrics['roc_auc']:.4f}"
    )

    print(
        f"Specificity: "
        f"{test_metrics['specificity']:.4f}"
    )

    print()
    print(
        "Model:"
    )

    print(
        MODEL_FILE
    )


if __name__ == "__main__":
    main()

