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

OUTPUT_DIR = (
    PROJECT_ROOT
    / "models"
    / "micross"
)

PREDICTION_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "micross"
)

TARGET = "incident_next_15m"

RANDOM_STATE = 42


# ============================================================
# TEMPORAL SPLIT
# ============================================================

TRAIN_RATIO = 0.70
VALID_RATIO = 0.15
TEST_RATIO = 0.15


# ============================================================
# XGBOOST CONFIGURATION
# ============================================================

XGB_PARAMS = {
    "n_estimators": 300,
    "max_depth": 5,
    "learning_rate": 0.05,
    "subsample": 0.80,
    "colsample_bytree": 0.80,
    "objective": "binary:logistic",
    "eval_metric": "aucpr",
    "tree_method": "hist",
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}


# ============================================================
# HELPERS
# ============================================================


def ensure_directories():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PREDICTION_DIR.mkdir(parents=True, exist_ok=True)


def print_section(title):
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


def find_timestamp_column(df):
    candidates = [
        "timestamp",
        "datetime",
        "date",
        "time",
    ]

    for col in candidates:
        if col in df.columns:
            return col

    raise ValueError(
        "Impossible de trouver une colonne timestamp."
    )


def find_target_column(df):
    if TARGET in df.columns:
        return TARGET

    raise ValueError(
        f"La colonne cible '{TARGET}' est absente du dataset."
    )


def load_data():
    print_section("LOADING MULTISOURCE DATA")

    print(f"[INFO] Feature file:")
    print(FEATURE_FILE)

    print(f"[INFO] Label file:")
    print(LABEL_FILE)

    if not FEATURE_FILE.exists():
        raise FileNotFoundError(
            f"Feature file introuvable: {FEATURE_FILE}"
        )

    if not LABEL_FILE.exists():
        raise FileNotFoundError(
            f"Label file introuvable: {LABEL_FILE}"
        )

    features = pd.read_csv(FEATURE_FILE)
    labels = pd.read_csv(LABEL_FILE)

    print()
    print(f"[INFO] Features shape: {features.shape}")
    print(f"[INFO] Labels shape:   {labels.shape}")

    feature_timestamp = find_timestamp_column(features)
    label_timestamp = find_timestamp_column(labels)

    features[feature_timestamp] = pd.to_datetime(
        features[feature_timestamp],
        errors="coerce",
    )

    labels[label_timestamp] = pd.to_datetime(
        labels[label_timestamp],
        errors="coerce",
    )

    features = features.dropna(
        subset=[feature_timestamp]
    ).copy()

    labels = labels.dropna(
        subset=[label_timestamp]
    ).copy()

    features = features.sort_values(
        feature_timestamp
    ).reset_index(drop=True)

    labels = labels.sort_values(
        label_timestamp
    ).reset_index(drop=True)

    return (
        features,
        labels,
        feature_timestamp,
        label_timestamp,
    )


# ============================================================
# BUILD TARGET
# ============================================================


def prepare_dataset(
    features,
    labels,
    feature_timestamp,
    label_timestamp,
):
    print_section("BUILDING MODEL DATASET")

    # --------------------------------------------------------
    # If the target is already in the feature file, use it.
    # Otherwise merge it from the label file.
    # --------------------------------------------------------

    if TARGET in features.columns:
        print(
            f"[INFO] Target '{TARGET}' already exists "
            "in feature matrix."
        )

        df = features.copy()

    else:
        print(
            f"[INFO] Target '{TARGET}' not found in "
            "feature matrix."
        )

        required_label_columns = [
            label_timestamp,
            TARGET,
        ]

        missing = [
            c
            for c in required_label_columns
            if c not in labels.columns
        ]

        if missing:
            raise ValueError(
                f"Colonnes manquantes dans labels: {missing}"
            )

        labels_small = labels[
            required_label_columns
        ].copy()

        labels_small = labels_small.rename(
            columns={
                label_timestamp: feature_timestamp
            }
        )

        df = features.merge(
            labels_small,
            on=feature_timestamp,
            how="left",
        )

    # --------------------------------------------------------
    # Ensure target is binary numeric
    # --------------------------------------------------------

    df[TARGET] = pd.to_numeric(
        df[TARGET],
        errors="coerce",
    )

    df[TARGET] = df[TARGET].fillna(0).astype(int)

    df[TARGET] = (
        df[TARGET]
        .clip(lower=0, upper=1)
    )

    df = df.sort_values(
        feature_timestamp
    ).reset_index(drop=True)

    print(f"[INFO] Final dataset shape: {df.shape}")

    print(
        f"[INFO] Positive rows: "
        f"{df[TARGET].sum():,}"
    )

    print(
        f"[INFO] Positive rate: "
        f"{df[TARGET].mean() * 100:.2f}%"
    )

    return df


# ============================================================
# FEATURE SAFETY
# ============================================================


def identify_safe_features(
    df,
    timestamp_column,
):
    print_section("FEATURE SELECTION")

    forbidden_exact = {
        timestamp_column,
        TARGET,

        # Current/future incident labels
        "incident_now",
        "incident_next_5m",
        "incident_next_10m",
        "incident_next_15m",
        "incident_next_30m",

        # Future information
        "minutes_to_next_fault",
        "next_fault_timestamp",
        "fault_timestamp",
        "fault_start",
        "fault_end",

        # Explicit labels
        "fault",
        "failure",
        "failure_count",
        "failure_rate",
        "label",
    }

    forbidden_keywords = [
        "next_fault",
        "future_fault",
        "minutes_to_next",
        "fault_timestamp",
        "incident_next",
    ]

    numeric_candidates = []

    for col in df.columns:

        if col in forbidden_exact:
            continue

        lower_col = col.lower()

        if any(
            keyword in lower_col
            for keyword in forbidden_keywords
        ):
            continue

        if pd.api.types.is_numeric_dtype(df[col]):
            numeric_candidates.append(col)

    print(
        f"[INFO] Numeric candidate features: "
        f"{len(numeric_candidates)}"
    )

    if len(numeric_candidates) == 0:
        raise ValueError(
            "Aucune feature numérique utilisable."
        )

    # --------------------------------------------------------
    # Identify groups
    # --------------------------------------------------------

    metric_features = [
        c
        for c in numeric_candidates
        if c.startswith("metric_")
    ]

    mean_features = [
        c
        for c in numeric_candidates
        if c.startswith("mean_")
    ]

    count_features = [
        c
        for c in numeric_candidates
        if c.startswith("count_")
    ]

    anomaly_features = [
        c
        for c in numeric_candidates
        if "anomaly" in c.lower()
    ]

    temporal_features = [
        c
        for c in numeric_candidates
        if c.lower()
        in {
            "hour",
            "day_of_week",
            "day_of_month",
            "month",
            "is_weekend",
        }
    ]

    print()
    print(
        f"[INFO] Metric features:      "
        f"{len(metric_features)}"
    )

    print(
        f"[INFO] Mean features:        "
        f"{len(mean_features)}"
    )

    print(
        f"[INFO] Count features:       "
        f"{len(count_features)}"
    )

    print(
        f"[INFO] Anomaly features:     "
        f"{len(anomaly_features)}"
    )

    print(
        f"[INFO] Temporal features:    "
        f"{len(temporal_features)}"
    )

    return numeric_candidates


# ============================================================
# TEMPORAL SPLIT
# ============================================================


def temporal_split(
    df,
    timestamp_column,
):
    print_section("TEMPORAL TRAIN / VALIDATION / TEST SPLIT")

    df = df.sort_values(
        timestamp_column
    ).reset_index(drop=True)

    n = len(df)

    train_end = int(
        n * TRAIN_RATIO
    )

    valid_end = int(
        n * (TRAIN_RATIO + VALID_RATIO)
    )

    train_df = df.iloc[
        :train_end
    ].copy()

    valid_df = df.iloc[
        train_end:valid_end
    ].copy()

    test_df = df.iloc[
        valid_end:
    ].copy()

    print(
        f"[INFO] Train rows:      {len(train_df):,}"
    )

    print(
        f"[INFO] Validation rows: {len(valid_df):,}"
    )

    print(
        f"[INFO] Test rows:       {len(test_df):,}"
    )

    print()

    print(
        "[INFO] Train period:"
    )

    print(
        train_df[timestamp_column].min()
    )

    print(
        train_df[timestamp_column].max()
    )

    print()

    print(
        "[INFO] Validation period:"
    )

    print(
        valid_df[timestamp_column].min()
    )

    print(
        valid_df[timestamp_column].max()
    )

    print()

    print(
        "[INFO] Test period:"
    )

    print(
        test_df[timestamp_column].min()
    )

    print(
        test_df[timestamp_column].max()
    )

    return (
        train_df,
        valid_df,
        test_df,
    )


# ============================================================
# CLEAN FEATURES
# ============================================================


def prepare_X_y(
    df,
    feature_columns,
):
    X = df[
        feature_columns
    ].copy()

    y = df[
        TARGET
    ].astype(int)

    # Replace infinite values
    X = X.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    # XGBoost handles NaN natively.
    # We deliberately do NOT perform global imputation,
    # because that could introduce future information.

    return X, y


# ============================================================
# THRESHOLD OPTIMIZATION
# ============================================================


def find_best_threshold(
    y_true,
    probabilities,
):
    best_threshold = 0.50
    best_f1 = -1.0

    results = []

    thresholds = np.arange(
        0.10,
        0.91,
        0.01,
    )

    for threshold in thresholds:

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

        results.append(
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

    results_df = pd.DataFrame(
        results
    )

    return (
        best_threshold,
        best_f1,
        results_df,
    )


# ============================================================
# EVALUATION
# ============================================================


def evaluate_model(
    model,
    X,
    y,
    threshold,
    split_name,
):
    print_section(
        f"EVALUATION - {split_name.upper()}"
    )

    probabilities = model.predict_proba(
        X
    )[:, 1]

    predictions = (
        probabilities >= threshold
    ).astype(int)

    precision = precision_score(
        y,
        predictions,
        zero_division=0,
    )

    recall = recall_score(
        y,
        predictions,
        zero_division=0,
    )

    f1 = f1_score(
        y,
        predictions,
        zero_division=0,
    )

    pr_auc = average_precision_score(
        y,
        probabilities,
    )

    try:
        roc_auc = roc_auc_score(
            y,
            probabilities,
        )
    except ValueError:
        roc_auc = np.nan

    tn, fp, fn, tp = confusion_matrix(
        y,
        predictions,
        labels=[0, 1],
    ).ravel()

    specificity = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else 0.0
    )

    predicted_positive_rate = (
        predictions.mean()
    )

    print(
        f"Threshold:             {threshold:.2f}"
    )

    print(
        f"Precision:             {precision:.4f}"
    )

    print(
        f"Recall:                {recall:.4f}"
    )

    print(
        f"F1:                    {f1:.4f}"
    )

    print(
        f"PR-AUC:                {pr_auc:.4f}"
    )

    print(
        f"ROC-AUC:               {roc_auc:.4f}"
    )

    print(
        f"Specificity:           {specificity:.4f}"
    )

    print(
        f"Actual positives:      {int(y.sum()):,}"
    )

    print(
        f"Predicted positives:   {int(predictions.sum()):,}"
    )

    print(
        f"Predicted positive %:  "
        f"{predicted_positive_rate * 100:.2f}%"
    )

    print()
    print(
        "Confusion matrix:"
    )

    print(
        f"TN={tn:,}  FP={fp:,}"
    )

    print(
        f"FN={fn:,}  TP={tp:,}"
    )

    print()
    print(
        classification_report(
            y,
            predictions,
            target_names=[
                "No incident",
                "Incident",
            ],
            zero_division=0,
        )
    )

    metrics = {
        "split": split_name,
        "threshold": float(threshold),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "pr_auc": float(pr_auc),
        "roc_auc": (
            None
            if np.isnan(roc_auc)
            else float(roc_auc)
        ),
        "specificity": float(
            specificity
        ),
        "actual_positive_count": int(
            y.sum()
        ),
        "predicted_positive_count": int(
            predictions.sum()
        ),
        "predicted_positive_rate": float(
            predicted_positive_rate
        ),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }

    return (
        metrics,
        probabilities,
        predictions,
    )


# ============================================================
# FEATURE IMPORTANCE
# ============================================================


def save_feature_importance(
    model,
    feature_columns,
):
    importance = model.feature_importances_

    importance_df = pd.DataFrame(
        {
            "feature": feature_columns,
            "importance": importance,
        }
    ).sort_values(
        "importance",
        ascending=False,
    )

    output_file = (
        PREDICTION_DIR
        / "micross_multisource_risk_v2_feature_importance.csv"
    )

    importance_df.to_csv(
        output_file,
        index=False,
    )

    print(
        f"[INFO] Feature importance saved:"
    )

    print(output_file)

    print()
    print(
        "Top 20 features:"
    )

    print(
        importance_df.head(20).to_string(
            index=False
        )
    )

    return importance_df


# ============================================================
# SAVE PREDICTIONS
# ============================================================


def save_predictions(
    df,
    timestamp_column,
    probabilities,
    predictions,
    threshold,
    split_name,
):
    output = pd.DataFrame(
        {
            timestamp_column: df[
                timestamp_column
            ].values,

            TARGET: df[
                TARGET
            ].values,

            "risk_probability": probabilities,

            "risk_prediction": predictions,

            "threshold": threshold,

            "split": split_name,
        }
    )

    output_file = (
        PREDICTION_DIR
        / f"micross_risk_v2_{split_name}_predictions.csv"
    )

    output.to_csv(
        output_file,
        index=False,
    )

    print(
        f"[INFO] Predictions saved:"
    )

    print(output_file)

    return output


# ============================================================
# MAIN
# ============================================================


def main():

    print()
    print(
        "=" * 72
    )

    print(
        "MICROSS MULTISOURCE RISK MODEL V2"
    )

    print(
        "Real metric matrix + temporal incident prediction"
    )

    print(
        "=" * 72
    )

    ensure_directories()

    # --------------------------------------------------------
    # LOAD
    # --------------------------------------------------------

    (
        features,
        labels,
        feature_timestamp,
        label_timestamp,
    ) = load_data()

    # --------------------------------------------------------
    # PREPARE DATASET
    # --------------------------------------------------------

    df = prepare_dataset(
        features,
        labels,
        feature_timestamp,
        label_timestamp,
    )

    # --------------------------------------------------------
    # FEATURE SELECTION
    # --------------------------------------------------------

    feature_columns = identify_safe_features(
        df,
        feature_timestamp,
    )

    # --------------------------------------------------------
    # TEMPORAL SPLIT
    # --------------------------------------------------------

    (
        train_df,
        valid_df,
        test_df,
    ) = temporal_split(
        df,
        feature_timestamp,
    )

    # --------------------------------------------------------
    # X / Y
    # --------------------------------------------------------

    X_train, y_train = prepare_X_y(
        train_df,
        feature_columns,
    )

    X_valid, y_valid = prepare_X_y(
        valid_df,
        feature_columns,
    )

    X_test, y_test = prepare_X_y(
        test_df,
        feature_columns,
    )

    print_section("DATASET BALANCE")

    print(
        f"Train negative: "
        f"{int((y_train == 0).sum()):,}"
    )

    print(
        f"Train positive: "
        f"{int((y_train == 1).sum()):,}"
    )

    print(
        f"Train positive rate: "
        f"{y_train.mean() * 100:.2f}%"
    )

    print()

    print(
        f"Validation positive rate: "
        f"{y_valid.mean() * 100:.2f}%"
    )

    print(
        f"Test positive rate: "
        f"{y_test.mean() * 100:.2f}%"
    )

    # --------------------------------------------------------
    # CLASS WEIGHT
    # --------------------------------------------------------

    negative_count = (
        y_train == 0
    ).sum()

    positive_count = (
        y_train == 1
    ).sum()

    if positive_count == 0:
        raise ValueError(
            "Aucun événement positif dans le train."
        )

    scale_pos_weight = (
        negative_count
        / positive_count
    )

    print()
    print(
        f"[INFO] scale_pos_weight: "
        f"{scale_pos_weight:.4f}"
    )

    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    model_params = XGB_PARAMS.copy()

    model_params[
        "scale_pos_weight"
    ] = scale_pos_weight

    print_section("TRAINING XGBOOST V2")

    print(
        f"[INFO] Training features: "
        f"{len(feature_columns)}"
    )

    print(
        f"[INFO] Training rows: "
        f"{len(X_train):,}"
    )

    print(
        "[INFO] Model parameters:"
    )

    for key, value in model_params.items():
        print(
            f"  {key}: {value}"
        )

    model = XGBClassifier(
        **model_params
    )

    model.fit(
        X_train,
        y_train,
        eval_set=[
            (
                X_valid,
                y_valid,
            )
        ],
        verbose=False,
    )

    print(
        "[INFO] Training completed."
    )

    # --------------------------------------------------------
    # VALIDATION THRESHOLD
    # --------------------------------------------------------

    print_section(
        "VALIDATION THRESHOLD SELECTION"
    )

    valid_probabilities = (
        model.predict_proba(
            X_valid
        )[:, 1]
    )

    (
        best_threshold,
        best_f1,
        threshold_results,
    ) = find_best_threshold(
        y_valid,
        valid_probabilities,
    )

    threshold_file = (
        PREDICTION_DIR
        / "micross_risk_v2_threshold_search.csv"
    )

    threshold_results.to_csv(
        threshold_file,
        index=False,
    )

    print(
        f"[INFO] Best threshold: "
        f"{best_threshold:.2f}"
    )

    print(
        f"[INFO] Validation F1: "
        f"{best_f1:.4f}"
    )

    # --------------------------------------------------------
    # VALIDATION EVALUATION
    # --------------------------------------------------------

    (
        validation_metrics,
        _,
        _,
    ) = evaluate_model(
        model,
        X_valid,
        y_valid,
        best_threshold,
        "validation",
    )

    # --------------------------------------------------------
    # TEST EVALUATION
    # --------------------------------------------------------

    (
        test_metrics,
        test_probabilities,
        test_predictions,
    ) = evaluate_model(
        model,
        X_test,
        y_test,
        best_threshold,
        "test",
    )

    # --------------------------------------------------------
    # SAVE MODEL
    # --------------------------------------------------------

    model_file = (
        OUTPUT_DIR
        / "micross_multisource_risk_xgboost_v2.pkl"
    )

    joblib.dump(
        model,
        model_file,
    )

    print(
        f"[INFO] Model saved:"
    )

    print(model_file)

    # --------------------------------------------------------
    # SAVE FEATURE LIST
    # --------------------------------------------------------

    feature_file = (
        OUTPUT_DIR
        / "micross_multisource_risk_features_v2.json"
    )

    feature_metadata = {
        "version": "v2",
        "target": TARGET,
        "n_features": len(
            feature_columns
        ),
        "features": feature_columns,
        "train_ratio": TRAIN_RATIO,
        "validation_ratio": VALID_RATIO,
        "test_ratio": TEST_RATIO,
        "scale_pos_weight": float(
            scale_pos_weight
        ),
        "threshold": float(
            best_threshold
        ),
    }

    with open(
        feature_file,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            feature_metadata,
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(
        f"[INFO] Feature list saved:"
    )

    print(feature_file)

    # --------------------------------------------------------
    # FEATURE IMPORTANCE
    # --------------------------------------------------------

    importance_df = (
        save_feature_importance(
            model,
            feature_columns,
        )
    )

    # --------------------------------------------------------
    # SAVE VALIDATION PREDICTIONS
    # --------------------------------------------------------

    save_predictions(
        valid_df,
        feature_timestamp,
        valid_probabilities,
        (
            valid_probabilities
            >= best_threshold
        ).astype(int),
        best_threshold,
        "validation",
    )

    # --------------------------------------------------------
    # SAVE TEST PREDICTIONS
    # --------------------------------------------------------

    save_predictions(
        test_df,
        feature_timestamp,
        test_probabilities,
        test_predictions,
        best_threshold,
        "test",
    )

    # --------------------------------------------------------
    # CONFUSION MATRIX
    # --------------------------------------------------------

    tn, fp, fn, tp = confusion_matrix(
        y_test,
        test_predictions,
        labels=[0, 1],
    ).ravel()

    confusion_df = pd.DataFrame(
        [
            {
                "split": "test",
                "TN": int(tn),
                "FP": int(fp),
                "FN": int(fn),
                "TP": int(tp),
            }
        ]
    )

    confusion_file = (
        PREDICTION_DIR
        / "micross_multisource_risk_v2_confusion_matrix.csv"
    )

    confusion_df.to_csv(
        confusion_file,
        index=False,
    )

    # --------------------------------------------------------
    # METRICS JSON
    # --------------------------------------------------------

    metrics = {
        "version": "v2",
        "dataset_rows": int(len(df)),
        "feature_count": int(
            len(feature_columns)
        ),
        "target": TARGET,
        "target_positive_count": int(
            df[TARGET].sum()
        ),
        "target_positive_rate": float(
            df[TARGET].mean()
        ),
        "scale_pos_weight": float(
            scale_pos_weight
        ),
        "threshold": float(
            best_threshold
        ),
        "validation": validation_metrics,
        "test": test_metrics,
    }

    metrics_file = (
        PREDICTION_DIR
        / "micross_multisource_risk_v2_metrics.json"
    )

    with open(
        metrics_file,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            metrics,
            f,
            indent=2,
            ensure_ascii=False,
        )

    # --------------------------------------------------------
    # FINAL SUMMARY
    # --------------------------------------------------------

    print_section(
        "MICROSS MULTISOURCE RISK MODEL V2 COMPLETE"
    )

    print(
        f"Features used:        {len(feature_columns)}"
    )

    print(
        f"Train rows:            {len(train_df):,}"
    )

    print(
        f"Validation rows:       {len(valid_df):,}"
    )

    print(
        f"Test rows:             {len(test_df):,}"
    )

    print()

    print(
        "VALIDATION"
    )

    print(
        f"  Threshold:            "
        f"{validation_metrics['threshold']:.2f}"
    )

    print(
        f"  Precision:            "
        f"{validation_metrics['precision']:.4f}"
    )

    print(
        f"  Recall:               "
        f"{validation_metrics['recall']:.4f}"
    )

    print(
        f"  F1:                   "
        f"{validation_metrics['f1']:.4f}"
    )

    print(
        f"  PR-AUC:               "
        f"{validation_metrics['pr_auc']:.4f}"
    )

    print(
        f"  ROC-AUC:              "
        f"{validation_metrics['roc_auc']}"
    )

    print()

    print(
        "TEST"
    )

    print(
        f"  Threshold:            "
        f"{test_metrics['threshold']:.2f}"
    )

    print(
        f"  Precision:            "
        f"{test_metrics['precision']:.4f}"
    )

    print(
        f"  Recall:               "
        f"{test_metrics['recall']:.4f}"
    )

    print(
        f"  F1:                   "
        f"{test_metrics['f1']:.4f}"
    )

    print(
        f"  PR-AUC:               "
        f"{test_metrics['pr_auc']:.4f}"
    )

    print(
        f"  ROC-AUC:              "
        f"{test_metrics['roc_auc']}"
    )

    print(
        f"  Specificity:          "
        f"{test_metrics['specificity']:.4f}"
    )

    print()

    print(
        "OUTPUTS"
    )

    print(model_file)
    print(feature_file)
    print(confusion_file)
    print(metrics_file)

    print()
    print(
        "=" * 72
    )
    print(
        "DONE"
    )
    print(
        "=" * 72
    )


if __name__ == "__main__":
    main()