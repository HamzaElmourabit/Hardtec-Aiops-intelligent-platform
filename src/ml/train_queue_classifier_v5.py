"""
HARDTEC - Queue Classifier V5 Experiment

Purpose:
    Compare multiple queue-classification strategies on the SAME
    train/test split.

Variants:
    V5-A : Word TF-IDF (1,2), no class weighting
    V5-B : Word TF-IDF (1,2), class_weight="balanced"
    V5-C : Word + Character TF-IDF, class_weight="balanced"
    V5-D : Word TF-IDF + Type, class_weight="balanced"

Important:
    - Does NOT modify ticket_queue_model_v4.pkl
    - Does NOT modify predictor.py
    - Uses the same 80/20 split for all variants
    - Saves only experiment results
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import mlflow

from scipy.sparse import hstack

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

try:
    from src.ml.tracking import (
        end_training_run,
        log_classification_result,
        start_training_run,
    )
except ModuleNotFoundError:
    from tracking import end_training_run, log_classification_result, start_training_run

warnings.filterwarnings("ignore")


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = ROOT / "data" / "processed" / "tickets_clean.csv"

REPORT_DIR = ROOT / "reports" / "queue_v5"

RANDOM_STATE = 42
TEST_SIZE = 0.20

TEXT_COLUMN = "ticket_text"
QUEUE_COLUMN = "queue"
TYPE_COLUMN = "type"


# ============================================================
# UTILITIES
# ============================================================


def print_header(title: str) -> None:
    print()
    print("=" * 80)
    print(title)
    print("=" * 80)


def normalize_text(series: pd.Series) -> pd.Series:
    return (
        series.fillna("")
        .astype(str)
        .str.lower()
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )


def make_stratification_label(df: pd.DataFrame) -> pd.Series:
    """
    Try to preserve the joint Type + Queue distribution.

    If some combinations are too rare for stratified splitting,
    fall back to Queue only.
    """

    combined = (
        df[TYPE_COLUMN].astype(str)
        + "|"
        + df[QUEUE_COLUMN].astype(str)
    )

    counts = combined.value_counts()

    if counts.min() >= 2:
        print("Stratification: TYPE + QUEUE")
        return combined

    print("Some TYPE|QUEUE combinations are too rare.")
    print("Fallback stratification: QUEUE")

    return df[QUEUE_COLUMN].astype(str)


def save_confusion_matrix(
    y_true: pd.Series,
    y_pred: np.ndarray,
    model_name: str,
) -> None:

    labels = sorted(y_true.unique())

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=labels,
    )

    cm_df = pd.DataFrame(
        cm,
        index=[f"TRUE: {x}" for x in labels],
        columns=[f"PRED: {x}" for x in labels],
    )

    path = REPORT_DIR / f"{model_name}_confusion_matrix.csv"

    cm_df.to_csv(
        path,
        encoding="utf-8",
    )

    print(f"Confusion matrix saved: {path}")


def evaluate_model(
    model_name: str,
    y_true: pd.Series,
    y_pred: np.ndarray,
) -> dict:

    with mlflow.start_run(run_name=model_name, nested=True):
        accuracy = accuracy_score(
            y_true,
            y_pred,
        )

        precision_macro = precision_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0,
        )

        recall_macro = recall_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0,
        )

        f1_macro = f1_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0,
        )

        f1_weighted = f1_score(
            y_true,
            y_pred,
            average="weighted",
            zero_division=0,
        )

        log_classification_result(y_true, y_pred)

    print()
    print(f"MODEL: {model_name}")
    print("-" * 60)

    print(f"Accuracy        : {accuracy:.4f}")
    print(f"Precision Macro : {precision_macro:.4f}")
    print(f"Recall Macro    : {recall_macro:.4f}")
    print(f"Macro F1        : {f1_macro:.4f}")
    print(f"Weighted F1     : {f1_weighted:.4f}")

    report = classification_report(
        y_true,
        y_pred,
        zero_division=0,
        output_dict=True,
    )

    report_df = pd.DataFrame(
        report
    ).transpose()

    report_path = (
        REPORT_DIR
        / f"{model_name}_classification_report.csv"
    )

    report_df.to_csv(
        report_path,
        encoding="utf-8",
    )

    print(
        f"Classification report saved: {report_path}"
    )

    save_confusion_matrix(
        y_true,
        y_pred,
        model_name,
    )

    return {
        "model": model_name,
        "accuracy": accuracy,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "macro_f1": f1_macro,
        "weighted_f1": f1_weighted,
    }


# ============================================================
# MAIN
# ============================================================


def main() -> None:

    print_header(
        "HARDTEC QUEUE CLASSIFIER V5 EXPERIMENT"
    )

    start_training_run("queue_classifier_v5_benchmark", DATA_PATH)

    mlflow.log_params(
        {
            "test_size": TEST_SIZE,
            "random_state": RANDOM_STATE,
            "variants": "V5-A,V5-B,V5-C,V5-D",
            "target": QUEUE_COLUMN,
        }
    )

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ========================================================
    # LOAD DATA
    # ========================================================

    print("Loading dataset:")
    print(DATA_PATH)

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
        )

    df = pd.read_csv(
        DATA_PATH
    )

    required_columns = [
        TEXT_COLUMN,
        QUEUE_COLUMN,
        TYPE_COLUMN,
    ]

    missing = [
        col
        for col in required_columns
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing columns: {missing}"
        )

    df = df[
        required_columns
    ].copy()

    df[TEXT_COLUMN] = normalize_text(
        df[TEXT_COLUMN]
    )

    df[QUEUE_COLUMN] = (
        df[QUEUE_COLUMN]
        .astype(str)
        .str.strip()
    )

    df[TYPE_COLUMN] = (
        df[TYPE_COLUMN]
        .astype(str)
        .str.strip()
    )

    df = df[
        df[TEXT_COLUMN].str.len() > 0
    ].reset_index(drop=True)

    print()
    print(
        f"Dataset size: {len(df):,}"
    )

    print(
        f"Queues      : {df[QUEUE_COLUMN].nunique()}"
    )

    print(
        f"Types       : {df[TYPE_COLUMN].nunique()}"
    )

    # ========================================================
    # COMMON TRAIN / TEST SPLIT
    # ========================================================

    print_header(
        "COMMON TRAIN / TEST SPLIT"
    )

    stratify_label = (
        make_stratification_label(df)
    )

    train_idx, test_idx = train_test_split(
        np.arange(len(df)),
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=stratify_label,
    )

    train_df = df.iloc[
        train_idx
    ].copy()

    test_df = df.iloc[
        test_idx
    ].copy()

    print(
        f"Train size: {len(train_df):,}"
    )

    print(
        f"Test size : {len(test_df):,}"
    )

    X_train_text = train_df[
        TEXT_COLUMN
    ]

    X_test_text = test_df[
        TEXT_COLUMN
    ]

    y_train = train_df[
        QUEUE_COLUMN
    ]

    y_test = test_df[
        QUEUE_COLUMN
    ]

    # ========================================================
    # TYPE FEATURE
    # ========================================================

    print_header(
        "PREPARING TYPE FEATURE"
    )

    type_encoder = OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=True,
    )

    train_type = type_encoder.fit_transform(
        train_df[[TYPE_COLUMN]]
    )

    test_type = type_encoder.transform(
        test_df[[TYPE_COLUMN]]
    )

    # ========================================================
    # V5-A
    # WORD TF-IDF
    # ========================================================

    print_header(
        "V5-A - WORD TF-IDF"
    )

    word_a = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        min_df=2,
        max_features=100000,
        sublinear_tf=True,
    )

    X_train_a = word_a.fit_transform(
        X_train_text
    )

    X_test_a = word_a.transform(
        X_test_text
    )

    classifier_a = LogisticRegression(
        max_iter=2000,
        class_weight=None,
        C=2.0,
        solver="lbfgs",
    )

    classifier_a.fit(
        X_train_a,
        y_train,
    )

    pred_a = classifier_a.predict(
        X_test_a
    )

    result_a = evaluate_model(
        "V5_A_word_tfidf",
        y_test,
        pred_a,
    )

    # ========================================================
    # V5-B
    # WORD TF-IDF + BALANCED
    # ========================================================

    print_header(
        "V5-B - WORD TF-IDF + BALANCED"
    )

    word_b = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        min_df=2,
        max_features=100000,
        sublinear_tf=True,
    )

    X_train_b = word_b.fit_transform(
        X_train_text
    )

    X_test_b = word_b.transform(
        X_test_text
    )

    classifier_b = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        C=2.0,
        solver="lbfgs",
    )

    classifier_b.fit(
        X_train_b,
        y_train,
    )

    pred_b = classifier_b.predict(
        X_test_b
    )

    result_b = evaluate_model(
        "V5_B_word_tfidf_balanced",
        y_test,
        pred_b,
    )

    # ========================================================
    # V5-C
    # WORD + CHARACTER TF-IDF
    # ========================================================

    print_header(
        "V5-C - WORD + CHARACTER TF-IDF + BALANCED"
    )

    word_c = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        min_df=2,
        max_features=80000,
        sublinear_tf=True,
    )

    char_c = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=2,
        max_features=80000,
        sublinear_tf=True,
    )

    X_train_word_c = word_c.fit_transform(
        X_train_text
    )

    X_test_word_c = word_c.transform(
        X_test_text
    )

    X_train_char_c = char_c.fit_transform(
        X_train_text
    )

    X_test_char_c = char_c.transform(
        X_test_text
    )

    X_train_c = hstack(
        [
            X_train_word_c,
            X_train_char_c,
        ]
    ).tocsr()

    X_test_c = hstack(
        [
            X_test_word_c,
            X_test_char_c,
        ]
    ).tocsr()

    classifier_c = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        C=2.0,
        solver="lbfgs",
    )

    classifier_c.fit(
        X_train_c,
        y_train,
    )

    pred_c = classifier_c.predict(
        X_test_c
    )

    result_c = evaluate_model(
        "V5_C_word_char_balanced",
        y_test,
        pred_c,
    )

    # ========================================================
    # V5-D
    # WORD TF-IDF + TYPE
    # ========================================================

    print_header(
        "V5-D - WORD TF-IDF + TYPE + BALANCED"
    )

    word_d = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        min_df=2,
        max_features=100000,
        sublinear_tf=True,
    )

    X_train_word_d = word_d.fit_transform(
        X_train_text
    )

    X_test_word_d = word_d.transform(
        X_test_text
    )

    X_train_d = hstack(
        [
            X_train_word_d,
            train_type,
        ]
    ).tocsr()

    X_test_d = hstack(
        [
            X_test_word_d,
            test_type,
        ]
    ).tocsr()

    classifier_d = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        C=2.0,
        solver="lbfgs",
    )

    classifier_d.fit(
        X_train_d,
        y_train,
    )

    pred_d = classifier_d.predict(
        X_test_d
    )

    result_d = evaluate_model(
        "V5_D_word_tfidf_type_balanced",
        y_test,
        pred_d,
    )

    # ========================================================
    # FINAL COMPARISON
    # ========================================================

    print_header(
        "FINAL COMPARISON"
    )

    results = pd.DataFrame(
        [
            result_a,
            result_b,
            result_c,
            result_d,
        ]
    )

    results = (
        results
        .sort_values(
            "macro_f1",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    results_path = (
        REPORT_DIR
        / "queue_v5_comparison.csv"
    )

    results.to_csv(
        results_path,
        index=False,
        encoding="utf-8",
    )

    print(
        results.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    # ========================================================
    # BEST MODEL
    # ========================================================

    best = results.iloc[0]

    print()
    print("=" * 80)
    print("BEST V5 MODEL")
    print("=" * 80)

    print(
        f"Model      : {best['model']}"
    )

    print(
        f"Accuracy   : {best['accuracy']:.4f}"
    )

    print(
        f"Macro F1   : {best['macro_f1']:.4f}"
    )

    print(
        f"Weighted F1: {best['weighted_f1']:.4f}"
    )

    # ========================================================
    # COMPARE WITH V4 BASELINE
    # ========================================================

    v4_accuracy = 0.5025
    v4_macro_f1 = 0.4848

    improvement_accuracy = (
        best["accuracy"]
        - v4_accuracy
    )

    improvement_f1 = (
        best["macro_f1"]
        - v4_macro_f1
    )

    print()
    print("=" * 80)
    print("V4 → V5 IMPROVEMENT")
    print("=" * 80)

    print(
        f"V4 Accuracy  : {v4_accuracy:.4f}"
    )

    print(
        f"Best Accuracy: {best['accuracy']:.4f}"
    )

    print(
        f"Accuracy Δ   : {improvement_accuracy:+.4f}"
    )

    print()

    print(
        f"V4 Macro F1  : {v4_macro_f1:.4f}"
    )

    print(
        f"Best Macro F1: {best['macro_f1']:.4f}"
    )

    print(
        f"Macro F1 Δ   : {improvement_f1:+.4f}"
    )

    # ========================================================
    # SAVE METADATA
    # ========================================================

    metadata = {
        "dataset": str(DATA_PATH),
        "dataset_size": len(df),
        "train_size": len(train_df),
        "test_size": len(test_df),
        "random_state": RANDOM_STATE,
        "test_size_ratio": TEST_SIZE,
        "v4_baseline": {
            "accuracy": v4_accuracy,
            "macro_f1": v4_macro_f1,
        },
        "best_model": best["model"],
        "best_accuracy": float(
            best["accuracy"]
        ),
        "best_macro_f1": float(
            best["macro_f1"]
        ),
    }

    metadata_path = (
        REPORT_DIR
        / "queue_v5_metadata.json"
    )

    with open(
        metadata_path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            metadata,
            f,
            indent=4,
        )

    # ========================================================
    # FINISHED
    # ========================================================

    print()
    print("=" * 80)
    print("EXPERIMENT FINISHED")
    print("=" * 80)

    print()
    print("Generated files:")

    print(
        f"  {results_path}"
    )

    print(
        f"  {metadata_path}"
    )

    print(
        f"  {REPORT_DIR}\\*_classification_report.csv"
    )

    print(
        f"  {REPORT_DIR}\\*_confusion_matrix.csv"
    )

    print()
    print(
        "IMPORTANT: ticket_queue_model_v4.pkl "
        "was NOT modified."
    )

    end_training_run()


if __name__ == "__main__":
    main()

