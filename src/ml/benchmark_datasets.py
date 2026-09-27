# ============================================================
# HARDTEC - DATASET BENCHMARK
# ============================================================
#
# Compare:
#   Dataset A = HARDTEC current cleaned dataset
#   Dataset B = Kaggle/Hugging Face 20K benchmark dataset
#
# Tasks:
#   1. Ticket Type classification
#   2. Queue classification
#   3. Priority classification
#
# IMPORTANT:
# - This script DOES NOT modify production .pkl models.
# - It trains temporary models only for benchmarking.
# - Dataset A and Dataset B use the same 80/20 evaluation protocol.
#
# ============================================================

from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC

warnings.filterwarnings("ignore")


# ============================================================
# CONFIGURATION
# ============================================================

RANDOM_STATE = 42
TEST_SIZE = 0.20

MAX_FEATURES = 40_000
MIN_DF = 2

DATASET_A = Path("data/processed/tickets_clean.csv")
DATASET_B = Path("data/raw/benchmark/tickets_kaggle_20k.csv")

REPORT_DIR = Path("reports/dataset_benchmark")
REPORT_DIR.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------
# Model parameters
# ------------------------------------------------------------

TYPE_C = 1.0
QUEUE_C = 1.5
PRIORITY_C = 1.0


# ============================================================
# UTILITIES
# ============================================================


def normalize_text(series):
    """
    Normalize ticket text.
    """
    return (
        series.fillna("")
        .astype(str)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )


def build_text_dataset_a(df):
    """
    Dataset A already contains ticket_text.
    """
    if "ticket_text" not in df.columns:
        raise ValueError(
            "Dataset A must contain a 'ticket_text' column."
        )

    df = df.copy()
    df["ticket_text"] = normalize_text(df["ticket_text"])

    return df


def build_text_dataset_b(df):
    """
    Dataset B contains subject + body.
    Create ticket_text.
    """
    required = ["subject", "body", "type", "queue", "priority"]

    missing = [col for col in required if col not in df.columns]

    if missing:
        raise ValueError(
            f"Dataset B is missing required columns: {missing}"
        )

    df = df.copy()

    df["subject"] = normalize_text(df["subject"])
    df["body"] = normalize_text(df["body"])

    df["ticket_text"] = (
        df["subject"] + " " + df["body"]
    ).str.strip()

    return df


def check_labels(df, dataset_name):
    """
    Verify that the benchmark datasets use the expected
    HARDTEC label taxonomy.
    """

    expected_types = {
        "Incident",
        "Request",
        "Problem",
        "Change",
    }

    expected_queues = {
        "Technical Support",
        "Product Support",
        "Customer Service",
        "IT Support",
        "Billing and Payments",
        "Returns and Exchanges",
        "Service Outages and Maintenance",
        "Sales and Pre-Sales",
        "Human Resources",
        "General Inquiry",
    }

    expected_priorities = {
        "low",
        "medium",
        "high",
    }

    actual_types = set(df["type"].dropna().unique())
    actual_queues = set(df["queue"].dropna().unique())
    actual_priorities = set(df["priority"].dropna().unique())

    print("\n" + "=" * 70)
    print(f"LABEL CHECK - {dataset_name}")
    print("=" * 70)

    print("\nTypes:")
    print(actual_types)

    print("\nQueues:")
    print(actual_queues)

    print("\nPriorities:")
    print(actual_priorities)

    if actual_types != expected_types:
        print("\nWARNING: Type labels differ from expected taxonomy.")

    if actual_queues != expected_queues:
        print("\nWARNING: Queue labels differ from expected taxonomy.")

    if actual_priorities != expected_priorities:
        print("\nWARNING: Priority labels differ from expected taxonomy.")


def print_dataset_summary(df, dataset_name):
    """
    Print basic dataset information.
    """

    print("\n" + "=" * 70)
    print(f"DATASET SUMMARY - {dataset_name}")
    print("=" * 70)

    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns)}")

    print("\nColumns:")
    print(df.columns.tolist())

    print("\nTypes:")
    print(df["type"].value_counts())

    print("\nQueues:")
    print(df["queue"].value_counts())

    print("\nPriorities:")
    print(df["priority"].value_counts())


# ============================================================
# TEXT VECTORISATION
# ============================================================


def create_tfidf():
    """
    Create a consistent TF-IDF vectorizer for both datasets.
    """

    return TfidfVectorizer(
        max_features=MAX_FEATURES,
        ngram_range=(1, 2),
        min_df=MIN_DF,
        sublinear_tf=True,
        strip_accents="unicode",
    )


# ============================================================
# TYPE MODEL
# ============================================================


def train_type_model(X_train, y_train):
    """
    Temporary Type classifier.

    IMPORTANT:
    LogisticRegression uses lbfgs because liblinear does not
    support multiclass classification directly.
    """

    vectorizer = create_tfidf()

    model = Pipeline(
        [
            ("tfidf", vectorizer),
            (
                "classifier",
                LogisticRegression(
                    C=TYPE_C,
                    max_iter=2000,
                    class_weight="balanced",
                    solver="lbfgs",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )

    model.fit(X_train, y_train)

    return model


# ============================================================
# QUEUE MODEL
# ============================================================


def train_queue_model(X_train, y_train):
    """
    Temporary Queue classifier.

    Uses LinearSVC with balanced class weights,
    close to the current HARDTEC queue architecture.
    """

    vectorizer = create_tfidf()

    model = Pipeline(
        [
            ("tfidf", vectorizer),
            (
                "classifier",
                LinearSVC(
                    C=QUEUE_C,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )

    model.fit(X_train, y_train)

    return model


# ============================================================
# PRIORITY MODEL
# ============================================================


def train_priority_model(X_train, y_train):
    """
    Temporary Priority classifier.
    """

    vectorizer = create_tfidf()

    model = Pipeline(
        [
            ("tfidf", vectorizer),
            (
                "classifier",
                LogisticRegression(
                    C=PRIORITY_C,
                    max_iter=2000,
                    class_weight="balanced",
                    solver="lbfgs",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )

    model.fit(X_train, y_train)

    return model


# ============================================================
# METRICS
# ============================================================


def calculate_metrics(y_true, y_pred):
    """
    Calculate standard classification metrics.
    """

    precision_macro, recall_macro, f1_macro, _ = (
        precision_recall_fscore_support(
            y_true,
            y_pred,
            average="macro",
            zero_division=0,
        )
    )

    precision_weighted, recall_weighted, f1_weighted, _ = (
        precision_recall_fscore_support(
            y_true,
            y_pred,
            average="weighted",
            zero_division=0,
        )
    )

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "f1_macro": f1_macro,
        "precision_weighted": precision_weighted,
        "recall_weighted": recall_weighted,
        "f1_weighted": f1_weighted,
    }


def save_classification_report(
    y_true,
    y_pred,
    dataset_name,
    task,
):
    """
    Save sklearn classification report.
    """

    report = classification_report(
        y_true,
        y_pred,
        zero_division=0,
        output_dict=True,
    )

    report_df = pd.DataFrame(report).transpose()

    output_file = (
        REPORT_DIR
        / f"{dataset_name.lower()}_{task.lower()}_classification_report.csv"
    )

    report_df.to_csv(output_file)

    return output_file


def save_confusion_matrix(
    y_true,
    y_pred,
    dataset_name,
    task,
):
    """
    Save confusion matrix as CSV.
    """

    labels = sorted(set(y_true) | set(y_pred))

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=labels,
    )

    cm_df = pd.DataFrame(
        cm,
        index=labels,
        columns=labels,
    )

    output_file = (
        REPORT_DIR
        / f"{dataset_name.lower()}_{task.lower()}_confusion_matrix.csv"
    )

    cm_df.to_csv(output_file)

    return output_file


# ============================================================
# STRATIFICATION
# ============================================================


def create_stratification_labels(df):
    """
    Create combined labels:

        type | queue | priority

    This ensures that the 80/20 split preserves the
    joint distribution as much as possible.
    """

    combined = (
        df["type"].astype(str)
        + "|"
        + df["queue"].astype(str)
        + "|"
        + df["priority"].astype(str)
    )

    counts = combined.value_counts()

    # Stratification requires at least 2 samples per class.
    valid_classes = counts[counts >= 2].index

    if len(valid_classes) == len(counts):
        return combined

    print(
        "\nWARNING: Some type|queue|priority combinations "
        "have fewer than 2 samples."
    )

    # Fallback to queue stratification if necessary.
    return df["queue"].astype(str)


# ============================================================
# SINGLE DATASET BENCHMARK
# ============================================================


def benchmark_dataset(df, dataset_name):
    """
    Run complete benchmark on one dataset.

    Pipeline:

        Ticket Text
             |
             v
          Type
             |
             v
          Queue
             |
             v
         Priority
    """

    print("\n\n")
    print("#" * 70)
    print(f"# BENCHMARK: {dataset_name}")
    print("#" * 70)

    df = df.copy()

    # --------------------------------------------------------
    # Remove invalid rows
    # --------------------------------------------------------

    required_columns = [
        "ticket_text",
        "type",
        "queue",
        "priority",
    ]

    df = df.dropna(
        subset=required_columns
    ).reset_index(drop=True)

    df["ticket_text"] = normalize_text(
        df["ticket_text"]
    )

    # Remove empty tickets
    df = df[
        df["ticket_text"].str.len() > 0
    ].reset_index(drop=True)

    print(f"\nUsable rows: {len(df):,}")

    # --------------------------------------------------------
    # Train/test split
    # --------------------------------------------------------

    stratify_labels = create_stratification_labels(df)

    (
        train_df,
        test_df,
    ) = train_test_split(
        df,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=stratify_labels,
    )

    print(
        f"Train: {len(train_df):,} "
        f"| Test: {len(test_df):,}"
    )

    # ========================================================
    # 1. TYPE
    # ========================================================

    print("\n" + "-" * 70)
    print("1. TYPE CLASSIFICATION")
    print("-" * 70)

    X_train = train_df["ticket_text"]
    X_test = test_df["ticket_text"]

    y_train = train_df["type"]
    y_test = test_df["type"]

    type_model = train_type_model(
        X_train,
        y_train,
    )

    type_pred = type_model.predict(X_test)

    type_metrics = calculate_metrics(
        y_test,
        type_pred,
    )

    print(
        f"Accuracy : {type_metrics['accuracy']:.4f}"
    )
    print(
        f"Macro F1 : {type_metrics['f1_macro']:.4f}"
    )
    print(
        f"Weighted F1 : {type_metrics['f1_weighted']:.4f}"
    )

    save_classification_report(
        y_test,
        type_pred,
        dataset_name,
        "type",
    )

    save_confusion_matrix(
        y_test,
        type_pred,
        dataset_name,
        "type",
    )

    # ========================================================
    # 2. QUEUE
    # ========================================================

    print("\n" + "-" * 70)
    print("2. QUEUE CLASSIFICATION")
    print("-" * 70)

    # --------------------------------------------------------
    # IMPORTANT:
    #
    # Queue receives the PREDICTED TYPE,
    # not the true type.
    #
    # This reproduces the end-to-end pipeline:
    #
    # Ticket -> Type prediction -> Queue prediction
    # --------------------------------------------------------

    type_encoder = LabelEncoder()

    combined_queue_train = (
        train_df["ticket_text"]
        + " [TYPE] "
        + train_df["type"].astype(str)
    )

    combined_queue_test = (
        test_df["ticket_text"]
        + " [TYPE] "
        + pd.Series(
            type_pred,
            index=test_df.index,
        ).astype(str)
    )

    queue_model = train_queue_model(
        combined_queue_train,
        train_df["queue"],
    )

    queue_pred = queue_model.predict(
        combined_queue_test
    )

    queue_metrics = calculate_metrics(
        test_df["queue"],
        queue_pred,
    )

    print(
        f"Accuracy : {queue_metrics['accuracy']:.4f}"
    )
    print(
        f"Macro F1 : {queue_metrics['f1_macro']:.4f}"
    )
    print(
        f"Weighted F1 : {queue_metrics['f1_weighted']:.4f}"
    )

    save_classification_report(
        test_df["queue"],
        queue_pred,
        dataset_name,
        "queue",
    )

    save_confusion_matrix(
        test_df["queue"],
        queue_pred,
        dataset_name,
        "queue",
    )

    # ========================================================
    # 3. PRIORITY
    # ========================================================

    print("\n" + "-" * 70)
    print("3. PRIORITY CLASSIFICATION")
    print("-" * 70)

    # --------------------------------------------------------
    # Priority receives predicted Type + predicted Queue.
    # --------------------------------------------------------

    combined_priority_train = (
        train_df["ticket_text"]
        + " [TYPE] "
        + train_df["type"].astype(str)
        + " [QUEUE] "
        + train_df["queue"].astype(str)
    )

    combined_priority_test = (
        test_df["ticket_text"]
        + " [TYPE] "
        + pd.Series(
            type_pred,
            index=test_df.index,
        ).astype(str)
        + " [QUEUE] "
        + pd.Series(
            queue_pred,
            index=test_df.index,
        ).astype(str)
    )

    priority_model = train_priority_model(
        combined_priority_train,
        train_df["priority"],
    )

    priority_pred = priority_model.predict(
        combined_priority_test
    )

    priority_metrics = calculate_metrics(
        test_df["priority"],
        priority_pred,
    )

    print(
        f"Accuracy : {priority_metrics['accuracy']:.4f}"
    )
    print(
        f"Macro F1 : {priority_metrics['f1_macro']:.4f}"
    )
    print(
        f"Weighted F1 : {priority_metrics['f1_weighted']:.4f}"
    )

    save_classification_report(
        test_df["priority"],
        priority_pred,
        dataset_name,
        "priority",
    )

    save_confusion_matrix(
        test_df["priority"],
        priority_pred,
        dataset_name,
        "priority",
    )

    # ========================================================
    # SAVE PREDICTIONS
    # ========================================================

    predictions = test_df[
        [
            "ticket_text",
            "type",
            "queue",
            "priority",
        ]
    ].copy()

    predictions.rename(
        columns={
            "type": "true_type",
            "queue": "true_queue",
            "priority": "true_priority",
        },
        inplace=True,
    )

    predictions["pred_type"] = type_pred
    predictions["pred_queue"] = queue_pred
    predictions["pred_priority"] = priority_pred

    predictions["type_correct"] = (
        predictions["true_type"]
        == predictions["pred_type"]
    )

    predictions["queue_correct"] = (
        predictions["true_queue"]
        == predictions["pred_queue"]
    )

    predictions["priority_correct"] = (
        predictions["true_priority"]
        == predictions["pred_priority"]
    )

    prediction_file = (
        REPORT_DIR
        / f"{dataset_name.lower()}_predictions.csv"
    )

    predictions.to_csv(
        prediction_file,
        index=False,
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    summary = {
        "dataset": dataset_name,
        "rows_total": int(len(df)),
        "rows_train": int(len(train_df)),
        "rows_test": int(len(test_df)),
        "type_accuracy": type_metrics["accuracy"],
        "type_precision_macro": type_metrics[
            "precision_macro"
        ],
        "type_recall_macro": type_metrics[
            "recall_macro"
        ],
        "type_f1_macro": type_metrics[
            "f1_macro"
        ],
        "type_f1_weighted": type_metrics[
            "f1_weighted"
        ],
        "queue_accuracy": queue_metrics["accuracy"],
        "queue_precision_macro": queue_metrics[
            "precision_macro"
        ],
        "queue_recall_macro": queue_metrics[
            "recall_macro"
        ],
        "queue_f1_macro": queue_metrics[
            "f1_macro"
        ],
        "queue_f1_weighted": queue_metrics[
            "f1_weighted"
        ],
        "priority_accuracy": priority_metrics[
            "accuracy"
        ],
        "priority_precision_macro": priority_metrics[
            "precision_macro"
        ],
        "priority_recall_macro": priority_metrics[
            "recall_macro"
        ],
        "priority_f1_macro": priority_metrics[
            "f1_macro"
        ],
        "priority_f1_weighted": priority_metrics[
            "f1_weighted"
        ],
    }

    summary_file = (
        REPORT_DIR
        / f"{dataset_name.lower()}_summary.json"
    )

    with open(
        summary_file,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            summary,
            f,
            indent=4,
        )

    return summary


# ============================================================
# DATASET OVERLAP CHECK
# ============================================================


def check_dataset_overlap(
    df_a,
    df_b,
):
    """
    Check whether Dataset A and Dataset B contain
    the same ticket texts.

    This is important because the 20K Kaggle/HF dataset
    may be the same underlying dataset as Dataset A.
    """

    print("\n" + "=" * 70)
    print("DATASET OVERLAP CHECK")
    print("=" * 70)

    text_a = (
        df_a["ticket_text"]
        .fillna("")
        .astype(str)
        .str.lower()
        .str.replace(
            r"\s+",
            " ",
            regex=True,
        )
        .str.strip()
    )

    text_b = (
        df_b["ticket_text"]
        .fillna("")
        .astype(str)
        .str.lower()
        .str.replace(
            r"\s+",
            " ",
            regex=True,
        )
        .str.strip()
    )

    set_a = set(text_a)
    set_b = set(text_b)

    intersection = set_a & set_b

    print(
        f"Dataset A unique texts : {len(set_a):,}"
    )

    print(
        f"Dataset B unique texts : {len(set_b):,}"
    )

    print(
        f"Exact text matches     : {len(intersection):,}"
    )

    print(
        f"All A texts in B       : {set_a.issubset(set_b)}"
    )

    print(
        f"All B texts in A       : {set_b.issubset(set_a)}"
    )

    overlap_percentage = (
        len(intersection)
        / max(1, min(len(set_a), len(set_b)))
        * 100
    )

    print(
        f"Overlap relative to smaller dataset: "
        f"{overlap_percentage:.2f}%"
    )

    overlap_result = {
        "dataset_a_unique_texts": len(set_a),
        "dataset_b_unique_texts": len(set_b),
        "exact_text_matches": len(intersection),
        "all_a_in_b": set_a.issubset(set_b),
        "all_b_in_a": set_b.issubset(set_a),
        "overlap_percentage_smaller_dataset": overlap_percentage,
    }

    output_file = (
        REPORT_DIR
        / "dataset_overlap.json"
    )

    with open(
        output_file,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            overlap_result,
            f,
            indent=4,
        )

    return overlap_result


# ============================================================
# COMPARISON TABLE
# ============================================================


def create_comparison_table(
    summaries,
):
    """
    Create Dataset A vs Dataset B comparison table.
    """

    rows = []

    for summary in summaries:
        rows.append(
            {
                "Dataset": summary["dataset"],
                "Rows": summary["rows_total"],
                "Type Accuracy": summary[
                    "type_accuracy"
                ],
                "Type Macro F1": summary[
                    "type_f1_macro"
                ],
                "Queue Accuracy": summary[
                    "queue_accuracy"
                ],
                "Queue Macro F1": summary[
                    "queue_f1_macro"
                ],
                "Priority Accuracy": summary[
                    "priority_accuracy"
                ],
                "Priority Macro F1": summary[
                    "priority_f1_macro"
                ],
            }
        )

    comparison_df = pd.DataFrame(rows)

    output_file = (
        REPORT_DIR
        / "dataset_comparison.csv"
    )

    comparison_df.to_csv(
        output_file,
        index=False,
    )

    return comparison_df


# ============================================================
# MAIN
# ============================================================


def main():

    print("\n")
    print("=" * 70)
    print("HARDTEC DATASET BENCHMARK")
    print("=" * 70)

    print(
        "\nEvaluation protocol:"
    )

    print(
        "  • Test size       : 20%"
    )

    print(
        "  • Random state    : 42"
    )

    print(
        "  • TF-IDF features : 40,000"
    )

    print(
        "  • N-grams         : (1, 2)"
    )

    print(
        "  • Type            : LogisticRegression"
    )

    print(
        "  • Queue           : LinearSVC"
    )

    print(
        "  • Priority        : LogisticRegression"
    )

    # ========================================================
    # LOAD DATASET A
    # ========================================================

    if not DATASET_A.exists():
        raise FileNotFoundError(
            f"Dataset A not found:\n{DATASET_A}"
        )

    print(
        f"\nLoading Dataset A:\n{DATASET_A}"
    )

    df_a = pd.read_csv(
        DATASET_A
    )

    df_a = build_text_dataset_a(
        df_a
    )

    print_dataset_summary(
        df_a,
        "Dataset A - HARDTEC",
    )

    check_labels(
        df_a,
        "Dataset A - HARDTEC",
    )

    # ========================================================
    # LOAD DATASET B
    # ========================================================

    if not DATASET_B.exists():
        raise FileNotFoundError(
            f"Dataset B not found:\n{DATASET_B}"
        )

    print(
        f"\nLoading Dataset B:\n{DATASET_B}"
    )

    df_b = pd.read_csv(
        DATASET_B
    )

    df_b = build_text_dataset_b(
        df_b
    )

    print_dataset_summary(
        df_b,
        "Dataset B - Kaggle/HF 20K",
    )

    check_labels(
        df_b,
        "Dataset B - Kaggle/HF 20K",
    )

    # ========================================================
    # OVERLAP CHECK
    # ========================================================

    overlap = check_dataset_overlap(
        df_a,
        df_b,
    )

    # ========================================================
    # BENCHMARK DATASET A
    # ========================================================

    summary_a = benchmark_dataset(
        df_a,
        "dataset_a_hardtec",
    )

    # ========================================================
    # BENCHMARK DATASET B
    # ========================================================

    summary_b = benchmark_dataset(
        df_b,
        "dataset_b_kaggle_20k",
    )

    # ========================================================
    # COMPARISON
    # ========================================================

    comparison_df = create_comparison_table(
        [
            summary_a,
            summary_b,
        ]
    )

    print("\n")
    print("=" * 70)
    print("FINAL DATASET COMPARISON")
    print("=" * 70)

    print(
        comparison_df.to_string(
            index=False
        )
    )

    # ========================================================
    # SAVE FINAL JSON
    # ========================================================

    final_result = {
        "evaluation_protocol": {
            "test_size": TEST_SIZE,
            "random_state": RANDOM_STATE,
            "max_features": MAX_FEATURES,
            "min_df": MIN_DF,
            "ngram_range": [1, 2],
        },
        "dataset_a": summary_a,
        "dataset_b": summary_b,
        "overlap": overlap,
    }

    final_json = (
        REPORT_DIR
        / "final_benchmark.json"
    )

    with open(
        final_json,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            final_result,
            f,
            indent=4,
        )

    # ========================================================
    # FINAL MESSAGE
    # ========================================================

    print("\n")
    print("=" * 70)
    print("BENCHMARK FINISHED")
    print("=" * 70)

    print(
        f"\nReports saved in:\n{REPORT_DIR}"
    )

    print("\nGenerated files include:")

    print(
        "  • dataset_comparison.csv"
    )

    print(
        "  • final_benchmark.json"
    )

    print(
        "  • dataset_overlap.json"
    )

    print(
        "  • Dataset A classification reports"
    )

    print(
        "  • Dataset B classification reports"
    )

    print(
        "  • Confusion matrices"
    )

    print(
        "  • Test-set predictions"
    )

    print(
        "\nIMPORTANT:"
    )

    print(
        "If the overlap check shows ~100% overlap, "
        "Dataset B should NOT be presented as an "
        "independent external benchmark."
    )


# ============================================================
# ENTRY POINT
# ============================================================


if __name__ == "__main__":
    main()