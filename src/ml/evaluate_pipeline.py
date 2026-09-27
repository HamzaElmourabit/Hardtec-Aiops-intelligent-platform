"""
HARDTEC - End-to-End ML Pipeline Evaluation
============================================

Pipeline evaluated:

    Ticket Text
        ↓
    Type Model
        ↓
    Queue Model
        ↓
    Priority Model

This script evaluates the complete ML pipeline under realistic
inference conditions.

IMPORTANT:
- Existing .pkl models are NOT modified.
- New models are trained only in memory.
- A single common train/test split is used.
- Queue receives the PREDICTED type.
- Priority receives the PREDICTED type + PREDICTED queue.

Dataset:
    data/processed/tickets_clean.csv

Run from project root:

    python src/ml/evaluate_pipeline.py
"""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd

from scipy.sparse import hstack

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

warnings.filterwarnings("ignore")


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = PROJECT_ROOT / "data" / "processed" / "tickets_clean.csv"

RANDOM_STATE = 42
TEST_SIZE = 0.20


# ============================================================
# DISPLAY HELPERS
# ============================================================


def print_header(title: str) -> None:
    """Print a clean section header."""
    print()
    print("=" * 80)
    print(title)
    print("=" * 80)


def print_metrics(
    y_true: pd.Series,
    y_pred: np.ndarray,
    model_name: str,
) -> None:
    """Print standard classification metrics."""

    accuracy = accuracy_score(y_true, y_pred)

    precision = precision_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )

    recall = recall_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )

    macro_f1 = f1_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )

    weighted_f1 = f1_score(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0,
    )

    print()
    print(f"Model: {model_name}")
    print("-" * 50)

    print(f"Accuracy      : {accuracy:.4f}")
    print(f"Precision     : {precision:.4f}")
    print(f"Recall        : {recall:.4f}")
    print(f"Macro F1      : {macro_f1:.4f}")
    print(f"Weighted F1   : {weighted_f1:.4f}")

    print()
    print("Classification Report")
    print("-" * 50)

    print(
        classification_report(
            y_true,
            y_pred,
            zero_division=0,
        )
    )


def print_confusion_matrix(
    y_true: pd.Series,
    y_pred: np.ndarray,
    model_name: str,
) -> None:
    """Print a labeled confusion matrix."""

    labels = sorted(
        set(y_true.astype(str))
        | set(pd.Series(y_pred).astype(str))
    )

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=labels,
    )

    cm_df = pd.DataFrame(
        cm,
        index=[f"TRUE: {label}" for label in labels],
        columns=[f"PRED: {label}" for label in labels],
    )

    print()
    print(f"Confusion Matrix - {model_name}")
    print("-" * 50)
    print(cm_df.to_string())


# ============================================================
# DATA LOADING
# ============================================================


def load_dataset() -> pd.DataFrame:
    """Load and validate the cleaned dataset."""

    print_header("1. DATASET LOADING")

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found:\n{DATA_PATH}"
        )

    print(f"Dataset path: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)

    required_columns = [
        "ticket_text",
        "type",
        "queue",
        "priority",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "Missing required columns: "
            + ", ".join(missing_columns)
        )

    df = df[required_columns].copy()

    # Remove invalid rows
    df = df.dropna(
        subset=[
            "ticket_text",
            "type",
            "queue",
            "priority",
        ]
    )

    # Normalize text and labels
    for column in required_columns:
        df[column] = df[column].astype(str).str.strip()

    # Remove empty rows
    df = df[
        (df["ticket_text"] != "")
        & (df["type"] != "")
        & (df["queue"] != "")
        & (df["priority"] != "")
    ].copy()

    df = df.reset_index(drop=True)

    print()
    print(f"Number of tickets : {len(df):,}")
    print(f"Number of types   : {df['type'].nunique()}")
    print(f"Number of queues  : {df['queue'].nunique()}")
    print(f"Priorities        : {df['priority'].nunique()}")

    print()
    print("Type distribution:")
    print(df["type"].value_counts().to_string())

    print()
    print("Queue distribution:")
    print(df["queue"].value_counts().to_string())

    print()
    print("Priority distribution:")
    print(df["priority"].value_counts().to_string())

    return df


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================


def create_split(df: pd.DataFrame):
    """
    Create ONE common train/test split.

    We stratify using a combined label to preserve the
    relationship between type, queue and priority as much
    as possible.

    If some combinations occur only once, the script falls
    back to stratifying by type.
    """

    print_header("2. COMMON TRAIN / TEST SPLIT")

    df = df.copy()

    combined_label = (
        df["type"].astype(str)
        + "|||"
        + df["queue"].astype(str)
        + "|||"
        + df["priority"].astype(str)
    )

    combination_counts = combined_label.value_counts()

    can_stratify_combined = (
        combination_counts.min() >= 2
    )

    if can_stratify_combined:
        stratify_labels = combined_label

        print(
            "Stratification: "
            "type + queue + priority combination"
        )

    else:
        stratify_labels = df["type"]

        print(
            "Some type/queue/priority combinations are too rare."
        )
        print(
            "Fallback stratification: type"
        )

    train_df, test_df = train_test_split(
        df,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=stratify_labels,
    )

    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    print()
    print(f"Training set : {len(train_df):,}")
    print(f"Test set     : {len(test_df):,}")

    print()
    print(
        f"Test percentage: "
        f"{len(test_df) / len(df) * 100:.2f}%"
    )

    return train_df, test_df


# ============================================================
# TYPE MODEL
# ============================================================


def train_type_model(train_df: pd.DataFrame):
    """Train the Type classifier."""

    print_header("3. TRAINING TYPE MODEL")

    vectorizer = TfidfVectorizer(
        lowercase=True,
        max_features=30000,
        ngram_range=(1, 2),
        min_df=2,
        sublinear_tf=True,
    )

    X_train = vectorizer.fit_transform(
        train_df["ticket_text"]
    )

    y_train = train_df["type"]

    classifier = LinearSVC(
        C=1.0,
    )

    classifier.fit(
        X_train,
        y_train,
    )

    print("Type model trained successfully.")

    return vectorizer, classifier


def predict_type(
    vectorizer,
    classifier,
    test_df: pd.DataFrame,
) -> np.ndarray:
    """Predict ticket type."""

    X_test = vectorizer.transform(
        test_df["ticket_text"]
    )

    predictions = classifier.predict(X_test)

    return predictions


# ============================================================
# QUEUE MODEL
# ============================================================


def train_queue_model(train_df: pd.DataFrame):
    """
    Train Queue classifier.

    Metadata:
        type

    IMPORTANT:
    During evaluation, the model will receive the
    PREDICTED type rather than the true type.
    """

    print_header("4. TRAINING QUEUE MODEL")

    text_vectorizer = TfidfVectorizer(
        lowercase=True,
        max_features=40000,
        ngram_range=(1, 2),
        min_df=2,
        sublinear_tf=True,
    )

    X_text = text_vectorizer.fit_transform(
        train_df["ticket_text"]
    )

    metadata_encoder = OneHotEncoder(
        handle_unknown="ignore",
    )

    X_meta = metadata_encoder.fit_transform(
        train_df[["type"]]
    )

    X_train = hstack(
        [
            X_text,
            X_meta,
        ]
    ).tocsr()

    y_train = train_df["queue"]

    classifier = LinearSVC(
        C=1.5,
        class_weight="balanced",
    )

    classifier.fit(
        X_train,
        y_train,
    )

    print("Queue model trained successfully.")

    return (
        text_vectorizer,
        metadata_encoder,
        classifier,
    )


def predict_queue(
    vectorizer,
    encoder,
    classifier,
    test_df: pd.DataFrame,
    predicted_type: np.ndarray,
) -> np.ndarray:
    """
    Predict queue using:

        ticket text
        +
        PREDICTED type
    """

    X_text = vectorizer.transform(
        test_df["ticket_text"]
    )

    predicted_type_df = pd.DataFrame(
        {
            "type": predicted_type
        }
    )

    X_meta = encoder.transform(
        predicted_type_df
    )

    X_test = hstack(
        [
            X_text,
            X_meta,
        ]
    ).tocsr()

    predictions = classifier.predict(X_test)

    return predictions


# ============================================================
# PRIORITY MODEL
# ============================================================


def train_priority_model(train_df: pd.DataFrame):
    """
    Train Priority classifier.

    Metadata:
        type
        queue

    During realistic evaluation, these will be predicted
    values rather than ground truth values.
    """

    print_header("5. TRAINING PRIORITY MODEL")

    text_vectorizer = TfidfVectorizer(
        lowercase=True,
        max_features=30000,
        ngram_range=(1, 2),
        min_df=2,
        sublinear_tf=True,
    )

    X_text = text_vectorizer.fit_transform(
        train_df["ticket_text"]
    )

    metadata_encoder = OneHotEncoder(
        handle_unknown="ignore",
    )

    X_meta = metadata_encoder.fit_transform(
        train_df[
            [
                "type",
                "queue",
            ]
        ]
    )

    X_train = hstack(
        [
            X_text,
            X_meta,
        ]
    ).tocsr()

    y_train = train_df["priority"]

    classifier = LinearSVC(
        C=1.0,
        class_weight="balanced",
    )

    classifier.fit(
        X_train,
        y_train,
    )

    print("Priority model trained successfully.")

    return (
        text_vectorizer,
        metadata_encoder,
        classifier,
    )


def predict_priority(
    vectorizer,
    encoder,
    classifier,
    test_df: pd.DataFrame,
    predicted_type: np.ndarray,
    predicted_queue: np.ndarray,
) -> np.ndarray:
    """
    Predict priority using:

        ticket text
        +
        PREDICTED type
        +
        PREDICTED queue
    """

    X_text = vectorizer.transform(
        test_df["ticket_text"]
    )

    predicted_metadata = pd.DataFrame(
        {
            "type": predicted_type,
            "queue": predicted_queue,
        }
    )

    X_meta = encoder.transform(
        predicted_metadata
    )

    X_test = hstack(
        [
            X_text,
            X_meta,
        ]
    ).tocsr()

    predictions = classifier.predict(X_test)

    return predictions


# ============================================================
# PIPELINE EVALUATION
# ============================================================


def evaluate_pipeline(
    test_df: pd.DataFrame,
    predicted_type: np.ndarray,
    predicted_queue: np.ndarray,
    predicted_priority: np.ndarray,
) -> None:
    """Evaluate the complete pipeline."""

    print_header("6. END-TO-END PIPELINE RESULTS")

    # --------------------------------------------------------
    # TYPE
    # --------------------------------------------------------

    print_metrics(
        test_df["type"],
        predicted_type,
        "TYPE",
    )

    print_confusion_matrix(
        test_df["type"],
        predicted_type,
        "TYPE",
    )

    # --------------------------------------------------------
    # QUEUE
    # --------------------------------------------------------

    print_metrics(
        test_df["queue"],
        predicted_queue,
        "QUEUE",
    )

    print_confusion_matrix(
        test_df["queue"],
        predicted_queue,
        "QUEUE",
    )

    # --------------------------------------------------------
    # PRIORITY
    # --------------------------------------------------------

    print_metrics(
        test_df["priority"],
        predicted_priority,
        "PRIORITY",
    )

    print_confusion_matrix(
        test_df["priority"],
        predicted_priority,
        "PRIORITY",
    )


# ============================================================
# ERROR ANALYSIS
# ============================================================


def analyze_errors(
    test_df: pd.DataFrame,
    predicted_type: np.ndarray,
    predicted_queue: np.ndarray,
    predicted_priority: np.ndarray,
) -> None:
    """
    Display examples where the pipeline made mistakes.

    This is especially useful for diagnosing Queue errors.
    """

    print_header("7. ERROR ANALYSIS")

    results = test_df[
        [
            "ticket_text",
            "type",
            "queue",
            "priority",
        ]
    ].copy()

    results["predicted_type"] = predicted_type
    results["predicted_queue"] = predicted_queue
    results["predicted_priority"] = predicted_priority

    # --------------------------------------------------------
    # TYPE ERRORS
    # --------------------------------------------------------

    type_errors = results[
        results["type"]
        != results["predicted_type"]
    ]

    print()
    print(
        f"Type errors: {len(type_errors):,}"
    )

    if len(type_errors) > 0:
        print()
        print("Examples of Type errors:")
        print("-" * 80)

        for _, row in type_errors.head(10).iterrows():

            print()
            print(
                f"Ticket: {row['ticket_text']}"
            )

            print(
                f"True Type     : {row['type']}"
            )

            print(
                f"Predicted Type: "
                f"{row['predicted_type']}"
            )

    # --------------------------------------------------------
    # QUEUE ERRORS
    # --------------------------------------------------------

    queue_errors = results[
        results["queue"]
        != results["predicted_queue"]
    ]

    print()
    print(
        f"Queue errors: {len(queue_errors):,}"
    )

    if len(queue_errors) > 0:

        print()
        print("Examples of Queue errors:")
        print("-" * 80)

        for _, row in queue_errors.head(15).iterrows():

            print()
            print(
                f"Ticket: {row['ticket_text']}"
            )

            print(
                f"True Type       : {row['type']}"
            )

            print(
                f"Predicted Type  : "
                f"{row['predicted_type']}"
            )

            print(
                f"True Queue      : "
                f"{row['queue']}"
            )

            print(
                f"Predicted Queue : "
                f"{row['predicted_queue']}"
            )

    # --------------------------------------------------------
    # PRIORITY ERRORS
    # --------------------------------------------------------

    priority_errors = results[
        results["priority"]
        != results["predicted_priority"]
    ]

    print()
    print(
        f"Priority errors: {len(priority_errors):,}"
    )

    if len(priority_errors) > 0:

        print()
        print("Examples of Priority errors:")
        print("-" * 80)

        for _, row in priority_errors.head(10).iterrows():

            print()
            print(
                f"Ticket: {row['ticket_text']}"
            )

            print(
                f"True Priority      : "
                f"{row['priority']}"
            )

            print(
                f"Predicted Priority : "
                f"{row['predicted_priority']}"
            )


# ============================================================
# PIPELINE SUMMARY
# ============================================================


def print_pipeline_summary(
    test_df: pd.DataFrame,
    predicted_type: np.ndarray,
    predicted_queue: np.ndarray,
    predicted_priority: np.ndarray,
) -> None:
    """Print compact final summary."""

    print_header("8. FINAL PIPELINE SUMMARY")

    type_accuracy = accuracy_score(
        test_df["type"],
        predicted_type,
    )

    type_f1 = f1_score(
        test_df["type"],
        predicted_type,
        average="macro",
        zero_division=0,
    )

    queue_accuracy = accuracy_score(
        test_df["queue"],
        predicted_queue,
    )

    queue_f1 = f1_score(
        test_df["queue"],
        predicted_queue,
        average="macro",
        zero_division=0,
    )

    priority_accuracy = accuracy_score(
        test_df["priority"],
        predicted_priority,
    )

    priority_f1 = f1_score(
        test_df["priority"],
        predicted_priority,
        average="macro",
        zero_division=0,
    )

    summary = pd.DataFrame(
        {
            "Model": [
                "Type",
                "Queue",
                "Priority",
            ],
            "Accuracy": [
                type_accuracy,
                queue_accuracy,
                priority_accuracy,
            ],
            "Macro F1": [
                type_f1,
                queue_f1,
                priority_f1,
            ],
        }
    )

    summary["Accuracy"] = summary["Accuracy"].map(
        lambda x: f"{x:.4f}"
    )

    summary["Macro F1"] = summary["Macro F1"].map(
        lambda x: f"{x:.4f}"
    )

    print()
    print(summary.to_string(index=False))

    print()
    print("=" * 80)
    print("PIPELINE INTERPRETATION")
    print("=" * 80)

    print()
    print(
        "The Queue score is the most important metric to inspect "
        "because Queue classification is currently the main "
        "weakness of the HARDTEC ML pipeline."
    )

    print()
    print(
        "The Priority result is intentionally evaluated using "
        "predicted Type + predicted Queue, which reproduces "
        "the real production inference chain."
    )

    print()
    print(
        "No existing model files were modified by this script."
    )


# ============================================================
# MAIN
# ============================================================


def main() -> None:
    """Main execution function."""

    print()
    print("=" * 80)
    print("HARDTEC INTELLIGENT TICKETING")
    print("END-TO-END ML PIPELINE EVALUATION")
    print("=" * 80)

    print()
    print(
        "Pipeline:"
    )

    print(
        "Ticket Text"
        " -> Type"
        " -> Queue"
        " -> Priority"
    )

    print()
    print(
        "Existing .pkl models will NOT be modified."
    )

    # --------------------------------------------------------
    # 1. LOAD DATA
    # --------------------------------------------------------

    df = load_dataset()

    # --------------------------------------------------------
    # 2. COMMON SPLIT
    # --------------------------------------------------------

    train_df, test_df = create_split(df)

    # --------------------------------------------------------
    # 3. TYPE
    # --------------------------------------------------------

    type_vectorizer, type_classifier = train_type_model(
        train_df
    )

    predicted_type = predict_type(
        type_vectorizer,
        type_classifier,
        test_df,
    )

    print()
    print(
        "Type prediction completed."
    )

    # --------------------------------------------------------
    # 4. QUEUE
    # --------------------------------------------------------

    (
        queue_vectorizer,
        queue_encoder,
        queue_classifier,
    ) = train_queue_model(
        train_df
    )

    predicted_queue = predict_queue(
        queue_vectorizer,
        queue_encoder,
        queue_classifier,
        test_df,
        predicted_type,
    )

    print()
    print(
        "Queue prediction completed."
    )

    # --------------------------------------------------------
    # 5. PRIORITY
    # --------------------------------------------------------

    (
        priority_vectorizer,
        priority_encoder,
        priority_classifier,
    ) = train_priority_model(
        train_df
    )

    predicted_priority = predict_priority(
        priority_vectorizer,
        priority_encoder,
        priority_classifier,
        test_df,
        predicted_type,
        predicted_queue,
    )

    print()
    print(
        "Priority prediction completed."
    )

    # --------------------------------------------------------
    # 6. EVALUATE
    # --------------------------------------------------------

    evaluate_pipeline(
        test_df,
        predicted_type,
        predicted_queue,
        predicted_priority,
    )

    # --------------------------------------------------------
    # 7. ERROR ANALYSIS
    # --------------------------------------------------------

    analyze_errors(
        test_df,
        predicted_type,
        predicted_queue,
        predicted_priority,
    )

    # --------------------------------------------------------
    # 8. SUMMARY
    # --------------------------------------------------------

    print_pipeline_summary(
        test_df,
        predicted_type,
        predicted_queue,
        predicted_priority,
    )

    print()
    print("=" * 80)
    print("EVALUATION FINISHED SUCCESSFULLY")
    print("=" * 80)
    print()


if __name__ == "__main__":
    main()

