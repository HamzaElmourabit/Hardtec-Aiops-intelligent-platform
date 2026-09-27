
"""
HARDTEC - Queue Classifier V4 Error Analysis
=============================================

Purpose
-------
Analyze the errors made by the existing Queue Classifier V4.

IMPORTANT
---------
This script is READ-ONLY regarding the ML pipeline:
- Does NOT modify ticket_queue_model_v4.pkl
- Does NOT modify the dataset
- Does NOT modify predictor.py
- Does NOT modify Streamlit
- Does NOT retrain the model

Outputs
-------
reports/queue_v4_errors/
    queue_class_performance.csv
    queue_confusion_pairs.csv
    queue_error_summary.csv
    queue_error_examples.csv
    queue_error_keywords.csv

Usage
-----
From the project root:

    python src/ml/analyze_queue_errors_v4.py
"""

from pathlib import Path
from collections import Counter

import joblib
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from scipy.sparse import hstack


# =============================================================================
# CONFIGURATION
# =============================================================================

ROOT_DIR = Path(__file__).resolve().parents[2]

DATA_PATH = ROOT_DIR / "data" / "processed" / "tickets_clean.csv"
MODEL_PATH = ROOT_DIR / "models" / "ticket_queue_model_v4.pkl"

OUTPUT_DIR = ROOT_DIR / "reports" / "queue_v4_errors"

RANDOM_STATE = 42
TEST_SIZE = 0.20

TEXT_COLUMN = "ticket_text"
TYPE_COLUMN = "type"
QUEUE_COLUMN = "queue"


# =============================================================================
# HELPERS
# =============================================================================


def print_section(title: str) -> None:
    """Print a formatted section header."""

    print()
    print("=" * 80)
    print(title)
    print("=" * 80)


def safe_text(value) -> str:
    """Convert a value to clean text."""

    if pd.isna(value):
        return ""

    return str(value).strip()


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Validate and clean the required dataset columns."""

    required_columns = [
        TEXT_COLUMN,
        TYPE_COLUMN,
        QUEUE_COLUMN,
    ]

    missing = [
        column for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    df = df.copy()

    df[TEXT_COLUMN] = df[TEXT_COLUMN].apply(safe_text)
    df[TYPE_COLUMN] = df[TYPE_COLUMN].apply(safe_text)
    df[QUEUE_COLUMN] = df[QUEUE_COLUMN].apply(safe_text)

    df = df[
        (df[TEXT_COLUMN] != "")
        & (df[TYPE_COLUMN] != "")
        & (df[QUEUE_COLUMN] != "")
    ].copy()

    df.reset_index(drop=True, inplace=True)

    return df


# =============================================================================
# LOAD DATA
# =============================================================================


def load_data() -> pd.DataFrame:
    """Load and validate the cleaned dataset."""

    print_section("LOADING DATASET")

    print(f"Dataset path:")
    print(DATA_PATH)

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found:\n{DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)

    print(f"Original dataset size: {len(df):,}")

    df = clean_dataframe(df)

    print(f"Usable dataset size   : {len(df):,}")
    print(f"Queues                : {df[QUEUE_COLUMN].nunique()}")
    print(f"Types                 : {df[TYPE_COLUMN].nunique()}")

    return df


# =============================================================================
# LOAD V4 MODEL
# =============================================================================


def load_v4_model():
    """
    Load the existing V4 queue model.

    Expected structure:

        {
            "tfidf": ...,
            "encoder": ...,
            "classifier": ...
        }
    """

    print_section("LOADING QUEUE MODEL V4")

    print(f"Model path:")
    print(MODEL_PATH)

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"V4 model not found:\n{MODEL_PATH}"
        )

    model_data = joblib.load(MODEL_PATH)

    if not isinstance(model_data, dict):
        raise ValueError(
            "Unexpected V4 model format. Expected a dictionary."
        )

    required_keys = {
        "tfidf",
        "encoder",
        "classifier",
    }

    missing_keys = required_keys - set(model_data.keys())

    if missing_keys:
        raise ValueError(
            f"V4 model is missing keys: {missing_keys}"
        )

    print("V4 model loaded successfully.")

    print(f"TF-IDF : {type(model_data['tfidf']).__name__}")
    print(f"Encoder: {type(model_data['encoder']).__name__}")
    print(f"Model  : {type(model_data['classifier']).__name__}")

    return model_data


# =============================================================================
# COMMON SPLIT
# =============================================================================


def create_common_split(df: pd.DataFrame):
    """
    Create one reproducible train/test split.

    The split is stratified using TYPE + QUEUE when possible.

    The test set is used only for evaluating V4.
    """

    print_section("CREATING VALIDATION SPLIT")

    df = df.copy()

    stratify_labels = (
        df[TYPE_COLUMN].astype(str)
        + " | "
        + df[QUEUE_COLUMN].astype(str)
    )

    counts = stratify_labels.value_counts()

    if counts.min() >= 2:
        stratify = stratify_labels
        stratification_name = "TYPE + QUEUE"
    else:
        stratify = df[TYPE_COLUMN]
        stratification_name = "TYPE"

    train_df, test_df = train_test_split(
        df,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=stratify,
    )

    print(f"Stratification: {stratification_name}")
    print(f"Train size    : {len(train_df):,}")
    print(f"Test size     : {len(test_df):,}")

    return train_df, test_df


# =============================================================================
# BUILD V4 FEATURES
# =============================================================================


def build_v4_features(model_data, df: pd.DataFrame):
    """
    Reproduce the feature construction used by predictor.py.

    Queue V4 uses:

        ticket text TF-IDF
        +
        predicted/known type encoded with OneHotEncoder
    """

    tfidf = model_data["tfidf"]
    encoder = model_data["encoder"]

    text_vector = tfidf.transform(
        df[TEXT_COLUMN].tolist()
    )

    type_df = pd.DataFrame(
        {
            TYPE_COLUMN: df[TYPE_COLUMN].tolist()
        }
    )

    type_vector = encoder.transform(type_df)

    final_vector = hstack(
        [
            text_vector,
            type_vector,
        ]
    ).tocsr()

    return final_vector


# =============================================================================
# PREDICT QUEUE
# =============================================================================


def predict_queue(model_data, df: pd.DataFrame):
    """
    Predict queues using the existing V4 model.

    The evaluation intentionally uses the TRUE ticket type
    because this script focuses specifically on Queue V4.

    This isolates queue classification quality from Type model errors.
    """

    print_section("GENERATING V4 QUEUE PREDICTIONS")

    classifier = model_data["classifier"]

    features = build_v4_features(
        model_data,
        df,
    )

    predictions = classifier.predict(features)

    predictions = np.asarray(predictions).astype(str)

    print(f"Predictions generated: {len(predictions):,}")

    return predictions


# =============================================================================
# CLASS PERFORMANCE
# =============================================================================


def generate_class_performance(
    y_true,
    y_pred,
    output_dir: Path,
):
    """Generate per-queue precision, recall and F1."""

    print_section("PER-QUEUE PERFORMANCE")

    labels = sorted(
        set(y_true) | set(y_pred)
    )

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        zero_division=0,
    )

    result = pd.DataFrame(
        {
            "queue": labels,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }
    )

    result["error_count"] = (
        result["support"]
        * (1 - result["recall"])
    ).round().astype(int)

    result = result.sort_values(
        by="f1",
        ascending=True,
    )

    output_path = output_dir / "queue_class_performance.csv"

    result.to_csv(
        output_path,
        index=False,
    )

    print(result.to_string(index=False))

    print()
    print(f"Saved: {output_path}")

    return result


# =============================================================================
# CONFUSION PAIRS
# =============================================================================


def generate_confusion_pairs(
    y_true,
    y_pred,
    output_dir: Path,
):
    """
    Identify the most frequent incorrect transitions:

        TRUE QUEUE -> PREDICTED QUEUE
    """

    print_section("TOP QUEUE CONFUSION PAIRS")

    data = pd.DataFrame(
        {
            "true_queue": y_true,
            "predicted_queue": y_pred,
        }
    )

    errors = data[
        data["true_queue"]
        != data["predicted_queue"]
    ].copy()

    pairs = (
        errors.groupby(
            [
                "true_queue",
                "predicted_queue",
            ]
        )
        .size()
        .reset_index(name="error_count")
        .sort_values(
            "error_count",
            ascending=False,
        )
    )

    total_errors = len(errors)

    if total_errors > 0:
        pairs["percentage_of_all_errors"] = (
            pairs["error_count"]
            / total_errors
            * 100
        )
    else:
        pairs["percentage_of_all_errors"] = 0.0

    output_path = output_dir / "queue_confusion_pairs.csv"

    pairs.to_csv(
        output_path,
        index=False,
    )

    print(
        pairs.head(30).to_string(index=False)
    )

    print()
    print(f"Total queue errors: {total_errors:,}")
    print(f"Saved: {output_path}")

    return pairs


# =============================================================================
# ERROR SUMMARY
# =============================================================================


def generate_error_summary(
    df: pd.DataFrame,
    y_true,
    y_pred,
    output_dir: Path,
):
    """
    Produce a queue-level summary showing:

    - total tickets
    - correct predictions
    - errors
    - accuracy
    - most common wrong destination
    """

    print_section("QUEUE ERROR SUMMARY")

    work = df.copy()

    work["true_queue"] = y_true
    work["predicted_queue"] = y_pred

    work["correct"] = (
        work["true_queue"]
        == work["predicted_queue"]
    )

    rows = []

    for queue in sorted(work["true_queue"].unique()):

        queue_data = work[
            work["true_queue"] == queue
        ]

        total = len(queue_data)

        correct = int(
            queue_data["correct"].sum()
        )

        errors = total - correct

        accuracy = (
            correct / total
            if total > 0
            else 0
        )

        wrong_predictions = queue_data[
            ~queue_data["correct"]
        ]["predicted_queue"]

        if len(wrong_predictions) > 0:
            most_common_wrong = (
                wrong_predictions
                .value_counts()
                .index[0]
            )

            most_common_wrong_count = int(
                wrong_predictions
                .value_counts()
                .iloc[0]
            )
        else:
            most_common_wrong = ""
            most_common_wrong_count = 0

        rows.append(
            {
                "true_queue": queue,
                "total_tickets": total,
                "correct_predictions": correct,
                "error_count": errors,
                "accuracy": accuracy,
                "most_common_wrong_prediction":
                    most_common_wrong,
                "most_common_wrong_count":
                    most_common_wrong_count,
            }
        )

    summary = pd.DataFrame(rows)

    summary = summary.sort_values(
        by="error_count",
        ascending=False,
    )

    output_path = output_dir / "queue_error_summary.csv"

    summary.to_csv(
        output_path,
        index=False,
    )

    print(
        summary.to_string(index=False)
    )

    print()
    print(f"Saved: {output_path}")

    return summary


# =============================================================================
# ERROR EXAMPLES
# =============================================================================


def generate_error_examples(
    df: pd.DataFrame,
    y_true,
    y_pred,
    output_dir: Path,
):
    """
    Save representative real ticket examples for each
    true -> predicted queue confusion.

    Maximum:
        10 examples per confusion pair.
    """

    print_section("EXTRACTING ERROR EXAMPLES")

    work = df.copy()

    work["true_queue"] = y_true
    work["predicted_queue"] = y_pred

    errors = work[
        work["true_queue"]
        != work["predicted_queue"]
    ].copy()

    if errors.empty:
        print("No errors found.")
        return pd.DataFrame()

    error_groups = (
        errors.groupby(
            [
                "true_queue",
                "predicted_queue",
            ]
        )
        .size()
        .reset_index(name="error_count")
        .sort_values(
            "error_count",
            ascending=False,
        )
    )

    selected_groups = error_groups.head(30)

    examples = []

    for _, group in selected_groups.iterrows():

        true_queue = group["true_queue"]
        predicted_queue = group["predicted_queue"]

        subset = errors[
            (errors["true_queue"] == true_queue)
            & (
                errors["predicted_queue"]
                == predicted_queue
            )
        ].copy()

        subset = subset.head(10)

        for _, row in subset.iterrows():

            examples.append(
                {
                    "true_queue": true_queue,
                    "predicted_queue": predicted_queue,
                    "type": row[TYPE_COLUMN],
                    "ticket_text": row[TEXT_COLUMN],
                }
            )

    result = pd.DataFrame(examples)

    output_path = output_dir / "queue_error_examples.csv"

    result.to_csv(
        output_path,
        index=False,
    )

    print(
        f"Selected confusion pairs: "
        f"{len(selected_groups)}"
    )

    print(
        f"Saved examples: {len(result):,}"
    )

    print()
    print(f"Saved: {output_path}")

    return result


# =============================================================================
# ERROR KEYWORDS
# =============================================================================


def generate_error_keywords(
    df: pd.DataFrame,
    y_true,
    y_pred,
    output_dir: Path,
):
    """
    Extract frequent TF-IDF terms from error tickets.

    This is a diagnostic report only.

    It helps identify recurring vocabulary associated with
    queue classification failures.
    """

    print_section("ANALYZING ERROR KEYWORDS")

    work = df.copy()

    work["true_queue"] = y_true
    work["predicted_queue"] = y_pred

    errors = work[
        work["true_queue"]
        != work["predicted_queue"]
    ].copy()

    if errors.empty:
        print("No errors found.")
        return pd.DataFrame()

    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        min_df=2,
        max_features=10000,
        sublinear_tf=True,
    )

    matrix = vectorizer.fit_transform(
        errors[TEXT_COLUMN].tolist()
    )

    feature_names = np.asarray(
        vectorizer.get_feature_names_out()
    )

    rows = []

    for queue in sorted(
        errors["true_queue"].unique()
    ):

        queue_mask = (
            errors["true_queue"].values
            == queue
        )

        queue_matrix = matrix[queue_mask]

        if queue_matrix.shape[0] == 0:
            continue

        scores = np.asarray(
            queue_matrix.mean(axis=0)
        ).ravel()

        top_indices = np.argsort(
            scores
        )[::-1][:30]

        for rank, index in enumerate(
            top_indices,
            start=1,
        ):

            score = scores[index]

            if score <= 0:
                continue

            rows.append(
                {
                    "true_queue": queue,
                    "rank": rank,
                    "keyword": feature_names[index],
                    "tfidf_score": float(score),
                }
            )

    result = pd.DataFrame(rows)

    output_path = (
        output_dir
        / "queue_error_keywords.csv"
    )

    result.to_csv(
        output_path,
        index=False,
    )

    print(
        result.head(60).to_string(index=False)
    )

    print()
    print(f"Saved: {output_path}")

    return result


# =============================================================================
# GLOBAL METRICS
# =============================================================================


def print_global_metrics(
    y_true,
    y_pred,
):
    """Print overall queue metrics."""

    print_section("GLOBAL QUEUE V4 METRICS")

    accuracy = accuracy_score(
        y_true,
        y_pred,
    )

    precision, recall, f1, _ = (
        precision_recall_fscore_support(
            y_true,
            y_pred,
            average="macro",
            zero_division=0,
        )
    )

    weighted_precision, weighted_recall, weighted_f1, _ = (
        precision_recall_fscore_support(
            y_true,
            y_pred,
            average="weighted",
            zero_division=0,
        )
    )

    print(
        f"Accuracy        : {accuracy:.4f}"
    )

    print(
        f"Precision Macro : {precision:.4f}"
    )

    print(
        f"Recall Macro    : {recall:.4f}"
    )

    print(
        f"Macro F1        : {f1:.4f}"
    )

    print(
        f"Weighted F1     : {weighted_f1:.4f}"
    )

    return {
        "accuracy": accuracy,
        "precision_macro": precision,
        "recall_macro": recall,
        "macro_f1": f1,
        "weighted_f1": weighted_f1,
    }


# =============================================================================
# CLASSIFICATION REPORT
# =============================================================================


def save_classification_report(
    y_true,
    y_pred,
    output_dir: Path,
):
    """Save sklearn classification report."""

    report = classification_report(
        y_true,
        y_pred,
        output_dict=True,
        zero_division=0,
    )

    report_df = (
        pd.DataFrame(report)
        .transpose()
        .reset_index()
        .rename(
            columns={
                "index": "label"
            }
        )
    )

    output_path = (
        output_dir
        / "classification_report_v4.csv"
    )

    report_df.to_csv(
        output_path,
        index=False,
    )

    print()
    print(
        f"Saved: {output_path}"
    )


# =============================================================================
# CONFUSION MATRIX
# =============================================================================


def save_confusion_matrix(
    y_true,
    y_pred,
    output_dir: Path,
):
    """Save the complete queue confusion matrix."""

    labels = sorted(
        set(y_true) | set(y_pred)
    )

    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=labels,
    )

    matrix_df = pd.DataFrame(
        matrix,
        index=[
            f"TRUE: {label}"
            for label in labels
        ],
        columns=[
            f"PRED: {label}"
            for label in labels
        ],
    )

    output_path = (
        output_dir
        / "confusion_matrix_v4.csv"
    )

    matrix_df.to_csv(
        output_path
    )

    print(
        f"Saved: {output_path}"
    )


# =============================================================================
# MAIN
# =============================================================================


def main():

    print()
    print("=" * 80)
    print("HARDTEC QUEUE CLASSIFIER V4 ERROR ANALYSIS")
    print("=" * 80)

    print()
    print("IMPORTANT:")
    print("- Existing V4 model will NOT be modified.")
    print("- Dataset will NOT be modified.")
    print("- predictor.py will NOT be modified.")
    print("- This is a diagnostic analysis only.")

    # -------------------------------------------------------------------------
    # Create output directory
    # -------------------------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -------------------------------------------------------------------------
    # Load dataset
    # -------------------------------------------------------------------------

    df = load_data()

    # -------------------------------------------------------------------------
    # Load V4
    # -------------------------------------------------------------------------

    model_data = load_v4_model()

    # -------------------------------------------------------------------------
    # Create reproducible split
    # -------------------------------------------------------------------------

    train_df, test_df = create_common_split(
        df
    )

    # -------------------------------------------------------------------------
    # Predict test queues
    # -------------------------------------------------------------------------

    y_true = test_df[
        QUEUE_COLUMN
    ].astype(str).values

    y_pred = predict_queue(
        model_data,
        test_df,
    )

    # -------------------------------------------------------------------------
    # Global metrics
    # -------------------------------------------------------------------------

    metrics = print_global_metrics(
        y_true,
        y_pred,
    )

    # -------------------------------------------------------------------------
    # Classification report
    # -------------------------------------------------------------------------

    save_classification_report(
        y_true,
        y_pred,
        OUTPUT_DIR,
    )

    # -------------------------------------------------------------------------
    # Confusion matrix
    # -------------------------------------------------------------------------

    save_confusion_matrix(
        y_true,
        y_pred,
        OUTPUT_DIR,
    )

    # -------------------------------------------------------------------------
    # Per-class performance
    # -------------------------------------------------------------------------

    generate_class_performance(
        y_true,
        y_pred,
        OUTPUT_DIR,
    )

    # -------------------------------------------------------------------------
    # Confusion pairs
    # -------------------------------------------------------------------------

    generate_confusion_pairs(
        y_true,
        y_pred,
        OUTPUT_DIR,
    )

    # -------------------------------------------------------------------------
    # Error summary
    # -------------------------------------------------------------------------

    generate_error_summary(
        test_df,
        y_true,
        y_pred,
        OUTPUT_DIR,
    )

    # -------------------------------------------------------------------------
    # Error examples
    # -------------------------------------------------------------------------

    generate_error_examples(
        test_df,
        y_true,
        y_pred,
        OUTPUT_DIR,
    )

    # -------------------------------------------------------------------------
    # Error keywords
    # -------------------------------------------------------------------------

    generate_error_keywords(
        test_df,
        y_true,
        y_pred,
        OUTPUT_DIR,
    )

    # -------------------------------------------------------------------------
    # Final summary
    # -------------------------------------------------------------------------

    print_section("FINAL SUMMARY")

    total = len(y_true)

    errors = int(
        np.sum(
            y_true != y_pred
        )
    )

    correct = total - errors

    print(
        f"Test tickets : {total:,}"
    )

    print(
        f"Correct      : {correct:,}"
    )

    print(
        f"Errors       : {errors:,}"
    )

    print(
        f"Accuracy     : {metrics['accuracy']:.4f}"
    )

    print(
        f"Macro F1     : {metrics['macro_f1']:.4f}"
    )

    print()
    print(
        "All reports saved to:"
    )

    print(
        OUTPUT_DIR
    )

    print()
    print("=" * 80)
    print("ANALYSIS FINISHED")
    print("=" * 80)


if __name__ == "__main__":
    main()

