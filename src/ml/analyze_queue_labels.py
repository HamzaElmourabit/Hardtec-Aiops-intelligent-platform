# ============================================================
# HARDTEC - Queue Label Quality Analysis
# ============================================================
#
# Objectif :
#   Analyser la cohérence des labels "queue" avant de modifier
#   ou réentraîner le modèle de classification.
#
# Le script produit :
#   1. Distribution des queues
#   2. Distribution Queue x Type
#   3. Profils TF-IDF des queues
#   4. Tickets potentiellement incohérents
#   5. Exemples représentatifs par queue
#   6. Rapport CSV exploitable
#
# IMPORTANT :
#   Les tickets "suspects" ne sont PAS automatiquement considérés
#   comme mal étiquetés. Il s'agit d'un signal de cohérence
#   sémantique qui doit être inspecté humainement.
#
# ============================================================

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = PROJECT_ROOT / "data" / "processed" / "tickets_clean.csv"

REPORT_DIR = PROJECT_ROOT / "reports" / "queue_label_analysis"

REPORT_DIR.mkdir(parents=True, exist_ok=True)

TEXT_COLUMN = "ticket_text"
QUEUE_COLUMN = "queue"
TYPE_COLUMN = "type"

TOP_EXAMPLES_PER_QUEUE = 10
TOP_SUSPICIOUS_TICKETS = 300

RANDOM_STATE = 42


# ============================================================
# LOAD DATA
# ============================================================

def load_dataset() -> pd.DataFrame:

    print("=" * 70)
    print("HARDTEC - QUEUE LABEL QUALITY ANALYSIS")
    print("=" * 70)

    print("\n[1/7] Loading dataset...")

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found:\n{DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)

    required_columns = [
        TEXT_COLUMN,
        QUEUE_COLUMN,
        TYPE_COLUMN,
    ]

    missing = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing columns: {missing}"
        )

    df = df.copy()

    df[TEXT_COLUMN] = (
        df[TEXT_COLUMN]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    df[QUEUE_COLUMN] = (
        df[QUEUE_COLUMN]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    df[TYPE_COLUMN] = (
        df[TYPE_COLUMN]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    df = df[
        (df[TEXT_COLUMN] != "")
        & (df[QUEUE_COLUMN] != "")
        & (df[TYPE_COLUMN] != "")
    ].reset_index(drop=True)

    print(f"Dataset rows: {len(df):,}")
    print(f"Queues: {df[QUEUE_COLUMN].nunique()}")
    print(f"Types: {df[TYPE_COLUMN].nunique()}")

    return df


# ============================================================
# QUEUE DISTRIBUTION
# ============================================================

def analyze_queue_distribution(df: pd.DataFrame):

    print("\n[2/7] Queue distribution...")

    counts = (
        df[QUEUE_COLUMN]
        .value_counts()
        .rename_axis("queue")
        .reset_index(name="ticket_count")
    )

    counts["percentage"] = (
        counts["ticket_count"]
        / len(df)
        * 100
    )

    counts.to_csv(
        REPORT_DIR / "queue_distribution.csv",
        index=False
    )

    print("\nQueue distribution:")
    print(counts.to_string(index=False))

    return counts


# ============================================================
# QUEUE x TYPE
# ============================================================

def analyze_queue_by_type(df: pd.DataFrame):

    print("\n[3/7] Queue x Type analysis...")

    cross = pd.crosstab(
        df[QUEUE_COLUMN],
        df[TYPE_COLUMN]
    )

    cross.to_csv(
        REPORT_DIR / "queue_by_type_counts.csv"
    )

    cross_percentage = pd.crosstab(
        df[QUEUE_COLUMN],
        df[TYPE_COLUMN],
        normalize="index"
    ) * 100

    cross_percentage.to_csv(
        REPORT_DIR / "queue_by_type_percentage.csv"
    )

    print("\nQueue x Type:")
    print(cross.to_string())

    print("\nQueue x Type (% within queue):")
    print(
        cross_percentage
        .round(2)
        .to_string()
    )

    return cross, cross_percentage


# ============================================================
# TF-IDF REPRESENTATION
# ============================================================

def build_tfidf(df: pd.DataFrame):

    print("\n[4/7] Building TF-IDF representation...")

    vectorizer = TfidfVectorizer(
        lowercase=True,
        max_features=40000,
        ngram_range=(1, 2),
        min_df=2,
        sublinear_tf=True,
        stop_words="english",
    )

    X = vectorizer.fit_transform(
        df[TEXT_COLUMN]
    )

    print(
        f"TF-IDF matrix: "
        f"{X.shape[0]:,} documents x "
        f"{X.shape[1]:,} features"
    )

    return vectorizer, X


# ============================================================
# QUEUE CENTROIDS
# ============================================================

def calculate_queue_profiles(
    df: pd.DataFrame,
    X,
):

    print("\n[5/7] Calculating queue semantic profiles...")

    queues = sorted(
        df[QUEUE_COLUMN].unique()
    )

    queue_profiles = {}

    for queue in queues:

        mask = (
            df[QUEUE_COLUMN].values
            == queue
        )

        X_queue = X[mask]

        centroid = X_queue.mean(axis=0)

        queue_profiles[queue] = centroid

    return queue_profiles


# ============================================================
# SEMANTIC CONSISTENCY
# ============================================================

def calculate_semantic_consistency(
    df: pd.DataFrame,
    X,
    queue_profiles,
):

    print("\n[6/7] Calculating semantic consistency...")

    queues = list(queue_profiles.keys())

    profile_matrix = np.vstack(
        [
            np.asarray(
                queue_profiles[q]
            ).ravel()
            for q in queues
        ]
    )

    similarities = cosine_similarity(
        X,
        profile_matrix
    )

    assigned_queue_indices = [
        queues.index(queue)
        for queue in df[QUEUE_COLUMN]
    ]

    assigned_scores = similarities[
        np.arange(len(df)),
        assigned_queue_indices
    ]

    best_indices = similarities.argmax(
        axis=1
    )

    best_scores = similarities[
        np.arange(len(df)),
        best_indices
    ]

    best_queues = [
        queues[i]
        for i in best_indices
    ]

    margins = (
        best_scores
        - assigned_scores
    )

    result = df[
        [
            TEXT_COLUMN,
            TYPE_COLUMN,
            QUEUE_COLUMN,
        ]
    ].copy()

    result["assigned_queue_similarity"] = (
        assigned_scores
    )

    result["best_queue"] = best_queues

    result["best_queue_similarity"] = (
        best_scores
    )

    result["similarity_margin"] = margins

    result["potentially_inconsistent"] = (
        (
            result["best_queue"]
            != result[QUEUE_COLUMN]
        )
        &
        (
            result["similarity_margin"]
            >= 0.10
        )
    )

    result = result.sort_values(
        "similarity_margin",
        ascending=False
    )

    return result


# ============================================================
# SUSPICIOUS LABELS
# ============================================================

def save_suspicious_tickets(
    consistency_df: pd.DataFrame
):

    print("\nFinding potentially inconsistent labels...")

    suspicious = (
        consistency_df[
            consistency_df[
                "potentially_inconsistent"
            ]
        ]
        .head(TOP_SUSPICIOUS_TICKETS)
        .copy()
    )

    suspicious.to_csv(
        REPORT_DIR / "suspicious_queue_labels.csv",
        index=False
    )

    print(
        f"\nPotentially inconsistent tickets: "
        f"{len(suspicious):,}"
    )

    if len(suspicious) > 0:

        display_columns = [
            TEXT_COLUMN,
            TYPE_COLUMN,
            QUEUE_COLUMN,
            "best_queue",
            "assigned_queue_similarity",
            "best_queue_similarity",
            "similarity_margin",
        ]

        print(
            "\nTop suspicious examples:\n"
        )

        print(
            suspicious[
                display_columns
            ]
            .head(30)
            .to_string(index=False)
        )

    return suspicious


# ============================================================
# REPRESENTATIVE EXAMPLES
# ============================================================

def save_representative_examples(
    df: pd.DataFrame,
    X,
):

    print(
        "\nCreating representative examples "
        "for each queue..."
    )

    queues = sorted(
        df[QUEUE_COLUMN].unique()
    )

    rows = []

    for queue in queues:

        indices = np.where(
            df[QUEUE_COLUMN].values
            == queue
        )[0]

        if len(indices) == 0:
            continue

        X_queue = X[indices]

        centroid = np.asarray(
            X_queue.mean(axis=0)
        )

        similarities = cosine_similarity(
            X_queue,
            centroid
        ).ravel()

        top_local_indices = (
            similarities
            .argsort()[::-1]
            [:TOP_EXAMPLES_PER_QUEUE]
        )

        for local_idx in top_local_indices:

            original_idx = indices[
                local_idx
            ]

            rows.append(
                {
                    "queue": queue,
                    "type": df.iloc[
                        original_idx
                    ][TYPE_COLUMN],
                    "ticket_text": df.iloc[
                        original_idx
                    ][TEXT_COLUMN],
                    "similarity_to_queue_profile":
                        similarities[local_idx],
                }
            )

    representative_df = pd.DataFrame(rows)

    representative_df.to_csv(
        REPORT_DIR
        / "representative_examples_by_queue.csv",
        index=False
    )

    return representative_df


# ============================================================
# MOST AMBIGUOUS QUEUE PAIRS
# ============================================================

def analyze_queue_profile_similarity(
    queue_profiles
):

    print(
        "\nAnalyzing similarity between queue profiles..."
    )

    queues = list(queue_profiles.keys())

    matrix = np.vstack(
        [
            np.asarray(
                queue_profiles[q]
            ).ravel()
            for q in queues
        ]
    )

    similarity_matrix = cosine_similarity(
        matrix
    )

    rows = []

    for i in range(len(queues)):

        for j in range(i + 1, len(queues)):

            rows.append(
                {
                    "queue_1": queues[i],
                    "queue_2": queues[j],
                    "cosine_similarity":
                        similarity_matrix[i, j],
                }
            )

    pair_df = pd.DataFrame(rows)

    pair_df = pair_df.sort_values(
        "cosine_similarity",
        ascending=False
    )

    pair_df.to_csv(
        REPORT_DIR
        / "queue_profile_similarity.csv",
        index=False
    )

    print(
        "\nMost semantically similar queue pairs:"
    )

    print(
        pair_df
        .head(20)
        .to_string(index=False)
    )

    return pair_df


# ============================================================
# SUMMARY
# ============================================================

def generate_summary(
    df,
    consistency_df,
    queue_counts,
    pair_df,
):

    print("\n" + "=" * 70)
    print("FINAL LABEL QUALITY SUMMARY")
    print("=" * 70)

    suspicious_count = int(
        consistency_df[
            "potentially_inconsistent"
        ].sum()
    )

    suspicious_percentage = (
        suspicious_count
        / len(df)
        * 100
    )

    print(
        f"\nTotal tickets analyzed : {len(df):,}"
    )

    print(
        f"Total queues          : "
        f"{df[QUEUE_COLUMN].nunique()}"
    )

    print(
        f"Potentially inconsistent: "
        f"{suspicious_count:,} "
        f"({suspicious_percentage:.2f}%)"
    )

    print(
        "\nMost ambiguous queue pairs:"
    )

    print(
        pair_df.head(5)
        .to_string(index=False)
    )

    print(
        "\nReports saved in:"
    )

    print(
        REPORT_DIR
    )

    print("\nGenerated files:")

    for file in sorted(
        REPORT_DIR.glob("*.csv")
    ):
        print(
            f"  - {file.name}"
        )


# ============================================================
# MAIN
# ============================================================

def main():

    df = load_dataset()

    queue_counts = (
        analyze_queue_distribution(df)
    )

    analyze_queue_by_type(df)

    vectorizer, X = build_tfidf(df)

    queue_profiles = (
        calculate_queue_profiles(
            df,
            X
        )
    )

    consistency_df = (
        calculate_semantic_consistency(
            df,
            X,
            queue_profiles
        )
    )

    save_suspicious_tickets(
        consistency_df
    )

    save_representative_examples(
        df,
        X
    )

    pair_df = (
        analyze_queue_profile_similarity(
            queue_profiles
        )
    )

    generate_summary(
        df,
        consistency_df,
        queue_counts,
        pair_df,
    )

    print(
        "\nAnalysis completed successfully."
    )


if __name__ == "__main__":
    main()