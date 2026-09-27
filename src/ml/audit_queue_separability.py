from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# CONFIGURATION
# ============================================================

ROOT_DIR = Path(__file__).resolve().parents[2]

DATA_PATH = ROOT_DIR / "data" / "processed" / "tickets_clean.csv"

REPORT_DIR = ROOT_DIR / "reports" / "queue_separability"

RANDOM_STATE = 42

TARGET_QUEUES = [
    "Technical Support",
    "Product Support",
    "IT Support",
    "Customer Service",
    "Returns and Exchanges",
]


# ============================================================
# LOAD DATA
# ============================================================

def load_data():
    print("=" * 80)
    print("QUEUE LABEL SEPARABILITY AUDIT")
    print("=" * 80)

    print("\nLoading dataset:")
    print(DATA_PATH)

    df = pd.read_csv(DATA_PATH)

    required_columns = [
        "ticket_text",
        "type",
        "queue",
    ]

    missing = [
        col
        for col in required_columns
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    df = df.dropna(
        subset=[
            "ticket_text",
            "queue",
        ]
    ).copy()

    df["ticket_text"] = (
        df["ticket_text"]
        .astype(str)
        .str.strip()
    )

    df["queue"] = (
        df["queue"]
        .astype(str)
        .str.strip()
    )

    df["type"] = (
        df["type"]
        .astype(str)
        .str.strip()
    )

    print(
        f"\nDataset size: {len(df):,} tickets"
    )

    return df


# ============================================================
# QUEUE DISTRIBUTION
# ============================================================

def analyze_distribution(df):

    print("\n" + "=" * 80)
    print("1. QUEUE DISTRIBUTION")
    print("=" * 80)

    distribution = (
        df["queue"]
        .value_counts()
        .rename_axis("queue")
        .reset_index(name="ticket_count")
    )

    distribution["percentage"] = (
        distribution["ticket_count"]
        / len(df)
        * 100
    )

    print(
        distribution.to_string(
            index=False
        )
    )

    return distribution


# ============================================================
# QUEUE × TYPE
# ============================================================

def analyze_queue_type(df):

    print("\n" + "=" * 80)
    print("2. QUEUE × TYPE")
    print("=" * 80)

    counts = pd.crosstab(
        df["queue"],
        df["type"]
    )

    percentages = (
        pd.crosstab(
            df["queue"],
            df["type"],
            normalize="index"
        )
        * 100
    )

    print("\nCounts:")
    print(counts)

    print("\nPercentages:")
    print(
        percentages.round(2)
    )

    return counts, percentages


# ============================================================
# TF-IDF
# ============================================================

def build_tfidf(df):

    print("\n" + "=" * 80)
    print("3. TF-IDF REPRESENTATION")
    print("=" * 80)

    vectorizer = TfidfVectorizer(
        lowercase=True,
        max_features=40000,
        ngram_range=(1, 2),
        min_df=2,
        sublinear_tf=True,
        stop_words="english",
    )

    matrix = vectorizer.fit_transform(
        df["ticket_text"]
    )

    print(
        f"TF-IDF matrix: "
        f"{matrix.shape[0]:,} documents × "
        f"{matrix.shape[1]:,} features"
    )

    return vectorizer, matrix


# ============================================================
# QUEUE CENTROIDS
# ============================================================

def compute_centroids(df, matrix):

    print("\n" + "=" * 80)
    print("4. QUEUE CENTROID SIMILARITY")
    print("=" * 80)

    centroids = {}

    for queue in sorted(
        df["queue"].unique()
    ):

        indices = np.where(
            df["queue"].values == queue
        )[0]

        # Convert np.matrix -> np.ndarray
        centroid = np.asarray(
            matrix[indices].mean(axis=0)
        )

        centroids[queue] = centroid

    queues = sorted(
        centroids.keys()
    )

    rows = []

    for q1, q2 in combinations(
        queues,
        2
    ):

        similarity = cosine_similarity(
            centroids[q1],
            centroids[q2]
        )[0][0]

        rows.append(
            {
                "queue_1": q1,
                "queue_2": q2,
                "cosine_similarity": float(
                    similarity
                ),
            }
        )

    result = (
        pd.DataFrame(rows)
        .sort_values(
            "cosine_similarity",
            ascending=False
        )
        .reset_index(drop=True)
    )

    print("\nMost similar queues:")

    print(
        result.head(20).to_string(
            index=False
        )
    )

    return result, centroids


# ============================================================
# TOP KEYWORDS PER QUEUE
# ============================================================

def extract_queue_keywords(
    df,
    vectorizer,
    matrix,
    top_n=15,
):

    print("\n" + "=" * 80)
    print("5. TOP TF-IDF FEATURES PER QUEUE")
    print("=" * 80)

    feature_names = np.array(
        vectorizer.get_feature_names_out()
    )

    rows = []

    for queue in sorted(
        df["queue"].unique()
    ):

        indices = np.where(
            df["queue"].values == queue
        )[0]

        queue_matrix = matrix[
            indices
        ]

        mean_scores = np.asarray(
            queue_matrix.mean(axis=0)
        ).ravel()

        top_indices = (
            mean_scores.argsort()[
                -top_n:
            ][::-1]
        )

        keywords = [
            feature_names[i]
            for i in top_indices
            if mean_scores[i] > 0
        ]

        rows.append(
            {
                "queue": queue,
                "top_keywords": ", ".join(
                    keywords
                ),
            }
        )

        print(
            f"\n{queue}:"
        )

        print(
            ", ".join(keywords)
        )

    return pd.DataFrame(rows)


# ============================================================
# CROSS-QUEUE NEAR DUPLICATES
# ============================================================

def find_cross_queue_similar_tickets(
    df,
    matrix,
    top_n=100,
    threshold=0.75,
):

    print("\n" + "=" * 80)
    print(
        "6. CROSS-QUEUE SEMANTICALLY "
        "SIMILAR TICKETS"
    )
    print("=" * 80)

    similarities = cosine_similarity(
        matrix
    )

    rows = []

    n = len(df)

    for i in range(n):

        candidate_indices = np.where(
            similarities[
                i,
                i + 1:
            ] >= threshold
        )[0]

        for relative_j in candidate_indices:

            j = i + 1 + relative_j

            if (
                df.iloc[i]["queue"]
                ==
                df.iloc[j]["queue"]
            ):
                continue

            similarity = similarities[
                i,
                j
            ]

            rows.append(
                {
                    "similarity": float(
                        similarity
                    ),
                    "queue_1": df.iloc[i][
                        "queue"
                    ],
                    "queue_2": df.iloc[j][
                        "queue"
                    ],
                    "type_1": df.iloc[i][
                        "type"
                    ],
                    "type_2": df.iloc[j][
                        "type"
                    ],
                    "ticket_1": df.iloc[i][
                        "ticket_text"
                    ],
                    "ticket_2": df.iloc[j][
                        "ticket_text"
                    ],
                }
            )

    if rows:

        result = (
            pd.DataFrame(rows)
            .sort_values(
                "similarity",
                ascending=False
            )
            .head(top_n)
        )

    else:

        result = pd.DataFrame(
            columns=[
                "similarity",
                "queue_1",
                "queue_2",
                "type_1",
                "type_2",
                "ticket_1",
                "ticket_2",
            ]
        )

    print(
        f"\nFound {len(rows):,} "
        f"cross-queue pairs above similarity "
        f"threshold {threshold}."
    )

    if len(result) > 0:

        print(
            "\nTop examples:"
        )

        for _, row in (
            result.head(20).iterrows()
        ):

            print(
                "\n" + "-" * 80
            )

            print(
                f"Similarity: "
                f"{row['similarity']:.4f}"
            )

            print(
                f"Queue 1: "
                f"{row['queue_1']}"
            )

            print(
                f"Queue 2: "
                f"{row['queue_2']}"
            )

            print(
                f"Ticket 1: "
                f"{row['ticket_1'][:300]}"
            )

            print(
                f"Ticket 2: "
                f"{row['ticket_2'][:300]}"
            )

    return result


# ============================================================
# TARGET QUEUE PAIR ANALYSIS
# ============================================================

def analyze_target_pairs(
    similarity_df
):

    print("\n" + "=" * 80)
    print("7. TARGET QUEUE PAIRS")
    print("=" * 80)

    rows = []

    for q1, q2 in combinations(
        TARGET_QUEUES,
        2
    ):

        direct = similarity_df[
            (
                (
                    similarity_df["queue_1"]
                    == q1
                )
                &
                (
                    similarity_df["queue_2"]
                    == q2
                )
            )
            |
            (
                (
                    similarity_df["queue_1"]
                    == q2
                )
                &
                (
                    similarity_df["queue_2"]
                    == q1
                )
            )
        ]

        if len(direct) == 0:
            continue

        similarity = direct.iloc[0][
            "cosine_similarity"
        ]

        rows.append(
            {
                "queue_1": q1,
                "queue_2": q2,
                "cosine_similarity": float(
                    similarity
                ),
            }
        )

    result = (
        pd.DataFrame(rows)
        .sort_values(
            "cosine_similarity",
            ascending=False
        )
        .reset_index(drop=True)
    )

    print(
        result.to_string(
            index=False
        )
    )

    return result


# ============================================================
# SAVE REPORTS
# ============================================================

def save_reports(
    distribution,
    counts,
    percentages,
    similarity_df,
    keywords_df,
    similar_tickets,
    target_pairs,
):

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    distribution.to_csv(
        REPORT_DIR
        / "queue_distribution.csv",
        index=False
    )

    counts.to_csv(
        REPORT_DIR
        / "queue_type_counts.csv"
    )

    percentages.to_csv(
        REPORT_DIR
        / "queue_type_percentages.csv"
    )

    similarity_df.to_csv(
        REPORT_DIR
        / "queue_centroid_similarity.csv",
        index=False
    )

    keywords_df.to_csv(
        REPORT_DIR
        / "queue_keywords.csv",
        index=False
    )

    similar_tickets.to_csv(
        REPORT_DIR
        / "cross_queue_similar_tickets.csv",
        index=False
    )

    target_pairs.to_csv(
        REPORT_DIR
        / "target_queue_pairs.csv",
        index=False
    )

    print("\n" + "=" * 80)
    print("REPORTS SAVED")
    print("=" * 80)

    print(
        f"\n{REPORT_DIR}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    df = load_data()

    distribution = (
        analyze_distribution(df)
    )

    counts, percentages = (
        analyze_queue_type(df)
    )

    vectorizer, matrix = (
        build_tfidf(df)
    )

    similarity_df, centroids = (
        compute_centroids(
            df,
            matrix
        )
    )

    keywords_df = (
        extract_queue_keywords(
            df,
            vectorizer,
            matrix
        )
    )

    similar_tickets = (
        find_cross_queue_similar_tickets(
            df,
            matrix,
            top_n=100,
            threshold=0.75,
        )
    )

    target_pairs = (
        analyze_target_pairs(
            similarity_df
        )
    )

    save_reports(
        distribution,
        counts,
        percentages,
        similarity_df,
        keywords_df,
        similar_tickets,
        target_pairs,
    )

    print("\n" + "=" * 80)
    print("AUDIT FINISHED")
    print("=" * 80)


if __name__ == "__main__":
    main()

