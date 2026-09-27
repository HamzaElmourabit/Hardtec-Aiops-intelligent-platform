# ============================================================
# HARDTEC - Queue Label Quality Report
# ============================================================
# Purpose:
#   Analyze the quality and ambiguity of queue labels.
#
# This script DOES NOT:
#   - modify the dataset
#   - modify any ML model
#   - modify predictor.py
#   - modify Streamlit
#
# Outputs:
#   reports/queue_label_quality/
#
# Main analyses:
#   1. Exact duplicate texts with conflicting queues
#   2. Near-duplicate cross-queue tickets
#   3. Nearest-neighbor queue agreement
#   4. Local label ambiguity / entropy
#   5. Intra/inter queue similarity
#   6. Queue difficulty ranking
# ============================================================

from pathlib import Path
from collections import Counter
import re

import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize


# ============================================================
# CONFIGURATION
# ============================================================

ROOT_DIR = Path(__file__).resolve().parents[2]

DATA_PATH = ROOT_DIR / "data" / "processed" / "tickets_clean.csv"

REPORT_DIR = ROOT_DIR / "reports" / "queue_label_quality"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42

# TF-IDF configuration aligned with the existing Queue V4 analysis
MAX_FEATURES = 40000
NGRAM_RANGE = (1, 2)
MIN_DF = 2

# Number of nearest neighbours used for local label agreement
K_VALUES = [1, 3, 5, 10]

# Similarity threshold for suspicious cross-label pairs
NEAR_DUPLICATE_THRESHOLD = 0.80

# Maximum number of suspicious pairs to save
MAX_NEAR_DUPLICATE_PAIRS = 1000


# ============================================================
# UTILS
# ============================================================

def normalize_text(text: str) -> str:
    """
    Normalize text for exact duplicate detection.
    """
    text = str(text).lower().strip()

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text)

    # Remove surrounding punctuation
    text = text.strip(" .,!?;:")

    return text


def entropy_from_counts(counts):
    """
    Shannon entropy normalized to [0, 1].

    0 = perfectly pure neighborhood
    1 = highly mixed neighborhood
    """
    total = sum(counts.values())

    if total == 0:
        return 0.0

    probabilities = np.array(list(counts.values()), dtype=float) / total

    entropy = -np.sum(
        probabilities * np.log2(probabilities + 1e-12)
    )

    if len(probabilities) <= 1:
        return 0.0

    max_entropy = np.log2(len(probabilities))

    if max_entropy == 0:
        return 0.0

    return float(entropy / max_entropy)


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("HARDTEC - QUEUE LABEL QUALITY REPORT")
print("=" * 70)

print("\n[1/8] Loading dataset...")

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found:\n{DATA_PATH}"
    )

df = pd.read_csv(DATA_PATH)

required_columns = {"ticket_text", "queue"}

missing_columns = required_columns - set(df.columns)

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )

df = df.copy()

df["ticket_text"] = df["ticket_text"].fillna("").astype(str)
df["queue"] = df["queue"].fillna("").astype(str)

df = df[
    (df["ticket_text"].str.strip() != "")
    & (df["queue"].str.strip() != "")
].reset_index(drop=True)

print(f"Dataset rows: {len(df):,}")
print(f"Queues: {df['queue'].nunique()}")

queues = sorted(df["queue"].unique())

print("\nQueue distribution:")
print(df["queue"].value_counts())


# ============================================================
# 1. EXACT DUPLICATE TEXT ANALYSIS
# ============================================================

print("\n[2/8] Checking exact duplicate ticket texts...")

df["normalized_text"] = df["ticket_text"].apply(normalize_text)

duplicate_groups = (
    df.groupby("normalized_text")
    .agg(
        occurrences=("queue", "size"),
        unique_queues=("queue", "nunique"),
        queues=("queue", lambda x: " | ".join(sorted(set(x)))),
    )
    .reset_index()
)

duplicate_groups = duplicate_groups[
    duplicate_groups["occurrences"] > 1
].copy()

conflicting_duplicates = duplicate_groups[
    duplicate_groups["unique_queues"] > 1
].copy()

conflicting_duplicates.to_csv(
    REPORT_DIR / "exact_duplicate_conflicts.csv",
    index=False,
    encoding="utf-8-sig",
)

print(
    f"Exact duplicate groups: "
    f"{len(duplicate_groups):,}"
)

print(
    f"Conflicting duplicate groups: "
    f"{len(conflicting_duplicates):,}"
)


# ============================================================
# 2. TF-IDF REPRESENTATION
# ============================================================

print("\n[3/8] Building TF-IDF representation...")

vectorizer = TfidfVectorizer(
    lowercase=True,
    max_features=MAX_FEATURES,
    ngram_range=NGRAM_RANGE,
    min_df=MIN_DF,
    sublinear_tf=True,
)

X = vectorizer.fit_transform(df["ticket_text"])

print(f"TF-IDF shape: {X.shape}")


# ============================================================
# 3. NEAREST NEIGHBOUR LABEL AGREEMENT
# ============================================================

print("\n[4/8] Computing nearest-neighbour queue agreement...")

# Normalize for cosine similarity
X_norm = normalize(X)

# Compute similarity matrix.
# 20k x 20k is large, but manageable for this project depending
# on available RAM. We process in batches to reduce peak memory.

N = X_norm.shape[0]

batch_size = 500

neighbor_results = {
    k: [] for k in K_VALUES
}

# For each ticket we store:
# - nearest neighbour queue
# - agreement for K neighbours
# - local entropy

nearest_queue = []
nearest_similarity = []
local_entropy = []

queue_array = df["queue"].to_numpy()

for start in range(0, N, batch_size):

    end = min(start + batch_size, N)

    print(
        f"   Processing {start:,} -> {end:,} / {N:,}",
        end="\r",
    )

    batch = X_norm[start:end]

    similarities = cosine_similarity(batch, X_norm)

    # Ignore self-similarity
    row_indices = np.arange(end - start)
    global_indices = np.arange(start, end)

    similarities[row_indices, global_indices] = -1.0

    # Get top 10 neighbours
    top_k = max(K_VALUES)

    # argpartition is much faster than full sorting
    candidate_indices = np.argpartition(
        similarities,
        -top_k,
        axis=1
    )[:, -top_k:]

    for local_row, candidates in enumerate(candidate_indices):

        sims = similarities[local_row, candidates]

        # Sort candidates by similarity descending
        order = np.argsort(sims)[::-1]

        candidates = candidates[order]
        sims = sims[order]

        neighbour_queues = queue_array[candidates]

        # Nearest neighbour
        nearest_queue.append(neighbour_queues[0])
        nearest_similarity.append(float(sims[0]))

        # Local entropy based on top 10 neighbours
        counts = Counter(neighbour_queues)

        local_entropy.append(
            entropy_from_counts(counts)
        )

        for k in K_VALUES:
            selected = neighbour_queues[:k]

            agreement = np.mean(
                selected == queue_array[start + local_row]
            )

            neighbor_results[k].append(float(agreement))

print("\nNearest-neighbour analysis completed.")


# ============================================================
# 4. SAVE NEIGHBOUR AGREEMENT
# ============================================================

print("\n[5/8] Saving nearest-neighbour agreement...")

nn_df = pd.DataFrame({
    "ticket_index": np.arange(N),
    "queue": df["queue"],
    "nearest_queue": nearest_queue,
    "nearest_similarity": nearest_similarity,
    "local_entropy": local_entropy,
})

for k in K_VALUES:
    nn_df[f"agreement_k{k}"] = neighbor_results[k]

nn_df.to_csv(
    REPORT_DIR / "queue_nn_agreement.csv",
    index=False,
    encoding="utf-8-sig",
)


# ============================================================
# 5. PER-QUEUE AMBIGUITY
# ============================================================

print("\n[6/8] Computing per-queue ambiguity...")

ambiguity_rows = []

for queue in queues:

    subset = nn_df[nn_df["queue"] == queue]

    row = {
        "queue": queue,
        "tickets": len(subset),
        "nearest_same_queue_rate": (
            subset["nearest_queue"] == queue
        ).mean(),
        "mean_nearest_similarity": (
            subset["nearest_similarity"].mean()
        ),
        "mean_local_entropy": (
            subset["local_entropy"].mean()
        ),
    }

    for k in K_VALUES:
        row[f"mean_agreement_k{k}"] = (
            subset[f"agreement_k{k}"].mean()
        )

    ambiguity_rows.append(row)

ambiguity_df = pd.DataFrame(ambiguity_rows)

# Difficulty:
# low nearest-neighbour agreement + high entropy = more ambiguous

ambiguity_df["ambiguity_score"] = (
    (
        1
        - ambiguity_df["nearest_same_queue_rate"]
    )
    * 0.4
    +
    ambiguity_df["mean_local_entropy"] * 0.6
)

ambiguity_df = ambiguity_df.sort_values(
    "ambiguity_score",
    ascending=False,
)

ambiguity_df.to_csv(
    REPORT_DIR / "queue_ambiguity_scores.csv",
    index=False,
    encoding="utf-8-sig",
)


# ============================================================
# 6. CROSS-QUEUE NEAR DUPLICATES
# ============================================================

print("\n[7/8] Searching for highly similar tickets with different queues...")

cross_queue_pairs = []

# Instead of storing the complete 20k x 20k matrix,
# process batches and only retain pairs above threshold.

for start in range(0, N, batch_size):

    end = min(start + batch_size, N)

    print(
        f"   Searching {start:,} -> {end:,} / {N:,}",
        end="\r",
    )

    batch = X_norm[start:end]

    similarities = cosine_similarity(batch, X_norm)

    for local_row in range(end - start):

        global_index = start + local_row

        # Exclude itself
        similarities[local_row, global_index] = -1.0

        # Only inspect sufficiently similar candidates
        candidate_indices = np.where(
            similarities[local_row] >= NEAR_DUPLICATE_THRESHOLD
        )[0]

        for candidate in candidate_indices:

            # Keep each pair only once
            if candidate <= global_index:
                continue

            similarity = float(
                similarities[local_row, candidate]
            )

            queue_a = queue_array[global_index]
            queue_b = queue_array[candidate]

            # We only care about different labels
            if queue_a == queue_b:
                continue

            cross_queue_pairs.append({
                "ticket_index_a": global_index,
                "ticket_index_b": int(candidate),
                "queue_a": queue_a,
                "queue_b": queue_b,
                "similarity": similarity,
                "ticket_a": df.iloc[
                    global_index
                ]["ticket_text"],
                "ticket_b": df.iloc[
                    candidate
                ]["ticket_text"],
            })

# Sort highest similarity first
cross_queue_pairs = sorted(
    cross_queue_pairs,
    key=lambda x: x["similarity"],
    reverse=True,
)

cross_queue_pairs = cross_queue_pairs[
    :MAX_NEAR_DUPLICATE_PAIRS
]

cross_queue_df = pd.DataFrame(
    cross_queue_pairs
)

cross_queue_df.to_csv(
    REPORT_DIR / "cross_queue_near_duplicates.csv",
    index=False,
    encoding="utf-8-sig",
)

print(
    f"\nCross-queue near-duplicate pairs: "
    f"{len(cross_queue_df):,}"
)


# ============================================================
# 7. QUEUE-TO-QUEUE SIMILARITY
# ============================================================

print("\n[8/8] Computing queue-level similarity...")

queue_centroids = {}

for queue in queues:

    indices = np.where(queue_array == queue)[0]

    centroid = np.asarray(
        X[indices].mean(axis=0)
    ).ravel()

    queue_centroids[queue] = centroid


centroid_matrix = np.vstack(
    [queue_centroids[q] for q in queues]
)

centroid_similarity = cosine_similarity(
    centroid_matrix
)

similarity_rows = []

for i, queue_a in enumerate(queues):

    for j, queue_b in enumerate(queues):

        if j <= i:
            continue

        similarity_rows.append({
            "queue_a": queue_a,
            "queue_b": queue_b,
            "centroid_similarity": float(
                centroid_similarity[i, j]
            ),
        })

queue_similarity_df = pd.DataFrame(
    similarity_rows
).sort_values(
    "centroid_similarity",
    ascending=False,
)

queue_similarity_df.to_csv(
    REPORT_DIR / "queue_pair_similarity.csv",
    index=False,
    encoding="utf-8-sig",
)


# ============================================================
# 8. GLOBAL SUMMARY
# ============================================================

global_summary = {
    "dataset_rows": len(df),
    "number_of_queues": len(queues),
    "exact_duplicate_groups": len(duplicate_groups),
    "conflicting_exact_duplicate_groups": len(
        conflicting_duplicates
    ),
    "cross_queue_near_duplicate_pairs": len(
        cross_queue_df
    ),
    "mean_nearest_same_queue_rate": nn_df.apply(
        lambda row: row["queue"] == row["nearest_queue"],
        axis=1,
    ).mean(),
    "mean_agreement_k1": nn_df["agreement_k1"].mean(),
    "mean_agreement_k3": nn_df["agreement_k3"].mean(),
    "mean_agreement_k5": nn_df["agreement_k5"].mean(),
    "mean_agreement_k10": nn_df["agreement_k10"].mean(),
    "mean_local_entropy": nn_df["local_entropy"].mean(),
    "max_queue_centroid_similarity": (
        queue_similarity_df["centroid_similarity"].max()
        if len(queue_similarity_df) > 0
        else np.nan
    ),
}

summary_df = pd.DataFrame(
    [global_summary]
)

summary_df.to_csv(
    REPORT_DIR / "queue_label_quality_summary.csv",
    index=False,
    encoding="utf-8-sig",
)


# ============================================================
# HUMAN-READABLE REPORT
# ============================================================

report_path = REPORT_DIR / "queue_label_quality_report.txt"

with open(
    report_path,
    "w",
    encoding="utf-8",
) as f:

    f.write("=" * 70 + "\n")
    f.write("HARDTEC - QUEUE LABEL QUALITY REPORT\n")
    f.write("=" * 70 + "\n\n")

    f.write("DATASET\n")
    f.write("-" * 70 + "\n")
    f.write(f"Rows: {len(df):,}\n")
    f.write(f"Queues: {len(queues)}\n\n")

    f.write("EXACT DUPLICATES\n")
    f.write("-" * 70 + "\n")
    f.write(
        f"Duplicate groups: "
        f"{len(duplicate_groups):,}\n"
    )
    f.write(
        f"Conflicting queue groups: "
        f"{len(conflicting_duplicates):,}\n\n"
    )

    f.write("NEAREST-NEIGHBOUR AGREEMENT\n")
    f.write("-" * 70 + "\n")

    for k in K_VALUES:
        f.write(
            f"K={k}: "
            f"{nn_df[f'agreement_k{k}'].mean():.4f}\n"
        )

    f.write(
        "\nMean local entropy: "
        f"{nn_df['local_entropy'].mean():.4f}\n"
    )

    f.write("\nCROSS-QUEUE NEAR DUPLICATES\n")
    f.write("-" * 70 + "\n")
    f.write(
        f"Pairs >= {NEAR_DUPLICATE_THRESHOLD}: "
        f"{len(cross_queue_df):,}\n\n"
    )

    if len(cross_queue_df) > 0:

        f.write("Top examples:\n\n")

        for _, row in cross_queue_df.head(20).iterrows():

            f.write(
                f"Similarity: {row['similarity']:.4f}\n"
            )

            f.write(
                f"Queue A: {row['queue_a']}\n"
            )

            f.write(
                f"Ticket A: {row['ticket_a']}\n"
            )

            f.write(
                f"Queue B: {row['queue_b']}\n"
            )

            f.write(
                f"Ticket B: {row['ticket_b']}\n"
            )

            f.write("-" * 60 + "\n")

    f.write("\nMOST AMBIGUOUS QUEUES\n")
    f.write("-" * 70 + "\n")

    for _, row in ambiguity_df.head(10).iterrows():

        f.write(
            f"{row['queue']}\n"
            f"  Tickets: {int(row['tickets'])}\n"
            f"  Same-queue NN rate: "
            f"{row['nearest_same_queue_rate']:.4f}\n"
            f"  Agreement K=5: "
            f"{row['mean_agreement_k5']:.4f}\n"
            f"  Local entropy: "
            f"{row['mean_local_entropy']:.4f}\n"
            f"  Ambiguity score: "
            f"{row['ambiguity_score']:.4f}\n\n"
        )

    f.write("MOST SIMILAR QUEUE PAIRS\n")
    f.write("-" * 70 + "\n")

    for _, row in queue_similarity_df.head(15).iterrows():

        f.write(
            f"{row['queue_a']} <-> "
            f"{row['queue_b']} : "
            f"{row['centroid_similarity']:.4f}\n"
        )

    f.write("\nINTERPRETATION\n")
    f.write("-" * 70 + "\n")
    f.write(
        "This report identifies potential queue-label ambiguity.\n"
    )
    f.write(
        "High semantic similarity between tickets with different "
        "queue labels is NOT, by itself, proof that a label is wrong.\n"
    )
    f.write(
        "The results should be interpreted together with the "
        "classifier evaluation and manual inspection.\n"
    )


# ============================================================
# FINAL CONSOLE SUMMARY
# ============================================================

print("\n")
print("=" * 70)
print("QUEUE LABEL QUALITY ANALYSIS COMPLETED")
print("=" * 70)

print(
    f"\nDataset: {len(df):,} tickets"
)

print(
    f"Queues: {len(queues)}"
)

print(
    f"\nExact duplicate groups: "
    f"{len(duplicate_groups):,}"
)

print(
    f"Conflicting exact duplicates: "
    f"{len(conflicting_duplicates):,}"
)

print(
    f"\nCross-queue near duplicates "
    f"(similarity >= {NEAR_DUPLICATE_THRESHOLD}): "
    f"{len(cross_queue_df):,}"
)

print(
    "\nNearest-neighbour agreement:"
)

for k in K_VALUES:
    print(
        f"  K={k}: "
        f"{nn_df[f'agreement_k{k}'].mean():.4f}"
    )

print(
    f"\nMean local entropy: "
    f"{nn_df['local_entropy'].mean():.4f}"
)

print("\nMost ambiguous queues:")

for _, row in ambiguity_df.head(5).iterrows():

    print(
        f"  - {row['queue']}: "
        f"ambiguity={row['ambiguity_score']:.4f}, "
        f"same-NN={row['nearest_same_queue_rate']:.4f}"
    )

print("\nMost similar queue pairs:")

for _, row in queue_similarity_df.head(5).iterrows():

    print(
        f"  - {row['queue_a']} <-> "
        f"{row['queue_b']}: "
        f"{row['centroid_similarity']:.4f}"
    )

print("\nReports saved to:")

print(REPORT_DIR)

print("\nGenerated files:")

for path in sorted(REPORT_DIR.glob("*")):
    print(f"  - {path.name}")

print("\n" + "=" * 70)