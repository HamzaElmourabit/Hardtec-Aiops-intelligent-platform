import os
import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

INPUT_PATH = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "aiops_windows.csv",
)

OUTPUT_PATH = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "aiops_anomalies.csv",
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "aiops_isolation_forest.pkl",
)

SCALER_PATH = os.path.join(
    BASE_DIR,
    "models",
    "aiops_scaler.pkl",
)


# ============================================================
# FEATURES
# IMPORTANT:
# failure_count / failure_rate are intentionally excluded
# from model training to avoid direct outcome leakage.
# ============================================================

FEATURES = [
    "event_count",
    "raw_row_count",
    "avg_duration_ms",
    "max_duration_ms",
    "p95_duration_ms",
    "high_severity_count",
    "high_severity_rate",
    "error_density",
    "unique_paths",
    "unique_event_types",
    "unique_categories",
    "unique_severities",
]


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("HARDTEC - AIOps Anomaly Detection")
    print("=" * 70)

    # --------------------------------------------------------
    # 1. LOAD DATA
    # --------------------------------------------------------

    print("\n[1/7] Loading AIOps windows...")

    if not os.path.exists(INPUT_PATH):
        raise FileNotFoundError(
            f"AIOps windows file not found:\n{INPUT_PATH}"
        )

    df = pd.read_csv(INPUT_PATH)

    print(f"Input: {INPUT_PATH}")
    print(f"Rows : {len(df)}")
    print(f"Cols : {len(df.columns)}")

    # --------------------------------------------------------
    # 2. VALIDATION
    # --------------------------------------------------------

    print("\n[2/7] Validating features...")

    missing_features = [
        feature for feature in FEATURES
        if feature not in df.columns
    ]

    if missing_features:
        raise ValueError(
            f"Missing features: {missing_features}"
        )

    df["window_start"] = pd.to_datetime(
        df["window_start"],
        utc=True,
        errors="coerce",
    )

    if df["window_start"].isna().any():
        raise ValueError(
            "Some window_start values could not be converted."
        )

    print("Required features : OK")
    print("Timestamp          : OK")

    # --------------------------------------------------------
    # 3. PREPARE FEATURES
    # --------------------------------------------------------

    print("\n[3/7] Preparing anomaly detection features...")

    X = df[FEATURES].copy()

    X = X.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    X = X.fillna(0)

    X = X.astype(float)

    print(f"Feature count : {len(FEATURES)}")

    for feature in FEATURES:
        print(f"  - {feature}")

    # --------------------------------------------------------
    # 4. NORMALIZATION
    # --------------------------------------------------------

    print("\n[4/7] Normalizing features...")

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    print("StandardScaler : OK")

    # --------------------------------------------------------
    # 5. ISOLATION FOREST
    # --------------------------------------------------------

    print("\n[5/7] Training Isolation Forest...")

    model = IsolationForest(
        n_estimators=200,
        contamination=0.15,
        random_state=42,
        n_jobs=-1,
    )

    model.fit(X_scaled)

    print("Isolation Forest : OK")
    print("Trees            : 200")
    print("Contamination    : 15%")

    # --------------------------------------------------------
    # 6. ANOMALY SCORING
    # --------------------------------------------------------

    print("\n[6/7] Computing anomaly scores...")

    # sklearn:
    # decision_function > 0 generally means more normal
    # decision_function < 0 generally means more anomalous
    #
    # We invert the value so:
    # HIGHER anomaly_score = MORE suspicious

    decision_scores = model.decision_function(X_scaled)

    anomaly_score = -decision_scores

    predictions = model.predict(X_scaled)

    is_anomaly = predictions == -1

    result = df.copy()

    result["anomaly_score"] = anomaly_score
    result["is_anomaly"] = is_anomaly

    # Higher score = more suspicious
    result["anomaly_rank"] = (
        result["anomaly_score"]
        .rank(
            ascending=False,
            method="min",
        )
        .astype(int)
    )

    # --------------------------------------------------------
    # Anomaly severity
    # --------------------------------------------------------

    score_min = result["anomaly_score"].min()
    score_max = result["anomaly_score"].max()

    if score_max > score_min:

        result["anomaly_score_normalized"] = (
            (result["anomaly_score"] - score_min)
            / (score_max - score_min)
        )

    else:

        result["anomaly_score_normalized"] = 0.0

    def classify_anomaly(score):

        normalized = (
            (score - score_min)
            / (score_max - score_min)
            if score_max > score_min
            else 0
        )

        if normalized >= 0.80:
            return "CRITICAL"

        elif normalized >= 0.60:
            return "HIGH"

        elif normalized >= 0.40:
            return "MEDIUM"

        else:
            return "LOW"

    result["anomaly_level"] = (
        result["anomaly_score"]
        .apply(classify_anomaly)
    )

    # --------------------------------------------------------
    # Operational interpretation
    # --------------------------------------------------------

    def generate_reason(row):

        reasons = []

        if row["event_count"] > df["event_count"].quantile(0.90):
            reasons.append("unusual event volume")

        if row["avg_duration_ms"] > df["avg_duration_ms"].quantile(0.90):
            reasons.append("high average latency")

        if row["p95_duration_ms"] > df["p95_duration_ms"].quantile(0.90):
            reasons.append("high P95 latency")

        if row["high_severity_rate"] > df["high_severity_rate"].quantile(0.90):
            reasons.append("high severity concentration")

        if row["error_density"] > df["error_density"].quantile(0.90):
            reasons.append("high error density")

        if row["unique_event_types"] > df["unique_event_types"].quantile(0.90):
            reasons.append("unusual event diversity")

        if not reasons:
            reasons.append("multivariate behavioral anomaly")

        return "; ".join(reasons)

    result["anomaly_reason"] = result.apply(
        generate_reason,
        axis=1,
    )

    # --------------------------------------------------------
    # Sort by anomaly rank
    # --------------------------------------------------------

    result = result.sort_values(
        "anomaly_score",
        ascending=False,
    ).reset_index(drop=True)

    # --------------------------------------------------------
    # 7. SAVE
    # --------------------------------------------------------

    print("\n[7/7] Saving anomaly detection results...")

    os.makedirs(
        os.path.dirname(OUTPUT_PATH),
        exist_ok=True,
    )

    os.makedirs(
        os.path.dirname(MODEL_PATH),
        exist_ok=True,
    )

    result.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    joblib.dump(
        model,
        MODEL_PATH,
    )

    joblib.dump(
        scaler,
        SCALER_PATH,
    )

    # --------------------------------------------------------
    # REPORT
    # --------------------------------------------------------

    anomaly_count = int(
        result["is_anomaly"].sum()
    )

    normal_count = len(result) - anomaly_count

    print("\n" + "=" * 70)
    print("AIOps ANOMALY DETECTION REPORT")
    print("=" * 70)

    print(f"Total windows       : {len(result)}")
    print(f"Anomalous windows   : {anomaly_count}")
    print(f"Normal windows      : {normal_count}")
    print(
        f"Anomaly rate        : "
        f"{anomaly_count / len(result):.2%}"
    )

    print("\nTop 10 suspicious windows")
    print("-" * 70)

    display_columns = [
        "window_start",
        "anomaly_score",
        "anomaly_level",
        "event_count",
        "avg_duration_ms",
        "p95_duration_ms",
        "high_severity_rate",
        "error_density",
        "failure_rate",
        "anomaly_reason",
    ]

    print(
        result[display_columns]
        .head(10)
        .to_string(index=False)
    )

    print("\nOutput:")
    print(OUTPUT_PATH)

    print("\nModel:")
    print(MODEL_PATH)

    print("\nScaler:")
    print(SCALER_PATH)

    print("\n" + "=" * 70)
    print("AIOps anomaly detection completed successfully.")
    print("=" * 70)


if __name__ == "__main__":
    main()