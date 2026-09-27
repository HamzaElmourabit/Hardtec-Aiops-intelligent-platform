"""
GAIA - Temporal AIOps Anomaly Detection
----------------------------------------

Combine:
- temporal error signals
- traffic/volume signals
- semantic anomaly signals
- temporal variations

Then applies Isolation Forest to detect anomalous
operational windows.

Input:
    data/processed/gaia_temporal_aiops_features.csv

Outputs:
    data/processed/gaia_temporal_anomalies.csv
    models/gaia_temporal_isolation_forest.pkl
    models/gaia_temporal_scaler.pkl
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = Path("data/processed/gaia_temporal_aiops_features.csv")

OUTPUT_FILE = Path("data/processed/gaia_temporal_anomalies.csv")

MODEL_FILE = Path("models/gaia_temporal_isolation_forest.pkl")

SCALER_FILE = Path("models/gaia_temporal_scaler.pkl")

CONTAMINATION = 0.10

RANDOM_STATE = 42


# ============================================================
# FEATURES
# ============================================================

FEATURES = [
    # Temporal / operational signals
    "total_logs",
    "error_count",
    "error_rate",
    "unique_sources",
    "unique_hosts",

    # Temporal variations
    "previous_error_rate",
    "error_rate_delta",
    "previous_total_logs",
    "volume_delta",
    "rolling_error_rate_3",
    "rolling_volume_3",

    # Semantic anomaly signals
    "semantic_total_logs",
    "semantic_anomaly_count",
    "semantic_anomaly_rate",
    "semantic_mean_confidence",
]


# ============================================================
# UTILITY
# ============================================================

def print_separator():
    print("=" * 70)


# ============================================================
# MAIN
# ============================================================

def main():

    print_separator()
    print("GAIA - TEMPORAL AIOPS ANOMALY DETECTION")
    print_separator()

    # --------------------------------------------------------
    # 1. LOAD DATA
    # --------------------------------------------------------

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Fichier introuvable : {INPUT_FILE}"
        )

    print("\n[1/7] Chargement des données...")

    df = pd.read_csv(INPUT_FILE)

    print(f"Fenêtres chargées : {len(df):,}")
    print(f"Colonnes           : {len(df.columns)}")

    # --------------------------------------------------------
    # 2. CHECK FEATURES
    # --------------------------------------------------------

    print("\n[2/7] Vérification des features...")

    missing_features = [
        feature for feature in FEATURES
        if feature not in df.columns
    ]

    if missing_features:
        raise ValueError(
            "Features manquantes : "
            + ", ".join(missing_features)
        )

    print("Toutes les features nécessaires sont présentes.")

    # --------------------------------------------------------
    # 3. PREPARE DATA
    # --------------------------------------------------------

    print("\n[3/7] Préparation des données...")

    df["window_start"] = pd.to_datetime(
        df["window_start"],
        utc=True,
        errors="coerce"
    )

    df = df.sort_values("window_start").reset_index(drop=True)

    X = df[FEATURES].copy()

    # Replace infinite values
    X = X.replace([np.inf, -np.inf], np.nan)

    # Missing values are possible for first temporal windows
    X = X.fillna(0)

    print(f"Samples : {len(X):,}")
    print(f"Features: {len(FEATURES)}")

    # --------------------------------------------------------
    # 4. STANDARDIZATION
    # --------------------------------------------------------

    print("\n[4/7] Standardisation...")

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    print("Standardisation terminée.")

    # --------------------------------------------------------
    # 5. ISOLATION FOREST
    # --------------------------------------------------------

    print("\n[5/7] Détection des anomalies...")

    model = IsolationForest(
        n_estimators=300,
        contamination=CONTAMINATION,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    model.fit(X_scaled)

    # Isolation Forest:
    # -1 = anomaly
    # +1 = normal

    predictions = model.predict(X_scaled)

    # decision_function:
    # higher = more normal
    #
    # We invert it so:
    # higher = more anomalous

    raw_scores = -model.decision_function(X_scaled)

    # Normalize 0 -> 1
    min_score = raw_scores.min()
    max_score = raw_scores.max()

    if max_score > min_score:
        anomaly_scores = (
            (raw_scores - min_score)
            / (max_score - min_score)
        )
    else:
        anomaly_scores = np.zeros(len(raw_scores))

    df["aiops_anomaly_score"] = anomaly_scores

    df["is_aiops_anomaly"] = (
        predictions == -1
    ).astype(int)

    # --------------------------------------------------------
    # 6. OPERATIONAL SEVERITY
    # --------------------------------------------------------

    print("\n[6/7] Calcul de la criticité opérationnelle...")

    def classify_severity(score):

        if score >= 0.80:
            return "CRITICAL"

        if score >= 0.60:
            return "HIGH"

        if score >= 0.40:
            return "MEDIUM"

        return "LOW"

    df["aiops_severity"] = (
        df["aiops_anomaly_score"]
        .apply(classify_severity)
    )

    # --------------------------------------------------------
    # EXPLANATION / REASONS
    # --------------------------------------------------------

    def build_reason(row):

        reasons = []

        # High semantic anomaly density
        if row["semantic_anomaly_rate"] >= 0.75:
            reasons.append(
                "HIGH_SEMANTIC_ANOMALY_DENSITY"
            )

        elif row["semantic_anomaly_rate"] >= 0.50:
            reasons.append(
                "ELEVATED_SEMANTIC_ANOMALY_DENSITY"
            )

        # Error rate
        if row["error_rate"] >= 0.75:
            reasons.append(
                "VERY_HIGH_ERROR_RATE"
            )

        elif row["error_rate"] >= 0.50:
            reasons.append(
                "HIGH_ERROR_RATE"
            )

        # Error rate increase
        if row["error_rate_delta"] >= 0.30:
            reasons.append(
                "RAPID_ERROR_RATE_INCREASE"
            )

        # Volume increase
        if row["volume_delta"] >= 0.50:
            reasons.append(
                "RAPID_TRAFFIC_INCREASE"
            )

        # Semantic anomaly count
        if row["semantic_anomaly_count"] >= 20:
            reasons.append(
                "HIGH_ANOMALOUS_EVENT_VOLUME"
            )

        # Multiple services
        if row["unique_sources"] >= 4:
            reasons.append(
                "MULTI_SERVICE_ACTIVITY"
            )

        if not reasons:
            reasons.append(
                "MULTIVARIATE_OPERATIONAL_ANOMALY"
            )

        return " | ".join(reasons)

    df["aiops_anomaly_reason"] = df.apply(
        build_reason,
        axis=1
    )

    # --------------------------------------------------------
    # ANOMALY RANK
    # --------------------------------------------------------

    df["anomaly_rank"] = (
        df["aiops_anomaly_score"]
        .rank(
            ascending=False,
            method="min"
        )
        .astype(int)
    )

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    MODEL_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    joblib.dump(
        model,
        MODEL_FILE
    )

    joblib.dump(
        scaler,
        SCALER_FILE
    )

    # --------------------------------------------------------
    # 7. RESULTS
    # --------------------------------------------------------

    print("\n[7/7] Résultats")
    print()

    total_windows = len(df)

    anomaly_count = int(
        df["is_aiops_anomaly"].sum()
    )

    anomaly_rate = (
        anomaly_count / total_windows
        if total_windows > 0
        else 0
    )

    print(f"Fenêtres analysées : {total_windows:,}")
    print(f"Anomalies AIOps    : {anomaly_count:,}")
    print(
        f"Taux d'anomalies   : "
        f"{anomaly_rate:.2%}"
    )

    print("\nDistribution de criticité :")

    severity_counts = (
        df["aiops_severity"]
        .value_counts()
    )

    for severity in [
        "CRITICAL",
        "HIGH",
        "MEDIUM",
        "LOW",
    ]:

        count = int(
            severity_counts.get(severity, 0)
        )

        print(
            f"  {severity:<10} : {count}"
        )

    # --------------------------------------------------------
    # TOP ANOMALIES
    # --------------------------------------------------------

    print("\nTop 15 anomalies AIOps :")
    print()

    top = (
        df[
            df["is_aiops_anomaly"] == 1
        ]
        .sort_values(
            "aiops_anomaly_score",
            ascending=False
        )
        .head(15)
    )

    if len(top) == 0:

        print("Aucune anomalie détectée.")

    else:

        display_columns = [
            "window_start",
            "total_logs",
            "error_rate",
            "semantic_anomaly_rate",
            "semantic_anomaly_count",
            "aiops_anomaly_score",
            "aiops_severity",
            "aiops_anomaly_reason",
        ]

        print(
            top[display_columns]
            .to_string(index=False)
        )

    # --------------------------------------------------------
    # FILES
    # --------------------------------------------------------

    print("\n")
    print_separator()
    print("FICHIERS CREES")
    print_separator()

    print(f"\n{OUTPUT_FILE}")
    print(MODEL_FILE)
    print(SCALER_FILE)

    print("\nTerminé.")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()