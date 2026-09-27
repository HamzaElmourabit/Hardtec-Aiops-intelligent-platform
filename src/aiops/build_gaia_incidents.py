import os
import pandas as pd
import numpy as np


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = "./data/processed/gaia_temporal_anomalies.csv"
OUTPUT_FILE = "./data/processed/gaia_incidents.csv"

# Une anomalie peut être séparée de la suivante par
# maximum 5 minutes et appartenir au même épisode.
MAX_GAP_MINUTES = 5

# Minimum de fenêtres anormales pour considérer un épisode
# comme significatif.
MIN_ANOMALOUS_WINDOWS = 1


# ============================================================
# UTILITAIRES
# ============================================================

def safe_float(value, default=0.0):
    try:
        if pd.isna(value):
            return default
        return float(value)
    except (ValueError, TypeError):
        return default


def classify_incident_type(row):
    """
    Détermine le type dominant de l'incident à partir
    des signaux disponibles.
    """

    semantic_rate = safe_float(row.get("semantic_anomaly_rate"))
    error_rate = safe_float(row.get("error_rate"))
    volume_delta = safe_float(row.get("volume_delta"))
    error_delta = safe_float(row.get("error_rate_delta"))
    unique_sources = safe_float(row.get("unique_sources"))
    anomaly_score = safe_float(row.get("aiops_anomaly_score"))

    signals = []

    if semantic_rate >= 0.50:
        signals.append("SEMANTIC_ANOMALY")

    if error_rate >= 0.50:
        signals.append("HIGH_ERROR_RATE")

    if error_delta >= 0.20:
        signals.append("RAPID_ERROR_INCREASE")

    if volume_delta >= 0.50:
        signals.append("TRAFFIC_SPIKE")

    if unique_sources >= 3:
        signals.append("MULTI_SERVICE_ANOMALY")

    if anomaly_score >= 0.80:
        signals.append("HIGH_MULTIVARIATE_ANOMALY")

    if not signals:
        return "MULTIVARIATE_OPERATIONAL_ANOMALY"

    # Priorité sémantique
    if "SEMANTIC_ANOMALY" in signals and "HIGH_ERROR_RATE" in signals:
        return "SEMANTIC_AND_ERROR_ANOMALY"

    if "HIGH_ERROR_RATE" in signals:
        return "ERROR_RATE_ANOMALY"

    if "TRAFFIC_SPIKE" in signals:
        return "TRAFFIC_ANOMALY"

    if "SEMANTIC_ANOMALY" in signals:
        return "SEMANTIC_ANOMALY"

    return signals[0]


def calculate_incident_severity(group):
    """
    Calcule une criticité agrégée pour tout l'incident.
    """

    max_score = group["aiops_anomaly_score"].max()

    max_semantic_rate = group["semantic_anomaly_rate"].max()
    max_error_rate = group["error_rate"].max()
    max_volume_delta = group["volume_delta"].max()

    if (
        max_score >= 0.90
        or max_semantic_rate >= 0.90
        or max_error_rate >= 0.90
    ):
        return "CRITICAL"

    if (
        max_score >= 0.75
        or max_semantic_rate >= 0.75
        or max_error_rate >= 0.75
        or max_volume_delta >= 1.00
    ):
        return "HIGH"

    if (
        max_score >= 0.50
        or max_semantic_rate >= 0.50
        or max_error_rate >= 0.50
    ):
        return "MEDIUM"

    return "LOW"


def build_incident_reason(group):
    """
    Produit une explication lisible de l'incident.
    """

    reasons = []

    max_semantic = group["semantic_anomaly_rate"].max()
    max_error = group["error_rate"].max()
    max_error_delta = group["error_rate_delta"].max()
    max_volume_delta = group["volume_delta"].max()
    max_sources = group["unique_sources"].max()
    max_score = group["aiops_anomaly_score"].max()

    if max_semantic >= 0.50:
        reasons.append("HIGH_SEMANTIC_ANOMALY_DENSITY")

    if max_error >= 0.50:
        reasons.append("HIGH_ERROR_RATE")

    if max_error >= 0.80:
        reasons.append("VERY_HIGH_ERROR_RATE")

    if max_error_delta >= 0.20:
        reasons.append("RAPID_ERROR_RATE_INCREASE")

    if max_volume_delta >= 0.50:
        reasons.append("TRAFFIC_INCREASE")

    if max_sources >= 3:
        reasons.append("MULTI_SERVICE_ACTIVITY")

    if max_score >= 0.80:
        reasons.append("HIGH_MULTIVARIATE_ANOMALY")

    if not reasons:
        reasons.append("MULTIVARIATE_OPERATIONAL_ANOMALY")

    return " | ".join(dict.fromkeys(reasons))


# ============================================================
# CHARGEMENT
# ============================================================

print("=" * 70)
print("GAIA - INCIDENT CORRELATION")
print("=" * 70)

if not os.path.exists(INPUT_FILE):
    raise FileNotFoundError(
        f"Fichier introuvable : {INPUT_FILE}"
    )

df = pd.read_csv(INPUT_FILE)

print(f"\nFenêtres chargées : {len(df):,}")


# ============================================================
# PREPARATION
# ============================================================

required_columns = [
    "window_start",
    "window_end",
    "is_aiops_anomaly",
    "aiops_anomaly_score",
    "semantic_anomaly_rate",
    "error_rate",
    "error_rate_delta",
    "volume_delta",
    "unique_sources",
]

missing = [
    col for col in required_columns
    if col not in df.columns
]

if missing:
    raise ValueError(
        f"Colonnes manquantes : {missing}"
    )

df["window_start"] = pd.to_datetime(
    df["window_start"],
    utc=True,
    errors="coerce"
)

df["window_end"] = pd.to_datetime(
    df["window_end"],
    utc=True,
    errors="coerce"
)

df = df.dropna(subset=["window_start"]).copy()

df["is_aiops_anomaly"] = (
    df["is_aiops_anomaly"]
    .astype(str)
    .str.lower()
    .isin(["true", "1", "yes"])
)

df = df.sort_values("window_start").reset_index(drop=True)


# ============================================================
# SELECTION DES FENETRES ANORMALES
# ============================================================

anomalies = df[
    df["is_aiops_anomaly"]
].copy()

print(f"Fenêtres normales   : {(~df['is_aiops_anomaly']).sum():,}")
print(f"Fenêtres anormales  : {len(anomalies):,}")

if anomalies.empty:
    print("\nAucune anomalie détectée.")
    print("Aucun incident ne peut être construit.")
    raise SystemExit(0)


# ============================================================
# CORRELATION TEMPORELLE
# ============================================================

anomalies = anomalies.sort_values(
    "window_start"
).reset_index(drop=True)

time_diff = (
    anomalies["window_start"]
    .diff()
    .dt.total_seconds()
    .div(60)
)

# Nouvel épisode lorsque l'écart dépasse MAX_GAP_MINUTES
anomalies["new_incident"] = (
    time_diff.isna()
    | (time_diff > MAX_GAP_MINUTES)
)

anomalies["incident_group"] = (
    anomalies["new_incident"]
    .cumsum()
)

print(
    f"\nGroupes temporels détectés : "
    f"{anomalies['incident_group'].nunique()}"
)


# ============================================================
# CONSTRUCTION DES INCIDENTS
# ============================================================

incident_rows = []

incident_id = 0

for group_id, group in anomalies.groupby(
    "incident_group"
):

    if len(group) < MIN_ANOMALOUS_WINDOWS:
        continue

    incident_id += 1

    group = group.sort_values(
        "window_start"
    ).copy()

    incident_start = group["window_start"].min()
    incident_end = group["window_end"].max()

    duration_minutes = (
        incident_end - incident_start
    ).total_seconds() / 60

    max_score = group[
        "aiops_anomaly_score"
    ].max()

    mean_score = group[
        "aiops_anomaly_score"
    ].mean()

    max_error_rate = group[
        "error_rate"
    ].max()

    mean_error_rate = group[
        "error_rate"
    ].mean()

    max_semantic_rate = group[
        "semantic_anomaly_rate"
    ].max()

    mean_semantic_rate = group[
        "semantic_anomaly_rate"
    ].mean()

    max_total_logs = group[
        "total_logs"
    ].max()

    total_anomalous_logs = group[
        "semantic_anomaly_count"
    ].sum()

    max_sources = group[
        "unique_sources"
    ].max()

    max_hosts = group[
        "unique_hosts"
    ].max()

    severity = calculate_incident_severity(
        group
    )

    incident_type = classify_incident_type(
        group.loc[
            group["aiops_anomaly_score"].idxmax()
        ]
    )

    reason = build_incident_reason(
        group
    )

    incident_rows.append({
        "incident_id": f"INC-{incident_id:04d}",
        "incident_start": incident_start,
        "incident_end": incident_end,
        "duration_minutes": round(
            duration_minutes, 2
        ),
        "anomalous_windows": len(group),
        "max_aiops_anomaly_score": round(
            max_score, 4
        ),
        "mean_aiops_anomaly_score": round(
            mean_score, 4
        ),
        "max_error_rate": round(
            max_error_rate, 4
        ),
        "mean_error_rate": round(
            mean_error_rate, 4
        ),
        "max_semantic_anomaly_rate": round(
            max_semantic_rate, 4
        ),
        "mean_semantic_anomaly_rate": round(
            mean_semantic_rate, 4
        ),
        "max_total_logs": int(
            max_total_logs
        ),
        "total_semantic_anomalous_logs": int(
            total_anomalous_logs
        ),
        "max_unique_sources": int(
            max_sources
        ),
        "max_unique_hosts": int(
            max_hosts
        ),
        "incident_severity": severity,
        "incident_type": incident_type,
        "incident_reason": reason,
    })


# ============================================================
# DATAFRAME FINAL
# ============================================================

incidents = pd.DataFrame(
    incident_rows
)

if incidents.empty:
    raise RuntimeError(
        "Aucun incident n'a pu être construit."
    )

incidents = incidents.sort_values(
    "incident_start"
).reset_index(drop=True)


# ============================================================
# STATISTIQUES
# ============================================================

print("\n" + "=" * 70)
print("RESULTATS")
print("=" * 70)

print(
    f"\nIncidents corrélés : "
    f"{len(incidents):,}"
)

print(
    f"Fenêtres anormales utilisées : "
    f"{len(anomalies):,}"
)

print(
    f"Ratio anomalies / incidents : "
    f"{len(anomalies) / len(incidents):.2f}"
)

print("\nDistribution de criticité :")

print(
    incidents["incident_severity"]
    .value_counts()
    .sort_index()
    .to_string()
)

print("\nDistribution des types :")

print(
    incidents["incident_type"]
    .value_counts()
    .to_string()
)


# ============================================================
# APERCU
# ============================================================

print("\n" + "=" * 70)
print("TOP INCIDENTS")
print("=" * 70)

display_columns = [
    "incident_id",
    "incident_start",
    "incident_end",
    "duration_minutes",
    "anomalous_windows",
    "incident_severity",
    "incident_type",
    "max_aiops_anomaly_score",
    "max_error_rate",
    "max_semantic_anomaly_rate",
]

print(
    incidents[
        display_columns
    ].head(20).to_string(index=False)
)


# ============================================================
# SAUVEGARDE
# ============================================================

os.makedirs(
    os.path.dirname(OUTPUT_FILE),
    exist_ok=True
)

incidents.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\n" + "=" * 70)
print("FICHIER CREE")
print("=" * 70)

print(f"\n{OUTPUT_FILE}")

print("\nTerminé.")