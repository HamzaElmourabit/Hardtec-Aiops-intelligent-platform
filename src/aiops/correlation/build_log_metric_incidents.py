from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[3]

LOG_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "gaia_temporal_anomalies.csv"
)

METRIC_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "gaia_metrics"
    / "gaia_metric_anomaly_predictions_v3.csv"
)

OUTPUT_DIR = (
    BASE_DIR
    / "data"
    / "processed"
    / "gaia_correlation"
)

OUTPUT_FILE = OUTPUT_DIR / "gaia_log_metric_correlation.csv"
INCIDENT_FILE = OUTPUT_DIR / "gaia_multi_source_incidents.csv"


# ============================================================
# PARAMETRES
# ============================================================

# Une métrique est considérée comme anormale lorsqu'au moins
# un détecteur time-series la signale.
#
# Les colonnes sont :
# robust_z_prediction
# rolling_mad_prediction
# ewma_prediction
# isolation_forest_prediction

# Fenêtre maximale de corrélation temporelle.
# Les logs sont en fenêtres de 5 minutes.
CORRELATION_WINDOW_MINUTES = 5

# Nombre minimum de métriques anormales pour considérer
# qu'un événement métrique est significatif.
MIN_METRIC_ANOMALIES = 1

# Nombre minimum de sources métriques distinctes pour
# classer une corrélation comme multi-metric.
MIN_METRIC_SERIES = 1


# ============================================================
# UTILITAIRES
# ============================================================

def safe_mean(series):
    values = pd.to_numeric(series, errors="coerce").dropna()

    if len(values) == 0:
        return 0.0

    return float(values.mean())


def safe_max(series):
    values = pd.to_numeric(series, errors="coerce").dropna()

    if len(values) == 0:
        return 0.0

    return float(values.max())


# ============================================================
# CHARGEMENT LOGS
# ============================================================

print("=" * 70)
print("HARDTEC AIOPS - LOG + METRIC CORRELATION")
print("=" * 70)

print("\nChargement des logs...")

logs = pd.read_csv(LOG_FILE)

logs["window_start"] = pd.to_datetime(
    logs["window_start"],
    utc=True,
    errors="coerce",
)

logs = logs.dropna(subset=["window_start"]).copy()

logs["log_anomaly"] = (
    logs["is_aiops_anomaly"].fillna(0).astype(int)
)

logs["log_anomaly_score"] = pd.to_numeric(
    logs["aiops_anomaly_score"],
    errors="coerce",
).fillna(0)

print(f"Fenêtres logs : {len(logs):,}")
print(
    f"Fenêtres logs anormales : "
    f"{logs['log_anomaly'].sum():,}"
)


# ============================================================
# CHARGEMENT METRICS
# ============================================================

print("\nChargement des métriques...")

metrics = pd.read_csv(METRIC_FILE)

metrics["timestamp"] = pd.to_datetime(
    metrics["timestamp"],
    utc=True,
    errors="coerce",
)

metrics = metrics.dropna(subset=["timestamp"]).copy()

print(f"Points métriques : {len(metrics):,}")
print(
    f"Séries métriques : "
    f"{metrics['series_id'].nunique():,}"
)


# ============================================================
# DETECTION METRIQUE
# ============================================================

print("\nConstruction des anomalies métriques...")

prediction_columns = [
    "robust_z_prediction",
    "rolling_mad_prediction",
    "ewma_prediction",
    "isolation_forest_prediction",
]

available_prediction_columns = [
    column
    for column in prediction_columns
    if column in metrics.columns
]

if not available_prediction_columns:
    raise ValueError(
        "Aucune colonne de prediction métrique disponible."
    )


# Une métrique est anormale si au moins un détecteur
# la considère comme anomalie.

metrics["metric_anomaly_votes"] = (
    metrics[available_prediction_columns]
    .fillna(0)
    .astype(int)
    .sum(axis=1)
)

metrics["metric_anomaly"] = (
    metrics["metric_anomaly_votes"]
    >= MIN_METRIC_ANOMALIES
).astype(int)


# Score métrique basé sur le meilleur score des détecteurs.

score_columns = [
    "robust_z_score",
    "rolling_mad_score",
    "ewma_score",
]

available_score_columns = [
    column
    for column in score_columns
    if column in metrics.columns
]

if available_score_columns:

    metrics["metric_anomaly_score"] = (
        metrics[available_score_columns]
        .apply(pd.to_numeric, errors="coerce")
        .max(axis=1)
        .fillna(0)
    )

else:

    metrics["metric_anomaly_score"] = 0.0


print(
    f"Points métriques anormaux : "
    f"{metrics['metric_anomaly'].sum():,}"
)

print(
    f"Taux métrique anormal : "
    f"{metrics['metric_anomaly'].mean():.2%}"
)


# ============================================================
# AGREGATION METRICS PAR FENETRE DE 5 MINUTES
# ============================================================

print("\nAgrégation métrique par fenêtre de 5 minutes...")

metrics["window_start"] = (
    metrics["timestamp"]
    .dt.floor("5min")
)


metric_windows = (
    metrics
    .groupby("window_start")
    .agg(
        metric_points=("value", "count"),

        metric_anomaly_points=(
            "metric_anomaly",
            "sum",
        ),

        metric_anomaly_rate=(
            "metric_anomaly",
            "mean",
        ),

        metric_max_anomaly_score=(
            "metric_anomaly_score",
            "max",
        ),

        metric_mean_anomaly_score=(
            "metric_anomaly_score",
            "mean",
        ),

        metric_unique_series=(
            "series_id",
            "nunique",
        ),

        metric_mean_value=(
            "value",
            "mean",
        ),

        metric_std_value=(
            "value",
            "std",
        ),
    )
    .reset_index()
)


metric_windows["metric_anomaly"] = (
    metric_windows["metric_anomaly_points"]
    >= MIN_METRIC_ANOMALIES
).astype(int)


print(
    f"Fenêtres métriques : "
    f"{len(metric_windows):,}"
)

print(
    f"Fenêtres métriques anormales : "
    f"{metric_windows['metric_anomaly'].sum():,}"
)


# ============================================================
# CORRELATION LOGS + METRICS
# ============================================================

print("\nCorrélation temporelle Logs + Metrics...")

correlation = logs.merge(
    metric_windows,
    on="window_start",
    how="left",
)


# Les fenêtres sans métrique disponible sont conservées.
metric_numeric_columns = [
    "metric_points",
    "metric_anomaly_points",
    "metric_anomaly_rate",
    "metric_max_anomaly_score",
    "metric_mean_anomaly_score",
    "metric_unique_series",
    "metric_mean_value",
    "metric_std_value",
    "metric_anomaly",
]

for column in metric_numeric_columns:

    if column in correlation.columns:

        correlation[column] = (
            pd.to_numeric(
                correlation[column],
                errors="coerce",
            )
            .fillna(0)
        )


# ============================================================
# SIGNALS MULTI-SOURCE
# ============================================================

correlation["log_signal"] = (
    correlation["log_anomaly"] == 1
).astype(int)

correlation["metric_signal"] = (
    correlation["metric_anomaly"] == 1
).astype(int)


correlation["multi_source_signal"] = (
    (
        correlation["log_signal"] == 1
    )
    &
    (
        correlation["metric_signal"] == 1
    )
).astype(int)


correlation["log_only_signal"] = (
    (
        correlation["log_signal"] == 1
    )
    &
    (
        correlation["metric_signal"] == 0
    )
).astype(int)


correlation["metric_only_signal"] = (
    (
        correlation["log_signal"] == 0
    )
    &
    (
        correlation["metric_signal"] == 1
    )
).astype(int)


# ============================================================
# SCORE MULTI-SOURCE
# ============================================================

# Score transparent, destiné à la corrélation opérationnelle.
#
# 50% anomalie logs
# 30% anomalie métrique
# 20% intensité métrique

correlation["multi_source_score"] = (
    50
    * correlation["log_anomaly_score"].clip(0, 1)
    +
    30
    * correlation["metric_anomaly"].clip(0, 1)
    +
    20
    * correlation["metric_max_anomaly_score"].clip(0, 1)
)


# ============================================================
# CLASSIFICATION DU SIGNAL
# ============================================================

def classify_signal(row):

    log_signal = row["log_signal"]
    metric_signal = row["metric_signal"]

    if log_signal == 1 and metric_signal == 1:

        return "MULTI_SOURCE"

    if log_signal == 1:

        return "LOG_ONLY"

    if metric_signal == 1:

        return "METRIC_ONLY"

    return "NORMAL"


correlation["signal_type"] = correlation.apply(
    classify_signal,
    axis=1,
)


# ============================================================
# SEVERITE
# ============================================================

def classify_severity(row):

    score = row["multi_source_score"]

    if row["multi_source_signal"] == 1:

        if score >= 80:
            return "CRITICAL"

        if score >= 60:
            return "HIGH"

        if score >= 40:
            return "MEDIUM"

        return "LOW"

    if row["log_signal"] == 1:

        if row["log_anomaly_score"] >= 0.80:
            return "HIGH"

        if row["log_anomaly_score"] >= 0.60:
            return "MEDIUM"

        return "LOW"

    if row["metric_signal"] == 1:

        if row["metric_max_anomaly_score"] >= 0.80:
            return "HIGH"

        if row["metric_max_anomaly_score"] >= 0.60:
            return "MEDIUM"

        return "LOW"

    return "NORMAL"


correlation["correlation_severity"] = (
    correlation.apply(
        classify_severity,
        axis=1,
    )
)


# ============================================================
# RAISONS
# ============================================================

def build_reason(row):

    reasons = []

    if row["log_signal"] == 1:

        reasons.append(
            "log_anomaly"
        )

    if row["metric_signal"] == 1:

        reasons.append(
            "metric_anomaly"
        )

    if row["metric_anomaly_rate"] >= 0.20:

        reasons.append(
            "high_metric_anomaly_density"
        )

    if row["metric_unique_series"] >= 2:

        reasons.append(
            "multi_metric_series"
        )

    if row["error_rate"] >= 0.50:

        reasons.append(
            "high_log_error_rate"
        )

    if row["semantic_anomaly_rate"] >= 0.50:

        reasons.append(
            "high_log_semantic_anomaly_rate"
        )

    if not reasons:

        return "normal"

    return " + ".join(reasons)


correlation["correlation_reason"] = (
    correlation.apply(
        build_reason,
        axis=1,
    )
)


# ============================================================
# INCIDENT SIGNAL
# ============================================================

correlation["incident_signal"] = (
    (
        correlation["multi_source_signal"] == 1
    )
    |
    (
        correlation["log_signal"] == 1
    )
    |
    (
        correlation["metric_signal"] == 1
    )
).astype(int)


# ============================================================
# SAUVEGARDE CORRELATION
# ============================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

correlation.to_csv(
    OUTPUT_FILE,
    index=False,
)


# ============================================================
# CREATION DES EPISODES D'INCIDENT
# ============================================================

print("\nConstruction des épisodes multi-source...")

signals = correlation[
    correlation["incident_signal"] == 1
].copy()

signals = signals.sort_values(
    "window_start"
).reset_index(drop=True)


if len(signals) == 0:

    incidents = pd.DataFrame()

else:

    time_difference = (
        signals["window_start"]
        .diff()
        .dt.total_seconds()
        .div(60)
    )

    # Nouveau groupe lorsque le gap dépasse 5 minutes.

    signals["episode_break"] = (
        time_difference > CORRELATION_WINDOW_MINUTES
    ).astype(int)

    signals["episode_group"] = (
        signals["episode_break"]
        .cumsum()
    )

    incidents = (
        signals
        .groupby("episode_group")
        .agg(
            incident_start=(
                "window_start",
                "min",
            ),

            incident_end=(
                "window_start",
                "max",
            ),

            incident_windows=(
                "window_start",
                "count",
            ),

            max_log_anomaly_score=(
                "log_anomaly_score",
                "max",
            ),

            max_metric_anomaly_score=(
                "metric_max_anomaly_score",
                "max",
            ),

            max_metric_anomaly_rate=(
                "metric_anomaly_rate",
                "max",
            ),

            mean_log_error_rate=(
                "error_rate",
                "mean",
            ),

            max_log_error_rate=(
                "error_rate",
                "max",
            ),

            max_semantic_anomaly_rate=(
                "semantic_anomaly_rate",
                "max",
            ),

            max_metric_series=(
                "metric_unique_series",
                "max",
            ),

            total_metric_anomaly_points=(
                "metric_anomaly_points",
                "sum",
            ),

            multi_source_windows=(
                "multi_source_signal",
                "sum",
            ),

            log_only_windows=(
                "log_only_signal",
                "sum",
            ),

            metric_only_windows=(
                "metric_only_signal",
                "sum",
            ),

            max_multi_source_score=(
                "multi_source_score",
                "max",
            ),

            mean_multi_source_score=(
                "multi_source_score",
                "mean",
            ),
        )
        .reset_index(drop=True)
    )

    incidents["incident_id"] = [
        f"INC-MS-{i:04d}"
        for i in range(1, len(incidents) + 1)
    ]

    incidents["duration_minutes"] = (
        (
            incidents["incident_end"]
            -
            incidents["incident_start"]
        )
        .dt.total_seconds()
        .div(60)
        + 5
    )


    # ========================================================
    # TYPE INCIDENT
    # ========================================================

    def incident_type(row):

        if row["multi_source_windows"] > 0:

            if (
                row["max_metric_series"] >= 2
                and
                row["max_log_error_rate"] >= 0.50
            ):

                return "MULTI_SOURCE_MULTI_SERVICE"

            return "LOG_METRIC_CORRELATED"

        if row["log_only_windows"] > 0:

            return "LOG_ONLY"

        return "METRIC_ONLY"


    incidents["incident_type"] = (
        incidents.apply(
            incident_type,
            axis=1,
        )
    )


    # ========================================================
    # SEVERITE INCIDENT
    # ========================================================

    def incident_severity(row):

        score = row["max_multi_source_score"]

        if score >= 80:

            return "CRITICAL"

        if score >= 60:

            return "HIGH"

        if score >= 40:

            return "MEDIUM"

        return "LOW"


    incidents["incident_severity"] = (
        incidents.apply(
            incident_severity,
            axis=1,
        )
    )


    # ========================================================
    # RAISON INCIDENT
    # ========================================================

    def incident_reason(row):

        reasons = []

        if row["multi_source_windows"] > 0:

            reasons.append(
                "logs_and_metrics_correlated"
            )

        if row["max_log_error_rate"] >= 0.50:

            reasons.append(
                "high_log_error_rate"
            )

        if row["max_semantic_anomaly_rate"] >= 0.50:

            reasons.append(
                "high_semantic_anomaly_rate"
            )

        if row["max_metric_anomaly_rate"] >= 0.20:

            reasons.append(
                "high_metric_anomaly_density"
            )

        if row["max_metric_series"] >= 2:

            reasons.append(
                "multiple_metric_series"
            )

        if not reasons:

            reasons.append(
                "single_source_anomaly"
            )

        return " + ".join(reasons)


    incidents["incident_reason"] = (
        incidents.apply(
            incident_reason,
            axis=1,
        )
    )


# ============================================================
# SAUVEGARDE INCIDENTS
# ============================================================

incidents.to_csv(
    INCIDENT_FILE,
    index=False,
)


# ============================================================
# STATISTIQUES FINALES
# ============================================================

print("\n" + "=" * 70)
print("RESULTATS DE LA CORRELATION")
print("=" * 70)

print(
    f"\nFenêtres logs : "
    f"{len(logs):,}"
)

print(
    f"Fenêtres logs anormales : "
    f"{logs['log_anomaly'].sum():,}"
)

print(
    f"Fenêtres métriques : "
    f"{len(metric_windows):,}"
)

print(
    f"Fenêtres métriques anormales : "
    f"{metric_windows['metric_anomaly'].sum():,}"
)

print(
    f"\nFenêtres multi-source : "
    f"{correlation['multi_source_signal'].sum():,}"
)

print(
    f"Fenêtres LOG_ONLY : "
    f"{correlation['log_only_signal'].sum():,}"
)

print(
    f"Fenêtres METRIC_ONLY : "
    f"{correlation['metric_only_signal'].sum():,}"
)

if len(incidents) > 0:

    print(
        f"\nEpisodes d'incidents : "
        f"{len(incidents):,}"
    )

    print("\nTypes :")

    print(
        incidents[
            "incident_type"
        ]
        .value_counts()
        .to_string()
    )

    print("\nSévérités :")

    print(
        incidents[
            "incident_severity"
        ]
        .value_counts()
        .to_string()
    )

    print("\nTop incidents :")

    print(
        incidents[
            [
                "incident_id",
                "incident_start",
                "incident_end",
                "incident_type",
                "incident_severity",
                "max_multi_source_score",
                "multi_source_windows",
            ]
        ]
        .sort_values(
            "max_multi_source_score",
            ascending=False,
        )
        .head(10)
        .to_string(index=False)
    )

else:

    print(
        "\nAucun épisode d'incident détecté."
    )


print("\n" + "-" * 70)

print(
    f"Correlation : "
    f"{OUTPUT_FILE}"
)

print(
    f"Incidents : "
    f"{INCIDENT_FILE}"
)

print("-" * 70)

print("\nCORRELATION LOGS + METRICS TERMINEE")