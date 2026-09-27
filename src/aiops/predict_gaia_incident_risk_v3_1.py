"""
HARDTEC AIOps
Incident Risk Prediction V3.1 - Balanced Risk Engine

Objectif
--------
Produire un score de risque d'incident à court terme à partir de :
- anomalie AIOps
- persistance temporelle
- taux d'erreur
- évolution du taux d'erreur
- volume de logs
- évolution du volume
- anomalies sémantiques
- évolution des anomalies sémantiques
- diversité des services / sources

V3.1 est une version équilibrée entre V2 et V3.

V2 :
    Bonne couverture mais trop de faux positifs.

V3 :
    Trop restrictive :
    - minimum 5 logs
    - 2 signaux obligatoires
    - 2 fenêtres persistantes obligatoires

V3.1 :
    - pénalise les très faibles volumes
    - utilise la persistance comme bonus
    - utilise la confirmation multi-signal comme bonus
    - accepte un signal unique lorsqu'il est suffisamment fort
    - conserve une évaluation au niveau épisode

IMPORTANT
---------
Ce moteur est un système de scoring prédictif hybride.
Ce n'est PAS un classifieur supervisé d'incidents.
"""

from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]

INPUT_ANOMALIES = (
    BASE_DIR
    / "data"
    / "processed"
    / "gaia_temporal_anomalies.csv"
)

INPUT_INCIDENTS = (
    BASE_DIR
    / "data"
    / "processed"
    / "gaia_incidents.csv"
)

OUTPUT_RISK = (
    BASE_DIR
    / "data"
    / "processed"
    / "gaia_incident_risk_v3_1.csv"
)

OUTPUT_EVALUATION = (
    BASE_DIR
    / "data"
    / "processed"
    / "gaia_incident_evaluation_v3_1.csv"
)

OUTPUT_COMPARISON = (
    BASE_DIR
    / "data"
    / "processed"
    / "gaia_risk_comparison_v3_1.csv"
)


# ============================================================
# PREDICTION PARAMETERS
# ============================================================

HORIZON_MINUTES = 15

TRAIN_RATIO = 0.70

EARLY_WARNING_THRESHOLD = 40.0

HIGH_RISK_THRESHOLD = 60.0

CRITICAL_THRESHOLD = 80.0


# ============================================================
# VOLUME PARAMETERS
# ============================================================

MIN_EVENT_VOLUME = 5

VERY_LOW_VOLUME = 2

MEANINGFUL_VOLUME = 5

STRONG_VOLUME = 10


# ============================================================
# TEMPORAL PARAMETERS
# ============================================================

PERSISTENCE_WINDOW = 3

SINGLE_SIGNAL_THRESHOLD = 52.0

STRONG_ANOMALY_THRESHOLD = 0.65

STRONG_RATE_THRESHOLD = 0.70


# ============================================================
# UTILITY FUNCTIONS
# ============================================================


def minmax_series(series):
    """
    Normalise une série entre 0 et 1.
    """

    series = pd.to_numeric(
        series,
        errors="coerce",
    ).fillna(0.0)

    min_value = series.min()

    max_value = series.max()

    if max_value == min_value:
        return pd.Series(
            0.0,
            index=series.index,
        )

    return (
        (series - min_value)
        / (max_value - min_value)
    )


def safe_numeric(df, columns):
    """
    Convertit les colonnes numériques.
    """

    for col in columns:

        if col in df.columns:

            df[col] = pd.to_numeric(
                df[col],
                errors="coerce",
            ).fillna(0.0)

    return df


def print_separator():
    print("=" * 70)


# ============================================================
# START
# ============================================================

print_separator()

print(
    "HARDTEC AIOps - INCIDENT RISK PREDICTION V3.1"
)

print(
    "Balanced Risk Engine"
)

print_separator()


# ============================================================
# LOAD DATA
# ============================================================

print("\nChargement des données...")


if not INPUT_ANOMALIES.exists():

    raise FileNotFoundError(
        f"Fichier introuvable : {INPUT_ANOMALIES}"
    )


if not INPUT_INCIDENTS.exists():

    raise FileNotFoundError(
        f"Fichier introuvable : {INPUT_INCIDENTS}"
    )


df = pd.read_csv(
    INPUT_ANOMALIES
)


incidents = pd.read_csv(
    INPUT_INCIDENTS
)


print(
    f"Fenêtres AIOps : {len(df):,}"
)

print(
    f"Lignes incidents : {len(incidents):,}"
)


# ============================================================
# TIMESTAMP NORMALIZATION
# ============================================================

print("\nNormalisation des timestamps...")


df["window_start"] = pd.to_datetime(
    df["window_start"],
    utc=True,
    errors="coerce",
)


incidents["incident_start"] = pd.to_datetime(
    incidents["incident_start"],
    utc=True,
    errors="coerce",
)


incidents["incident_end"] = pd.to_datetime(
    incidents["incident_end"],
    utc=True,
    errors="coerce",
)


df = df.dropna(
    subset=["window_start"]
).copy()


incidents = incidents.dropna(
    subset=[
        "incident_start",
        "incident_end",
    ]
).copy()


incidents = incidents.sort_values(
    "incident_start"
).reset_index(drop=True)


print(
    f"Incidents valides : {len(incidents):,}"
)


# ============================================================
# INCIDENT INTERVALS
# ============================================================

print(
    "\nConstruction des intervalles d'incidents..."
)


incident_intervals = []


for _, row in incidents.iterrows():

    incident_intervals.append(
        {
            "incident_id": str(
                row["incident_id"]
            ),
            "start": row["incident_start"],
            "end": row["incident_end"],
        }
    )


# ============================================================
# CURRENT INCIDENT
# ============================================================

df["current_incident_id"] = pd.Series(
    pd.NA,
    index=df.index,
    dtype="string",
)


for incident in incident_intervals:

    mask = (
        (df["window_start"] >= incident["start"])
        &
        (df["window_start"] <= incident["end"])
    )

    df.loc[
        mask,
        "current_incident_id",
    ] = incident["incident_id"]


df["incident_now"] = (
    df["current_incident_id"].notna()
)


# ============================================================
# NEXT INCIDENT
# ============================================================

print(
    "\nRecherche du prochain incident..."
)


df["next_incident_id"] = pd.Series(
    pd.NA,
    index=df.index,
    dtype="string",
)


# IMPORTANT:
# On utilise datetime64[ns, UTC]
# pour éviter le conflit tz-naive / tz-aware.

df["next_incident_start"] = pd.Series(
    pd.NaT,
    index=df.index,
    dtype="datetime64[ns, UTC]",
)


incident_starts = incidents[
    [
        "incident_id",
        "incident_start",
    ]
].sort_values(
    "incident_start"
)


for idx in df.index:

    current_time = df.at[
        idx,
        "window_start",
    ]

    future = incident_starts[
        incident_starts["incident_start"]
        > current_time
    ]

    if len(future) == 0:
        continue

    next_row = future.iloc[0]

    df.at[
        idx,
        "next_incident_id",
    ] = str(
        next_row["incident_id"]
    )

    df.at[
        idx,
        "next_incident_start",
    ] = next_row[
        "incident_start"
    ]


# ============================================================
# FUTURE INCIDENT
# ============================================================

df["minutes_to_next_incident"] = (
    df["next_incident_start"]
    - df["window_start"]
).dt.total_seconds() / 60.0


df["future_incident"] = (
    (~df["incident_now"])
    &
    (
        df["minutes_to_next_incident"]
        > 0
    )
    &
    (
        df["minutes_to_next_incident"]
        <= HORIZON_MINUTES
    )
)


# ============================================================
# TEMPORAL SORT
# ============================================================

df = df.sort_values(
    "window_start"
).reset_index(
    drop=True
)


# ============================================================
# NUMERIC FEATURES
# ============================================================

numeric_columns = [
    "total_logs",
    "error_count",
    "error_rate",
    "unique_sources",
    "unique_hosts",
    "previous_error_rate",
    "error_rate_delta",
    "previous_total_logs",
    "volume_delta",
    "rolling_error_rate_3",
    "rolling_volume_3",
    "semantic_total_logs",
    "semantic_anomaly_count",
    "semantic_anomaly_rate",
    "semantic_mean_confidence",
    "aiops_anomaly_score",
]


df = safe_numeric(
    df,
    numeric_columns,
)


# ============================================================
# SIGNALS
# ============================================================

print(
    "\nConstruction des signaux temporels..."
)


# ------------------------------------------------------------
# AIOps anomaly intensity
# ------------------------------------------------------------

df["signal_anomaly"] = (
    df["aiops_anomaly_score"]
    .clip(0, 1)
)


# ------------------------------------------------------------
# Error rate
# ------------------------------------------------------------

df["signal_error"] = (
    df["error_rate"]
    .clip(0, 1)
)


# ------------------------------------------------------------
# Error rate increase
# ------------------------------------------------------------

df["signal_error_increase"] = (
    minmax_series(
        df["error_rate_delta"]
    )
)


# ------------------------------------------------------------
# Traffic
# ------------------------------------------------------------

df["signal_traffic"] = (
    minmax_series(
        df["total_logs"]
    )
)


# ------------------------------------------------------------
# Traffic increase
# ------------------------------------------------------------

df["signal_traffic_increase"] = (
    minmax_series(
        df["volume_delta"]
    )
)


# ------------------------------------------------------------
# Semantic anomalies
# ------------------------------------------------------------

df["signal_semantic"] = (
    df["semantic_anomaly_rate"]
    .clip(0, 1)
)


# ------------------------------------------------------------
# Semantic increase
# ------------------------------------------------------------

df["semantic_delta"] = (
    df["semantic_anomaly_rate"]
    -
    df["semantic_anomaly_rate"]
    .shift(1)
    .fillna(0)
)


df["signal_semantic_increase"] = (
    minmax_series(
        df["semantic_delta"]
    )
)


# ------------------------------------------------------------
# Multi-service activity
# ------------------------------------------------------------

source_signal = minmax_series(
    df["unique_sources"]
)

host_signal = minmax_series(
    df["unique_hosts"]
)


df["signal_multi_service"] = (
    0.6 * source_signal
    +
    0.4 * host_signal
)


# ============================================================
# PERSISTENCE
# ============================================================

print(
    "\nConstruction de la persistance temporelle..."
)


df["anomaly_persistence"] = (
    df["signal_anomaly"]
    .rolling(
        PERSISTENCE_WINDOW,
        min_periods=1,
    )
    .mean()
)


df["error_persistence"] = (
    df["signal_error"]
    .rolling(
        PERSISTENCE_WINDOW,
        min_periods=1,
    )
    .mean()
)


df["semantic_persistence"] = (
    df["signal_semantic"]
    .rolling(
        PERSISTENCE_WINDOW,
        min_periods=1,
    )
    .mean()
)


df["persistent_anomaly_windows"] = (
    (
        df["signal_anomaly"]
        >= STRONG_ANOMALY_THRESHOLD
    )
    .rolling(
        PERSISTENCE_WINDOW,
        min_periods=1,
    )
    .sum()
)


# ============================================================
# VOLUME CONTROL
# ============================================================

print(
    "\nConstruction du contrôle du volume..."
)


df["volume_class"] = "VERY_LOW"


df.loc[
    df["total_logs"] >= MEANINGFUL_VOLUME,
    "volume_class",
] = "MEANINGFUL"


df.loc[
    df["total_logs"] >= STRONG_VOLUME,
    "volume_class",
] = "STRONG"


df["volume_confidence"] = (
    df["total_logs"]
    / STRONG_VOLUME
).clip(
    0,
    1,
)


# ============================================================
# MULTI-SIGNAL CONFIRMATION
# ============================================================

print(
    "\nConstruction de la confirmation multi-signal..."
)


signal_columns = [
    "signal_anomaly",
    "signal_error",
    "signal_traffic",
    "signal_semantic",
    "signal_multi_service",
]


df["confirming_signals"] = 0


for col in signal_columns:

    df["confirming_signals"] += (
        df[col] >= 0.50
    ).astype(int)


df["strong_signals"] = 0


strong_signal_columns = [
    "signal_anomaly",
    "signal_error",
    "signal_semantic",
]


for col in strong_signal_columns:

    df["strong_signals"] += (
        df[col]
        >= STRONG_RATE_THRESHOLD
    ).astype(int)


# ============================================================
# BASE RISK
# ============================================================

print(
    "\nCalcul du score de risque V3.1..."
)


df["base_risk_score"] = (
    0.25 * df["signal_anomaly"]
    +
    0.15 * df["anomaly_persistence"]
    +
    0.15 * df["signal_semantic"]
    +
    0.15 * df["signal_error"]
    +
    0.10 * df["signal_traffic"]
    +
    0.05 * df["signal_error_increase"]
    +
    0.05 * df["signal_traffic_increase"]
    +
    0.05 * df["signal_semantic_increase"]
    +
    0.05 * df["signal_multi_service"]
)


df["base_risk_score"] *= 100


# ============================================================
# VOLUME ADJUSTMENT
# ============================================================

def volume_adjustment(row):
    """
    Pénalisation progressive des fenêtres
    avec très faible volume.
    """

    logs = row["total_logs"]

    if logs <= 1:
        return 0.35

    if logs <= 2:
        return 0.50

    if logs < MIN_EVENT_VOLUME:
        return 0.70

    if logs < STRONG_VOLUME:
        return 0.90

    return 1.00


df["volume_adjustment"] = df.apply(
    volume_adjustment,
    axis=1,
)


# ============================================================
# PERSISTENCE BONUS
# ============================================================

def persistence_bonus(row):
    """
    La persistance est un bonus,
    pas une condition obligatoire.
    """

    persistent = row[
        "persistent_anomaly_windows"
    ]

    if persistent >= 3:
        return 8.0

    if persistent >= 2:
        return 5.0

    return 0.0


df["persistence_bonus"] = df.apply(
    persistence_bonus,
    axis=1,
)


# ============================================================
# MULTI-SIGNAL BONUS
# ============================================================

def multi_signal_bonus(row):
    """
    Plusieurs signaux concordants
    renforcent la confiance.
    """

    signals = row[
        "confirming_signals"
    ]

    if signals >= 4:
        return 8.0

    if signals >= 3:
        return 6.0

    if signals >= 2:
        return 3.0

    return 0.0


df["multi_signal_bonus"] = df.apply(
    multi_signal_bonus,
    axis=1,
)


# ============================================================
# STRONG SINGLE SIGNAL BONUS
# ============================================================

def strong_single_signal_bonus(row):
    """
    Permet à un signal suffisamment fort
    de générer un risque important sans
    exiger plusieurs signaux.
    """

    logs = row["total_logs"]

    strong_anomaly = (
        row["signal_anomaly"]
        >= STRONG_ANOMALY_THRESHOLD
    )

    strong_error = (
        row["signal_error"]
        >= STRONG_RATE_THRESHOLD
    )

    strong_semantic = (
        row["signal_semantic"]
        >= STRONG_RATE_THRESHOLD
    )

    if logs >= MEANINGFUL_VOLUME:

        if (
            strong_anomaly
            or strong_error
            or strong_semantic
        ):
            return 5.0

    return 0.0


df["strong_single_signal_bonus"] = df.apply(
    strong_single_signal_bonus,
    axis=1,
)


# ============================================================
# FINAL RISK SCORE
# ============================================================

df["risk_score"] = (
    df["base_risk_score"]
    *
    df["volume_adjustment"]
    +
    df["persistence_bonus"]
    +
    df["multi_signal_bonus"]
    +
    df["strong_single_signal_bonus"]
)


df["risk_score"] = (
    df["risk_score"]
    .clip(0, 100)
    .round(2)
)


# ============================================================
# RISK LEVEL
# ============================================================

def risk_level(score):

    if score >= CRITICAL_THRESHOLD:
        return "CRITICAL"

    if score >= HIGH_RISK_THRESHOLD:
        return "HIGH"

    if score >= EARLY_WARNING_THRESHOLD:
        return "MEDIUM"

    return "LOW"


df["risk_level"] = (
    df["risk_score"]
    .apply(risk_level)
)


# ============================================================
# RISK REASON
# ============================================================

def build_reason(row):

    reasons = []

    if row["signal_anomaly"] >= 0.65:

        reasons.append(
            "STRONG_AIOPS_ANOMALY"
        )

    if row["signal_error"] >= 0.70:

        reasons.append(
            "HIGH_ERROR_RATE"
        )

    if row["signal_error_increase"] >= 0.70:

        reasons.append(
            "RAPID_ERROR_RATE_INCREASE"
        )

    if row["signal_semantic"] >= 0.70:

        reasons.append(
            "HIGH_SEMANTIC_ANOMALY"
        )

    if row["signal_semantic_increase"] >= 0.70:

        reasons.append(
            "RAPID_SEMANTIC_INCREASE"
        )

    if row["signal_traffic"] >= 0.70:

        reasons.append(
            "HIGH_TRAFFIC_VOLUME"
        )

    if row["signal_traffic_increase"] >= 0.70:

        reasons.append(
            "RAPID_TRAFFIC_INCREASE"
        )

    if row["signal_multi_service"] >= 0.70:

        reasons.append(
            "MULTI_SERVICE_ACTIVITY"
        )

    if row[
        "persistent_anomaly_windows"
    ] >= 2:

        reasons.append(
            "ANOMALY_PERSISTENCE"
        )

    if row[
        "confirming_signals"
    ] >= 2:

        reasons.append(
            "MULTI_SIGNAL_CONFIRMATION"
        )

    if row[
        "volume_class"
    ] == "VERY_LOW":

        reasons.append(
            "LOW_EVENT_VOLUME"
        )

    if not reasons:

        reasons.append(
            "LOW_CONFIDENCE_OPERATIONAL_SIGNAL"
        )

    return " | ".join(reasons)


df["risk_reason"] = df.apply(
    build_reason,
    axis=1,
)


# ============================================================
# EARLY WARNING
# ============================================================

print(
    "\nApplication des règles Early Warning V3.1..."
)


df["early_warning_candidate"] = (
    (~df["incident_now"])
    &
    df["future_incident"]
    &
    (
        df["risk_score"]
        >= EARLY_WARNING_THRESHOLD
    )
)


# ============================================================
# STRONG SINGLE SIGNAL
# ============================================================

df["strong_single_signal"] = (
    (
        df["signal_anomaly"]
        >= STRONG_ANOMALY_THRESHOLD
    )
    |
    (
        (
            df["signal_error"]
            >= STRONG_RATE_THRESHOLD
        )
        &
        (
            df["total_logs"]
            >= MEANINGFUL_VOLUME
        )
    )
    |
    (
        (
            df["signal_semantic"]
            >= STRONG_RATE_THRESHOLD
        )
        &
        (
            df["total_logs"]
            >= MEANINGFUL_VOLUME
        )
    )
)


# ============================================================
# CONFIRMATION
# ============================================================

df["multi_signal_confirmation"] = (
    df["confirming_signals"] >= 2
)


# ============================================================
# PERSISTENCE
# ============================================================

df["persistent_signal"] = (
    df["persistent_anomaly_windows"]
    >= 2
)


# ============================================================
# BALANCED V3.1 DECISION
# ============================================================

df["v31_warning_logic"] = (
    (
        (
            df["risk_score"] >= 45
        )
        &
        (
            df["multi_signal_confirmation"]
        )
    )
    |
    (
        (
            df["risk_score"] >= 48
        )
        &
        (
            df["persistent_signal"]
        )
    )
    |
    (
        (
            df["risk_score"]
            >= SINGLE_SIGNAL_THRESHOLD
        )
        &
        (
            df["strong_single_signal"]
        )
    )
    |
    (
        (
            df["risk_score"] >= 55
        )
        &
        (
            df["total_logs"]
            >= MEANINGFUL_VOLUME
        )
    )
)


df["early_warning"] = (
    df["early_warning_candidate"]
    &
    df["v31_warning_logic"]
)


# ============================================================
# HIGH-RISK WARNING
# ============================================================

df["high_risk_warning"] = (
    df["early_warning"]
    &
    (
        df["risk_score"]
        >= HIGH_RISK_THRESHOLD
    )
)


# ============================================================
# FALSE POSITIVE WARNING
# ============================================================

df["false_positive_warning"] = (
    (~df["incident_now"])
    &
    (~df["future_incident"])
    &
    (
        df["risk_score"]
        >= EARLY_WARNING_THRESHOLD
    )
    &
    df["v31_warning_logic"]
)


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

split_index = int(
    len(df)
    * TRAIN_RATIO
)


df["dataset_split"] = "TRAIN"


df.loc[
    split_index:,
    "dataset_split",
] = "TEST"


test_start = df.iloc[
    split_index
]["window_start"]


test_incidents = incidents[
    incidents["incident_start"]
    >= test_start
].copy()


# ============================================================
# INCIDENT-LEVEL EVALUATION
# ============================================================

print(
    "\nEvaluation au niveau incident..."
)


evaluation_rows = []


for _, incident in test_incidents.iterrows():

    incident_id = str(
        incident["incident_id"]
    )

    incident_start = incident[
        "incident_start"
    ]

    candidate_warnings = df[
        (
            df["early_warning"]
        )
        &
        (
            df["next_incident_id"]
            == incident_id
        )
        &
        (
            df["window_start"]
            < incident_start
        )
    ].copy()


    if len(candidate_warnings) > 0:

        candidate_warnings[
            "lead_time"
        ] = (
            incident_start
            -
            candidate_warnings[
                "window_start"
            ]
        ).dt.total_seconds() / 60.0


        warning = (
            candidate_warnings
            .sort_values(
                "lead_time",
                ascending=False,
            )
            .iloc[0]
        )


        preceded = True


        lead_time = float(
            warning["lead_time"]
        )


        warning_score = float(
            warning["risk_score"]
        )


        warning_level = warning[
            "risk_level"
        ]


        warning_logs = float(
            warning["total_logs"]
        )


        warning_signals = int(
            warning[
                "confirming_signals"
            ]
        )

    else:

        preceded = False

        lead_time = np.nan

        warning_score = np.nan

        warning_level = pd.NA

        warning_logs = np.nan

        warning_signals = np.nan


    evaluation_rows.append(
        {
            "incident_id": incident_id,
            "incident_start": incident_start,
            "preceded_by_warning": preceded,
            "lead_time_minutes": lead_time,
            "warning_risk_score": warning_score,
            "warning_risk_level": warning_level,
            "warning_total_logs": warning_logs,
            "warning_confirming_signals": warning_signals,
        }
    )


evaluation = pd.DataFrame(
    evaluation_rows
)


# ============================================================
# METRICS
# ============================================================

test_incident_count = len(
    evaluation
)


preceded_count = int(
    evaluation[
        "preceded_by_warning"
    ].sum()
)


incident_coverage = (
    preceded_count
    /
    test_incident_count
    *
    100
    if test_incident_count > 0
    else 0
)


early_warning_windows = int(
    df["early_warning"].sum()
)


false_positive_windows = int(
    df["false_positive_warning"].sum()
)


warning_precision = (
    early_warning_windows
    /
    (
        early_warning_windows
        +
        false_positive_windows
    )
    *
    100
    if (
        early_warning_windows
        +
        false_positive_windows
    ) > 0
    else 0
)


lead_times = evaluation.loc[
    evaluation[
        "preceded_by_warning"
    ],
    "lead_time_minutes",
].dropna()


if len(lead_times) > 0:

    mean_lead_time = float(
        lead_times.mean()
    )

    median_lead_time = float(
        lead_times.median()
    )

else:

    mean_lead_time = 0.0

    median_lead_time = 0.0


# ============================================================
# LEAD TIME COVERAGE
# ============================================================

coverage_5 = int(
    (
        evaluation[
            "lead_time_minutes"
        ]
        >= 5
    ).sum()
)


coverage_10 = int(
    (
        evaluation[
            "lead_time_minutes"
        ]
        >= 10
    ).sum()
)


coverage_15 = int(
    (
        evaluation[
            "lead_time_minutes"
        ]
        >= 15
    ).sum()
)


coverage_5_pct = (
    coverage_5
    /
    test_incident_count
    *
    100
    if test_incident_count > 0
    else 0
)


coverage_10_pct = (
    coverage_10
    /
    test_incident_count
    *
    100
    if test_incident_count > 0
    else 0
)


coverage_15_pct = (
    coverage_15
    /
    test_incident_count
    *
    100
    if test_incident_count > 0
    else 0
)


# ============================================================
# PRINT RESULTS
# ============================================================

print("\n")

print_separator()

print(
    "RESULTATS INCIDENT RISK PREDICTION V3.1"
)

print_separator()


print(
    f"\nFenêtres analysées        : "
    f"{len(df):,}"
)


print(
    f"Incidents totaux          : "
    f"{len(incidents):,}"
)


print(
    f"Incidents dans TEST       : "
    f"{test_incident_count:,}"
)


print(
    f"Horizon de prédiction     : "
    f"{HORIZON_MINUTES} minutes"
)


print(
    f"\nEarly-warning windows     : "
    f"{early_warning_windows}"
)


print(
    f"False-positive windows    : "
    f"{false_positive_windows}"
)


print(
    f"Incidents précédés       : "
    f"{preceded_count}/"
    f"{test_incident_count}"
)


print(
    f"Incident coverage        : "
    f"{incident_coverage:.2f}%"
)


print(
    f"Warning precision        : "
    f"{warning_precision:.2f}%"
)


print(
    f"Lead time moyen          : "
    f"{mean_lead_time:.2f} min"
)


print(
    f"Lead time médian         : "
    f"{median_lead_time:.2f} min"
)


print(
    "\nCouverture des incidents :"
)


print(
    f"  >= 5 minutes  : "
    f"{coverage_5}/"
    f"{test_incident_count} "
    f"({coverage_5_pct:.2f}%)"
)


print(
    f"  >= 10 minutes : "
    f"{coverage_10}/"
    f"{test_incident_count} "
    f"({coverage_10_pct:.2f}%)"
)


print(
    f"  >= 15 minutes : "
    f"{coverage_15}/"
    f"{test_incident_count} "
    f"({coverage_15_pct:.2f}%)"
)


# ============================================================
# RISK DISTRIBUTION
# ============================================================

print(
    "\nDistribution du risque :"
)


risk_distribution = (
    df["risk_level"]
    .value_counts()
)


for level in [
    "CRITICAL",
    "HIGH",
    "MEDIUM",
    "LOW",
]:

    count = int(
        risk_distribution.get(
            level,
            0,
        )
    )

    print(
        f"  {level:<9} : {count:,}"
    )


# ============================================================
# TOP EARLY WARNINGS
# ============================================================

print(
    "\nTop early warnings :"
)


top_warnings = df[
    df["early_warning"]
].sort_values(
    [
        "risk_score",
        "window_start",
    ],
    ascending=[
        False,
        True,
    ],
)


if len(top_warnings) == 0:

    print(
        "Aucun early warning détecté."
    )

else:

    display_columns = [
        "window_start",
        "next_incident_id",
        "minutes_to_next_incident",
        "risk_score",
        "risk_level",
        "total_logs",
        "error_rate",
        "semantic_anomaly_rate",
        "confirming_signals",
        "persistent_anomaly_windows",
    ]


    print(
        top_warnings[
            display_columns
        ]
        .head(20)
        .to_string(
            index=False
        )
    )


# ============================================================
# INCIDENT EVALUATION
# ============================================================

print(
    "\nEvaluation par incident :"
)


print(
    evaluation.to_string(
        index=False
    )
)


# ============================================================
# SAVE RISK DATA
# ============================================================

df.to_csv(
    OUTPUT_RISK,
    index=False,
)


# ============================================================
# SAVE EVALUATION
# ============================================================

evaluation.to_csv(
    OUTPUT_EVALUATION,
    index=False,
)


# ============================================================
# V1 / V2 / V3 / V3.1 COMPARISON
# ============================================================

print(
    "\nConstruction de la comparaison "
    "V1 / V2 / V3 / V3.1..."
)


comparison_rows = []


# ============================================================
# V1
# ============================================================

v1_file = (
    BASE_DIR
    / "data"
    / "processed"
    / "gaia_incident_risk.csv"
)


if v1_file.exists():

    v1 = pd.read_csv(
        v1_file
    )


    v1_early = int(
        v1.get(
            "early_warning",
            pd.Series(
                dtype=bool
            ),
        )
        .fillna(False)
        .sum()
    )


    v1_fp = int(
        v1.get(
            "false_positive_warning",
            pd.Series(
                dtype=bool
            ),
        )
        .fillna(False)
        .sum()
    )


    comparison_rows.append(
        {
            "version": "V1",
            "early_warning_windows": v1_early,
            "false_positive_windows": v1_fp,
        }
    )


# ============================================================
# V2
# ============================================================

v2_file = (
    BASE_DIR
    / "data"
    / "processed"
    / "gaia_incident_risk_v2.csv"
)


if v2_file.exists():

    v2 = pd.read_csv(
        v2_file
    )


    v2_early = int(
        v2.get(
            "early_warning",
            pd.Series(
                dtype=bool
            ),
        )
        .fillna(False)
        .sum()
    )


    v2_fp = int(
        v2.get(
            "false_positive_warning",
            pd.Series(
                dtype=bool
            ),
        )
        .fillna(False)
        .sum()
    )


    comparison_rows.append(
        {
            "version": "V2",
            "early_warning_windows": v2_early,
            "false_positive_windows": v2_fp,
        }
    )


# ============================================================
# V3
# ============================================================

v3_file = (
    BASE_DIR
    / "data"
    / "processed"
    / "gaia_incident_risk_v3.csv"
)


if v3_file.exists():

    v3 = pd.read_csv(
        v3_file
    )


    v3_early = int(
        v3.get(
            "early_warning",
            pd.Series(
                dtype=bool
            ),
        )
        .fillna(False)
        .sum()
    )


    v3_fp = int(
        v3.get(
            "false_positive_warning",
            pd.Series(
                dtype=bool
            ),
        )
        .fillna(False)
        .sum()
    )


    comparison_rows.append(
        {
            "version": "V3",
            "early_warning_windows": v3_early,
            "false_positive_windows": v3_fp,
        }
    )


# ============================================================
# V3.1
# ============================================================

comparison_rows.append(
    {
        "version": "V3.1",
        "early_warning_windows": (
            early_warning_windows
        ),
        "false_positive_windows": (
            false_positive_windows
        ),
    }
)


comparison = pd.DataFrame(
    comparison_rows
)


comparison["test_incidents"] = (
    test_incident_count
)


comparison[
    "incidents_preceded"
] = np.nan


comparison[
    "incident_coverage_pct"
] = np.nan


comparison[
    "warning_precision_pct"
] = np.nan


comparison.loc[
    comparison["version"] == "V3.1",
    "incidents_preceded",
] = preceded_count


comparison.loc[
    comparison["version"] == "V3.1",
    "incident_coverage_pct",
] = incident_coverage


comparison.loc[
    comparison["version"] == "V3.1",
    "warning_precision_pct",
] = warning_precision


comparison.to_csv(
    OUTPUT_COMPARISON,
    index=False,
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n")

print_separator()

print(
    "FICHIERS SAUVEGARDES"
)

print_separator()


print(
    f"\nRisk V3.1 :\n"
    f"{OUTPUT_RISK}"
)


print(
    f"\nIncident evaluation V3.1 :\n"
    f"{OUTPUT_EVALUATION}"
)


print(
    f"\nComparaison V1 / V2 / V3 / V3.1 :\n"
    f"{OUTPUT_COMPARISON}"
)


print("\n")

print_separator()

print(
    "Terminé."
)

print_separator()