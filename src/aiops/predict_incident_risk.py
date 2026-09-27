
"""
HARDTEC - AIOps Incident Risk Prediction

Purpose
-------
Estimate the short-term risk of an incident occurring within the next
N real minutes using temporal operational signals.

Important
---------
This is a transparent AIOps Proof-of-Concept risk scoring system.

It is NOT a supervised incident classifier because the dataset contains
only 41 observed temporal windows and 9 correlated incident episodes.

The incident target is used retrospectively for evaluation only.
No future operational feature is used to calculate the risk score.
"""

from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_ANOMALIES = (
    PROJECT_ROOT / "data" / "processed" / "aiops_anomalies.csv"
)

INPUT_INCIDENTS = (
    PROJECT_ROOT / "data" / "processed" / "aiops_incidents.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT / "data" / "processed" / "aiops_incident_risk.csv"
)

PREDICTION_HORIZON_MINUTES = 3


# ============================================================
# HELPERS
# ============================================================


def safe_numeric(series):
    """Convert a pandas series to numeric safely."""
    return pd.to_numeric(series, errors="coerce").fillna(0.0)


def minmax_score(series):
    """
    Normalize a numerical series between 0 and 100.

    If the series is constant, return zeros.
    """
    values = safe_numeric(series)

    minimum = values.min()
    maximum = values.max()

    if pd.isna(minimum) or pd.isna(maximum) or maximum <= minimum:
        return pd.Series(0.0, index=series.index)

    return ((values - minimum) / (maximum - minimum) * 100.0).clip(0, 100)


def percentile_score(series, percentile=90):
    """
    Convert a metric into a 0-100 operational severity score.

    Values around or above the selected percentile receive a higher score.
    """
    values = safe_numeric(series)

    threshold = values.quantile(percentile / 100.0)

    if pd.isna(threshold) or threshold <= 0:
        return minmax_score(values)

    score = (values / threshold) * 100.0

    return score.clip(0, 100)


def mark_current_incidents(df):
    """
    Mark windows that are currently inside an incident episode.

    Incident intervals are taken from the enriched aiops_incidents.csv file.
    """

    df = df.copy()

    df["incident_now"] = False

    valid = df[
        df["incident_id"].notna()
        & df["incident_start"].notna()
    ].copy()

    if valid.empty:
        return df

    episodes = (
        valid[
            [
                "incident_id",
                "incident_start",
                "incident_end",
            ]
        ]
        .drop_duplicates(subset=["incident_id"])
        .copy()
    )

    episodes["incident_start"] = pd.to_datetime(
        episodes["incident_start"],
        errors="coerce",
        utc=True,
    )

    episodes["incident_end"] = pd.to_datetime(
        episodes["incident_end"],
        errors="coerce",
        utc=True,
    )

    episodes["incident_end"] = episodes["incident_end"].fillna(
        episodes["incident_start"]
    )

    for _, episode in episodes.iterrows():

        if pd.isna(episode["incident_start"]):
            continue

        start = episode["incident_start"]
        end = episode["incident_end"]

        mask = (
            (df["window_start"] >= start)
            & (df["window_start"] <= end)
        )

        df.loc[mask, "incident_now"] = True

    return df


def calculate_future_incident(df, horizon_minutes):
    """
    Determine whether an incident episode STARTS during the next
    N real minutes.

    This uses timestamps rather than dataframe row positions.

    Example:

        current = 13:38
        incident starts = 13:39
        horizon = 3 min

        future_incident = True

    But:

        current = 14:02
        incident starts = 05:52 next day

        future_incident = False

    because the actual time difference is much greater than 3 minutes.
    """

    df = df.copy()

    df["future_incident"] = False

    # --------------------------------------------------------
    # Extract unique incident episodes
    # --------------------------------------------------------

    episode_columns = [
        "incident_id",
        "incident_start",
        "incident_end",
    ]

    episodes = df[
        df["incident_id"].notna()
        & df["incident_start"].notna()
    ][episode_columns].copy()

    if episodes.empty:
        return df

    episodes = (
        episodes
        .drop_duplicates(subset=["incident_id"])
        .copy()
    )

    episodes["incident_start"] = pd.to_datetime(
        episodes["incident_start"],
        errors="coerce",
        utc=True,
    )

    episodes = episodes.dropna(subset=["incident_start"])

    episode_starts = episodes["incident_start"].sort_values().tolist()

    # --------------------------------------------------------
    # Evaluate each observed window
    # --------------------------------------------------------

    horizon = pd.Timedelta(minutes=horizon_minutes)

    future_flags = []

    for current_time in df["window_start"]:

        lower_bound = current_time
        upper_bound = current_time + horizon

        future = any(
            (start > lower_bound)
            and (start <= upper_bound)
            for start in episode_starts
        )

        future_flags.append(future)

    df["future_incident"] = future_flags

    # An incident that is already active is not considered
    # an "early warning" opportunity.
    df.loc[
        df["incident_now"],
        "future_incident"
    ] = False

    return df


def create_temporal_features(df):
    """
    Create temporal features using current and previously observed windows.

    No future values are used.
    """

    df = df.copy()

    base_features = [
        "anomaly_score",
        "is_anomaly",
        "event_count",
        "avg_duration_ms",
        "p95_duration_ms",
        "high_severity_rate",
        "error_density",
    ]

    for feature in base_features:
        if feature not in df.columns:
            df[feature] = 0.0

        df[feature] = safe_numeric(df[feature])

    # --------------------------------------------------------
    # Observed-window lags
    # --------------------------------------------------------

    for lag in range(1, 4):

        df[f"anomaly_score_lag_{lag}"] = (
            df["anomaly_score"].shift(lag).fillna(0)
        )

        df[f"is_anomaly_lag_{lag}"] = (
            df["is_anomaly"].shift(lag).fillna(0)
        )

        df[f"event_count_lag_{lag}"] = (
            df["event_count"].shift(lag).fillna(0)
        )

        df[f"avg_duration_lag_{lag}"] = (
            df["avg_duration_ms"].shift(lag).fillna(0)
        )

        df[f"p95_duration_lag_{lag}"] = (
            df["p95_duration_ms"].shift(lag).fillna(0)
        )

        df[f"error_density_lag_{lag}"] = (
            df["error_density"].shift(lag).fillna(0)
        )

    # --------------------------------------------------------
    # Deltas relative to previous observed window
    # --------------------------------------------------------

    df["event_count_delta"] = (
        df["event_count"]
        - df["event_count"].shift(1).fillna(df["event_count"])
    )

    df["avg_latency_delta"] = (
        df["avg_duration_ms"]
        - df["avg_duration_ms"].shift(1).fillna(
            df["avg_duration_ms"]
        )
    )

    df["p95_latency_delta"] = (
        df["p95_duration_ms"]
        - df["p95_duration_ms"].shift(1).fillna(
            df["p95_duration_ms"]
        )
    )

    df["error_density_delta"] = (
        df["error_density"]
        - df["error_density"].shift(1).fillna(
            df["error_density"]
        )
    )

    # --------------------------------------------------------
    # Rolling operational context
    # --------------------------------------------------------

    df["event_count_rolling_3"] = (
        df["event_count"]
        .rolling(window=3, min_periods=1)
        .mean()
    )

    df["avg_latency_rolling_3"] = (
        df["avg_duration_ms"]
        .rolling(window=3, min_periods=1)
        .mean()
    )

    df["p95_latency_rolling_3"] = (
        df["p95_duration_ms"]
        .rolling(window=3, min_periods=1)
        .mean()
    )

    df["error_density_rolling_3"] = (
        df["error_density"]
        .rolling(window=3, min_periods=1)
        .mean()
    )

    # --------------------------------------------------------
    # Anomaly persistence
    # --------------------------------------------------------

    anomaly_values = (
        pd.to_numeric(
            df["is_anomaly"],
            errors="coerce",
        )
        .fillna(0)
        .astype(int)
    )

    df["anomaly_persistence_3"] = (
        anomaly_values
        .rolling(window=3, min_periods=1)
        .sum()
    )

    return df


def calculate_risk_score(df):
    """
    Calculate a transparent operational risk score.

    Components:

        25% anomaly intensity
        20% anomaly persistence
        15% traffic / volume
        15% latency
        10% error density
         5% high severity
         5% volume increase
         3% latency increase
         2% error-density increase

    Total = 100%
    """

    df = df.copy()

    # --------------------------------------------------------
    # 1. Anomaly intensity
    # --------------------------------------------------------

    anomaly_score_component = minmax_score(
        df["anomaly_score"]
    )

    # --------------------------------------------------------
    # 2. Anomaly persistence
    # --------------------------------------------------------

    persistence_component = (
        safe_numeric(df["anomaly_persistence_3"])
        / 3.0
        * 100.0
    ).clip(0, 100)

    # Give a direct anomaly signal a strong operational meaning.
    is_anomaly_component = (
        safe_numeric(df["is_anomaly"]) * 100.0
    )

    anomaly_component = (
        0.60 * anomaly_score_component
        + 0.40 * is_anomaly_component
    ).clip(0, 100)

    # --------------------------------------------------------
    # 3. Traffic / volume
    # --------------------------------------------------------

    volume_component = percentile_score(
        df["event_count"],
        percentile=90,
    )

    # --------------------------------------------------------
    # 4. Latency
    # --------------------------------------------------------

    latency_component = percentile_score(
        df["p95_duration_ms"],
        percentile=90,
    )

    # --------------------------------------------------------
    # 5. Error density
    # --------------------------------------------------------

    error_component = (
        safe_numeric(df["error_density"]) * 100.0
    ).clip(0, 100)

    # --------------------------------------------------------
    # 6. High severity
    # --------------------------------------------------------

    severity_component = (
        safe_numeric(df["high_severity_rate"]) * 100.0
    ).clip(0, 100)

    # --------------------------------------------------------
    # 7. Volume increase
    # --------------------------------------------------------

    volume_delta_component = minmax_score(
        df["event_count_delta"].clip(lower=0)
    )

    # --------------------------------------------------------
    # 8. Latency increase
    # --------------------------------------------------------

    latency_delta_component = minmax_score(
        df["p95_latency_delta"].clip(lower=0)
    )

    # --------------------------------------------------------
    # 9. Error increase
    # --------------------------------------------------------

    error_delta_component = minmax_score(
        df["error_density_delta"].clip(lower=0)
    )

    # --------------------------------------------------------
    # Weighted risk score
    # --------------------------------------------------------

    risk_score = (
        0.25 * anomaly_component
        + 0.20 * persistence_component
        + 0.15 * volume_component
        + 0.15 * latency_component
        + 0.10 * error_component
        + 0.05 * severity_component
        + 0.05 * volume_delta_component
        + 0.03 * latency_delta_component
        + 0.02 * error_delta_component
    )

    df["risk_score"] = risk_score.clip(0, 100)

    # --------------------------------------------------------
    # Risk levels
    # --------------------------------------------------------

    df["risk_level"] = np.select(
        [
            df["risk_score"] >= 80,
            df["risk_score"] >= 60,
            df["risk_score"] >= 40,
        ],
        [
            "CRITICAL",
            "HIGH",
            "MEDIUM",
        ],
        default="LOW",
    )

    # --------------------------------------------------------
    # Risk explanation
    # --------------------------------------------------------

    reasons = []

    volume_threshold = safe_numeric(
        df["event_count"]
    ).quantile(0.90)

    latency_threshold = safe_numeric(
        df["p95_duration_ms"]
    ).quantile(0.90)

    for _, row in df.iterrows():

        current_reasons = []

        if row["is_anomaly"] == 1:
            current_reasons.append("anomaly signal")

        if row["anomaly_persistence_3"] >= 2:
            current_reasons.append(
                "persistent anomaly activity"
            )

        if row["event_count"] >= volume_threshold:
            current_reasons.append(
                "unusual event volume"
            )

        if row["p95_duration_ms"] >= latency_threshold:
            current_reasons.append(
                "high P95 latency"
            )

        if row["error_density"] >= 0.30:
            current_reasons.append(
                "high error density"
            )

        if row["high_severity_rate"] >= 0.15:
            current_reasons.append(
                "high severity concentration"
            )

        if row["event_count_delta"] > 0:
            current_reasons.append(
                "increasing event volume"
            )

        if row["p95_latency_delta"] > 0:
            current_reasons.append(
                "increasing latency"
            )

        if not current_reasons:
            current_reasons.append(
                "normal multivariate behavior"
            )

        reasons.append("; ".join(current_reasons))

    df["risk_reason"] = reasons

    return df


# ============================================================
# MAIN
# ============================================================


def main():

    print("=" * 70)
    print("HARDTEC - AIOps Incident Risk Prediction")
    print("=" * 70)

    # --------------------------------------------------------
    # Validate files
    # --------------------------------------------------------

    if not INPUT_INCIDENTS.exists():
        raise FileNotFoundError(
            f"Missing file:\n{INPUT_INCIDENTS}"
        )

    if not INPUT_ANOMALIES.exists():
        raise FileNotFoundError(
            f"Missing file:\n{INPUT_ANOMALIES}"
        )

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    incidents = pd.read_csv(INPUT_INCIDENTS)

    anomalies = pd.read_csv(INPUT_ANOMALIES)

    print()
    print("Input files loaded")
    print(f"Incident file rows : {len(incidents)}")
    print(f"Anomaly file rows  : {len(anomalies)}")

    # --------------------------------------------------------
    # Use enriched incident file as primary source
    # --------------------------------------------------------

    df = incidents.copy()

    # --------------------------------------------------------
    # Parse timestamp
    # --------------------------------------------------------

    if "window_start" not in df.columns:
        raise ValueError(
            "Column 'window_start' is missing from aiops_incidents.csv"
        )

    df["window_start"] = pd.to_datetime(
        df["window_start"],
        errors="coerce",
        utc=True,
    )

    df = df.dropna(
        subset=["window_start"]
    ).sort_values(
        "window_start"
    ).reset_index(drop=True)

    # --------------------------------------------------------
    # Parse incident timestamps
    # --------------------------------------------------------

    if "incident_start" in df.columns:

        df["incident_start"] = pd.to_datetime(
            df["incident_start"],
            errors="coerce",
            utc=True,
        )

    else:
        df["incident_start"] = pd.NaT

    if "incident_end" in df.columns:

        df["incident_end"] = pd.to_datetime(
            df["incident_end"],
            errors="coerce",
            utc=True,
        )

    else:
        df["incident_end"] = pd.NaT

    # --------------------------------------------------------
    # Incident IDs
    # --------------------------------------------------------

    if "incident_id" not in df.columns:
        df["incident_id"] = np.nan

    unique_incidents = (
        df.loc[
            df["incident_id"].notna(),
            "incident_id"
        ]
        .nunique()
    )

    # --------------------------------------------------------
    # Current incident status
    # --------------------------------------------------------

    df = mark_current_incidents(df)

    # --------------------------------------------------------
    # Future incident target
    # --------------------------------------------------------

    df = calculate_future_incident(
        df,
        PREDICTION_HORIZON_MINUTES,
    )

    # --------------------------------------------------------
    # Temporal features
    # --------------------------------------------------------

    df = create_temporal_features(df)

    # --------------------------------------------------------
    # Risk score
    # --------------------------------------------------------

    df = calculate_risk_score(df)

    # --------------------------------------------------------
    # Early-warning signal
    # --------------------------------------------------------

    df["early_warning"] = (
        (~df["incident_now"])
        & (df["future_incident"])
        & (df["risk_score"] >= 40)
    )

    df["high_risk_warning"] = (
        (~df["incident_now"])
        & (df["future_incident"])
        & (df["risk_score"] >= 60)
    )

    # --------------------------------------------------------
    # Ranking
    # --------------------------------------------------------

    df["risk_rank"] = (
        df["risk_score"]
        .rank(
            ascending=False,
            method="min",
        )
        .astype(int)
    )

    # --------------------------------------------------------
    # Select useful output columns
    # --------------------------------------------------------

    output_columns = [
        "window_start",

        # Current operational signals
        "event_count",
        "raw_row_count",
        "avg_duration_ms",
        "max_duration_ms",
        "p95_duration_ms",
        "high_severity_rate",
        "error_density",

        # Anomaly detection
        "anomaly_score",
        "is_anomaly",
        "anomaly_level",
        "anomaly_reason",

        # Incident context
        "incident_id",
        "incident_now",
        "future_incident",

        # Temporal features
        "event_count_delta",
        "avg_latency_delta",
        "p95_latency_delta",
        "error_density_delta",
        "anomaly_persistence_3",

        # Risk
        "risk_score",
        "risk_level",
        "risk_rank",
        "risk_reason",

        # Early warning
        "early_warning",
        "high_risk_warning",
    ]

    # Keep only columns that actually exist.
    output_columns = [
        column
        for column in output_columns
        if column in df.columns
    ]

    result = df[output_columns].copy()

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    result.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    # ========================================================
    # REPORT
    # ========================================================

    future_count = int(
        result["future_incident"].sum()
    )

    current_count = int(
        result["incident_now"].sum()
    )

    early_warning_count = int(
        result["early_warning"].sum()
    )

    high_risk_count = int(
        result["high_risk_warning"].sum()
    )

    critical_count = int(
        (result["risk_level"] == "CRITICAL").sum()
    )

    high_count = int(
        (result["risk_level"] == "HIGH").sum()
    )

    medium_count = int(
        (result["risk_level"] == "MEDIUM").sum()
    )

    low_count = int(
        (result["risk_level"] == "LOW").sum()
    )

    print()
    print("=" * 70)
    print("DATASET / INCIDENT SUMMARY")
    print("=" * 70)

    print(
        f"Windows loaded          : {len(result)}"
    )

    print(
        f"Incident episodes       : {unique_incidents}"
    )

    print(
        f"Windows currently in incident : {current_count}"
    )

    print(
        f"Future-incident windows : {future_count}"
    )

    print(
        f"Prediction horizon      : "
        f"{PREDICTION_HORIZON_MINUTES} real minutes"
    )

    print()
    print("=" * 70)
    print("RISK ANALYSIS")
    print("=" * 70)

    print(
        f"CRITICAL                : {critical_count}"
    )

    print(
        f"HIGH                    : {high_count}"
    )

    print(
        f"MEDIUM                  : {medium_count}"
    )

    print(
        f"LOW                     : {low_count}"
    )

    print(
        f"High-risk warnings      : {high_risk_count}"
    )

    print(
        f"Early-warning matches   : "
        f"{early_warning_count}"
    )

    if future_count > 0:

        early_warning_rate = (
            early_warning_count
            / future_count
            * 100
        )

        print(
            f"Early-warning rate      : "
            f"{early_warning_rate:.2f}%"
        )

    else:

        print(
            "Early-warning rate      : "
            "N/A"
        )

    # --------------------------------------------------------
    # Top risk windows
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("TOP RISK WINDOWS")
    print("=" * 70)

    display_columns = [
        "window_start",
        "risk_score",
        "risk_level",
        "future_incident",
        "incident_now",
        "is_anomaly",
        "event_count",
        "p95_duration_ms",
        "error_density",
        "risk_reason",
    ]

    top_risk = (
        result[
            display_columns
        ]
        .sort_values(
            "risk_score",
            ascending=False,
        )
        .head(10)
    )

    print(
        top_risk.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Output
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("OUTPUT")
    print("=" * 70)

    print(
        "Saved to:"
    )

    print(
        OUTPUT_FILE
    )

    print()
    print(
        "Methodology:"
    )

    print(
        "- Isolation Forest anomalies are used as operational signals."
    )

    print(
        "- Incident episodes are correlated from temporal signals."
    )

    print(
        "- Future incidents are defined using real timestamps."
    )

    print(
        "- Risk is calculated using current and historical signals only."
    )

    print(
        "- This is an AIOps POC, not a production-grade "
        "supervised incident classifier."
    )

    print(
        "- The dataset contains only 41 observed temporal windows."
    )

    print("=" * 70)


if __name__ == "__main__":
    main()

