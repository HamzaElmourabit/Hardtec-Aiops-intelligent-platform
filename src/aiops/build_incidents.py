"""
HARDTEC - AIOps Incident Correlation

Purpose
-------
Transform individual anomalous/severe monitoring windows into
temporally correlated incident episodes.

Important:
- An anomaly is NOT automatically a real-world incident.
- This script creates an operational incident signal for the POC.
- It does not claim to provide ground-truth incident labels.
"""

from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = PROJECT_ROOT / "data" / "processed" / "aiops_anomalies.csv"
OUTPUT_FILE = PROJECT_ROOT / "data" / "processed" / "aiops_incidents.csv"


# ============================================================
# CONFIGURATION
# ============================================================

# Quantile used to identify unusually severe operational behavior.
SEVERE_QUANTILE = 0.90

# Maximum gap between two incident-signal windows.
# Because our windows are 1 minute, a gap of 1 minute allows
# consecutive operational signals to belong to the same episode.
MAX_GAP_MINUTES = 1

# Minimum number of independent severe signals required when
# a window is not already detected by Isolation Forest.
MIN_SEVERE_SIGNALS = 2


# ============================================================
# LOAD DATA
# ============================================================

def load_data() -> pd.DataFrame:
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input file not found:\n{INPUT_FILE}\n\n"
            "Run detect_anomalies.py first."
        )

    df = pd.read_csv(INPUT_FILE)

    if "window_start" not in df.columns:
        raise ValueError("Column 'window_start' is missing.")

    df["window_start"] = pd.to_datetime(
        df["window_start"],
        utc=True,
        errors="coerce",
    )

    df = df.dropna(subset=["window_start"])
    df = df.sort_values("window_start").reset_index(drop=True)

    return df


# ============================================================
# COMPUTE SEVERITY THRESHOLDS
# ============================================================

def compute_thresholds(df: pd.DataFrame) -> dict:
    """
    Compute dataset-relative thresholds.

    These are operational thresholds for this POC,
    not external SLA thresholds.
    """

    features = [
        "event_count",
        "avg_duration_ms",
        "p95_duration_ms",
        "high_severity_rate",
        "error_density",
    ]

    thresholds = {}

    for feature in features:
        if feature not in df.columns:
            continue

        values = pd.to_numeric(df[feature], errors="coerce")

        thresholds[feature] = float(
            values.quantile(SEVERE_QUANTILE)
        )

    return thresholds


# ============================================================
# CREATE OPERATIONAL SEVERITY SIGNALS
# ============================================================

def create_incident_signals(
    df: pd.DataFrame,
    thresholds: dict,
) -> pd.DataFrame:

    df = df.copy()

    # --------------------------------------------------------
    # Isolation Forest anomaly
    # --------------------------------------------------------

    df["is_anomaly"] = (
        df["is_anomaly"]
        .fillna(False)
        .astype(bool)
    )

    # --------------------------------------------------------
    # Individual severe operational signals
    # --------------------------------------------------------

    signal_columns = []

    for feature, threshold in thresholds.items():

        signal_name = f"severe_{feature}"

        values = pd.to_numeric(
            df[feature],
            errors="coerce",
        ).fillna(0)

        df[signal_name] = values >= threshold

        signal_columns.append(signal_name)

    # Number of independent severe indicators
    df["severe_signal_count"] = (
        df[signal_columns]
        .sum(axis=1)
        if signal_columns
        else 0
    )

    # --------------------------------------------------------
    # Composite incident signal
    # --------------------------------------------------------
    #
    # A window becomes an incident signal when:
    #
    #   1. Isolation Forest detects an anomaly
    #
    # OR
    #
    #   2. At least two independent operational indicators
    #      are simultaneously extreme.
    #
    # This prevents a single high-volume or high-latency
    # observation from automatically becoming an incident.
    # --------------------------------------------------------

    df["incident_signal"] = (
        df["is_anomaly"]
        |
        (
            df["severe_signal_count"]
            >= MIN_SEVERE_SIGNALS
        )
    )

    return df


# ============================================================
# GROUP INCIDENT EPISODES
# ============================================================

def group_incident_episodes(df: pd.DataFrame) -> pd.DataFrame:

    df = df.copy()

    df["incident_id"] = np.nan

    current_incident = 0
    previous_signal_time = None

    for idx, row in df.iterrows():

        if not row["incident_signal"]:
            continue

        current_time = row["window_start"]

        # First incident signal
        if previous_signal_time is None:
            current_incident += 1

        else:
            gap_minutes = (
                current_time - previous_signal_time
            ).total_seconds() / 60.0

            # New incident if the gap is too large
            if gap_minutes > MAX_GAP_MINUTES:
                current_incident += 1

        df.loc[idx, "incident_id"] = current_incident

        previous_signal_time = current_time

    # Convert to integer only for actual incidents
    df["incident_id"] = df["incident_id"].astype("Int64")

    return df


# ============================================================
# BUILD INCIDENT SUMMARY
# ============================================================

def build_incident_summary(df: pd.DataFrame) -> pd.DataFrame:

    incidents = df[
        df["incident_signal"]
    ].copy()

    if incidents.empty:
        return pd.DataFrame()

    summary = (
        incidents
        .groupby("incident_id", as_index=False)
        .agg(
            incident_start=("window_start", "min"),
            incident_end=("window_start", "max"),

            duration_windows=(
                "window_start",
                "count",
            ),

            max_anomaly_score=(
                "anomaly_score",
                "max",
            ),

            anomalous_windows=(
                "is_anomaly",
                "sum",
            ),

            max_event_count=(
                "event_count",
                "max",
            ),

            max_avg_duration_ms=(
                "avg_duration_ms",
                "max",
            ),

            max_p95_duration_ms=(
                "p95_duration_ms",
                "max",
            ),

            max_high_severity_rate=(
                "high_severity_rate",
                "max",
            ),

            max_error_density=(
                "error_density",
                "max",
            ),

            max_failure_rate=(
                "failure_rate",
                "max",
            ),

            max_severe_signal_count=(
                "severe_signal_count",
                "max",
            ),
        )
    )

    # Duration in minutes
    summary["duration_minutes"] = (
        (
            summary["incident_end"]
            - summary["incident_start"]
        ).dt.total_seconds() / 60.0
    ) + 1

    # --------------------------------------------------------
    # Incident severity
    # --------------------------------------------------------

    def classify_severity(row):

        if (
            row["max_anomaly_score"] >= 0.10
            or row["max_severe_signal_count"] >= 4
        ):
            return "CRITICAL"

        if (
            row["max_anomaly_score"] >= 0.02
            or row["max_severe_signal_count"] >= 3
        ):
            return "HIGH"

        if (
            row["max_severe_signal_count"] >= 2
            or row["anomalous_windows"] >= 1
        ):
            return "MEDIUM"

        return "LOW"

    summary["incident_severity"] = summary.apply(
        classify_severity,
        axis=1,
    )

    # --------------------------------------------------------
    # Incident type / dominant signal
    # --------------------------------------------------------

    def determine_type(row):

        if row["max_p95_duration_ms"] >= thresholds_global.get(
            "p95_duration_ms", float("inf")
        ):
            return "LATENCY_ANOMALY"

        if row["max_event_count"] >= thresholds_global.get(
            "event_count", float("inf")
        ):
            return "TRAFFIC_OR_VOLUME_ANOMALY"

        if row["max_error_density"] >= thresholds_global.get(
            "error_density", float("inf")
        ):
            return "ERROR_DENSITY_ANOMALY"

        if row["max_high_severity_rate"] >= thresholds_global.get(
            "high_severity_rate", float("inf")
        ):
            return "HIGH_SEVERITY_ANOMALY"

        return "MULTIVARIATE_ANOMALY"

    summary["incident_type"] = summary.apply(
        determine_type,
        axis=1,
    )

    # --------------------------------------------------------
    # Human-readable explanation
    # --------------------------------------------------------

    def build_reason(row):

        reasons = []

        if row["anomalous_windows"] > 0:
            reasons.append(
                f"{int(row['anomalous_windows'])} anomaly window(s)"
            )

        if (
            row["max_event_count"]
            >= thresholds_global.get(
                "event_count",
                float("inf"),
            )
        ):
            reasons.append("unusual event volume")

        if (
            row["max_avg_duration_ms"]
            >= thresholds_global.get(
                "avg_duration_ms",
                float("inf"),
            )
        ):
            reasons.append("high average latency")

        if (
            row["max_p95_duration_ms"]
            >= thresholds_global.get(
                "p95_duration_ms",
                float("inf"),
            )
        ):
            reasons.append("high P95 latency")

        if (
            row["max_error_density"]
            >= thresholds_global.get(
                "error_density",
                float("inf"),
            )
        ):
            reasons.append("high error density")

        if (
            row["max_high_severity_rate"]
            >= thresholds_global.get(
                "high_severity_rate",
                float("inf"),
            )
        ):
            reasons.append("high severity concentration")

        if not reasons:
            reasons.append(
                "multivariate operational anomaly"
            )

        return "; ".join(reasons)

    summary["incident_reason"] = summary.apply(
        build_reason,
        axis=1,
    )

    return summary


# ============================================================
# MAIN
# ============================================================

def main():

    global thresholds_global

    print("=" * 70)
    print("HARDTEC - AIOps Incident Correlation")
    print("=" * 70)

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    df = load_data()

    print(f"\nInput file      : {INPUT_FILE}")
    print(f"Total windows   : {len(df)}")

    # --------------------------------------------------------
    # Thresholds
    # --------------------------------------------------------

    thresholds_global = compute_thresholds(df)

    print("\nOperational thresholds")
    print("-" * 70)

    for feature, threshold in thresholds_global.items():
        print(
            f"{feature:<25}: {threshold:.4f}"
        )

    # --------------------------------------------------------
    # Signals
    # --------------------------------------------------------

    df = create_incident_signals(
        df,
        thresholds_global,
    )

    print("\nIncident signal generation")
    print("-" * 70)

    print(
        f"Isolation Forest anomalies : "
        f"{int(df['is_anomaly'].sum())}"
    )

    print(
        f"Incident-signal windows    : "
        f"{int(df['incident_signal'].sum())}"
    )

    # --------------------------------------------------------
    # Group episodes
    # --------------------------------------------------------

    df = group_incident_episodes(df)

    incident_windows = df[
        df["incident_signal"]
    ]

    incident_count = (
        incident_windows["incident_id"]
        .nunique()
    )

    print("\nIncident correlation")
    print("-" * 70)

    print(
        f"Incident episodes detected : "
        f"{incident_count}"
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    summary = build_incident_summary(df)

    if summary.empty:

        print("\nNo incident episodes detected.")

        # Still save the window-level dataset
        df.to_csv(
            OUTPUT_FILE,
            index=False,
        )

        return

    # --------------------------------------------------------
    # Merge incident summary information back into windows
    # --------------------------------------------------------

    df = df.merge(
        summary[
            [
                "incident_id",
                "incident_start",
                "incident_end",
                "duration_windows",
                "duration_minutes",
                "incident_severity",
                "incident_type",
                "incident_reason",
            ]
        ],
        on="incident_id",
        how="left",
        suffixes=("", "_summary"),
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    # --------------------------------------------------------
    # Display summary
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("INCIDENT EPISODES")
    print("=" * 70)

    display_columns = [
        "incident_id",
        "incident_start",
        "incident_end",
        "duration_minutes",
        "incident_severity",
        "incident_type",
        "incident_reason",
    ]

    print(
        summary[display_columns]
        .to_string(index=False)
    )

    print("\n" + "=" * 70)
    print("OUTPUT")
    print("=" * 70)

    print(f"\nSaved to:")
    print(OUTPUT_FILE)

    print("\nImportant:")
    print(
        "These incidents are operational POC signals, "
        "not ground-truth production incidents."
    )


if __name__ == "__main__":
    main()
