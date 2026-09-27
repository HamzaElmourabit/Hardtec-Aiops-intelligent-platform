"""
HARDTEC - AIOps Window Granularity Analysis

Tests several temporal granularities before selecting
the final window size for anomaly detection.
"""

from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "processed_events_analysis.csv"
)

WINDOW_SIZES = [
    "1min",
    "2min",
    "5min",
    "10min",
    "15min",
]


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    print("=" * 75)
    print("HARDTEC - AIOps Window Granularity Analysis")
    print("=" * 75)

    print("\n[1/4] Loading dataset...")
    print(f"Input: {INPUT_FILE}")

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found:\n{INPUT_FILE}"
        )

    df = pd.read_csv(INPUT_FILE)

    print(f"Rows    : {len(df):,}")
    print(f"Columns : {len(df.columns)}")

    return df


# ============================================================
# PREPARE SIGNALS
# ============================================================

def prepare_data(df):

    print("\n[2/4] Preparing temporal signals...")

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        errors="coerce",
        utc=True,
    )

    df = df.dropna(
        subset=["timestamp"]
    ).copy()

    # Normalize categorical columns
    for column in [
        "event_type",
        "category",
        "severity",
        "status",
        "level",
        "path",
    ]:

        if column in df.columns:

            df[column] = (
                df[column]
                .astype("string")
                .fillna("")
            )

    df["duration_ms"] = pd.to_numeric(
        df["duration_ms"],
        errors="coerce",
    )

    # --------------------------------------------------------
    # Failure signal
    # --------------------------------------------------------

    failure_event_types = {
        "SERVER_ERROR",
        "ROUTING_ERROR",
        "HTTP_EXCEPTION",
        "DB_ERROR",
        "HTTP_ERROR",
    }

    df["failure_signal"] = (
        (
            df["status"]
            .eq("FAILURE")
        )
        |
        (
            df["event_type"]
            .isin(failure_event_types)
        )
        |
        (
            df["level"]
            .eq("ERROR")
        )
    ).fillna(False).astype(int)

    # --------------------------------------------------------
    # High severity
    # --------------------------------------------------------

    df["high_severity_signal"] = (
        df["severity"]
        .eq("HIGH")
        .fillna(False)
        .astype(int)
    )

    # --------------------------------------------------------
    # Error event types
    # --------------------------------------------------------

    df["server_error"] = (
        df["event_type"]
        .eq("SERVER_ERROR")
        .fillna(False)
        .astype(int)
    )

    df["routing_error"] = (
        df["event_type"]
        .eq("ROUTING_ERROR")
        .fillna(False)
        .astype(int)
    )

    df["db_error"] = (
        df["event_type"]
        .eq("DB_ERROR")
        .fillna(False)
        .astype(int)
    )

    df["http_error"] = (
        df["event_type"]
        .isin([
            "HTTP_ERROR",
            "HTTP_EXCEPTION",
        ])
        .fillna(False)
        .astype(int)
    )

    return df


# ============================================================
# ANALYZE ONE GRANULARITY
# ============================================================

def analyze_granularity(df, window_size):

    temp = (
        df
        .sort_values("timestamp")
        .set_index("timestamp")
    )

    grouped = temp.resample(window_size)

    windows = grouped.agg(
        raw_rows=("event_type", "size"),

        event_count=("event_count", "sum"),

        failure_count=("failure_signal", "sum"),

        high_severity_count=(
            "high_severity_signal",
            "sum",
        ),

        server_error_count=(
            "server_error",
            "sum",
        ),

        routing_error_count=(
            "routing_error",
            "sum",
        ),

        db_error_count=(
            "db_error",
            "sum",
        ),

        http_error_count=(
            "http_error",
            "sum",
        ),

        avg_duration_ms=(
            "duration_ms",
            "mean",
        ),

        max_duration_ms=(
            "duration_ms",
            "max",
        ),
    )

    # Keep only non-empty windows
    windows = windows[
        windows["raw_rows"] > 0
    ].copy()

    if windows.empty:

        return {
            "window_size": window_size,
            "windows": 0,
            "windows_without_failure": 0,
            "windows_with_failure": 0,
            "failure_window_rate": 0,
            "mean_events": 0,
            "median_events": 0,
            "max_events": 0,
            "mean_failure_rate": 0,
            "max_failure_rate": 0,
            "mean_latency": 0,
            "max_latency": 0,
        }

    # Failure rate
    windows["failure_rate"] = (
        windows["failure_count"]
        / windows["raw_rows"]
    )

    return {
        "window_size": window_size,

        "windows": len(windows),

        "windows_without_failure": (
            windows["failure_count"] == 0
        ).sum(),

        "windows_with_failure": (
            windows["failure_count"] > 0
        ).sum(),

        "failure_window_rate": (
            windows["failure_count"] > 0
        ).mean(),

        "mean_events": (
            windows["event_count"]
            .mean()
        ),

        "median_events": (
            windows["event_count"]
            .median()
        ),

        "max_events": (
            windows["event_count"]
            .max()
        ),

        "mean_failure_rate": (
            windows["failure_rate"]
            .mean()
        ),

        "max_failure_rate": (
            windows["failure_rate"]
            .max()
        ),

        "mean_latency": (
            windows["avg_duration_ms"]
            .mean()
        ),

        "max_latency": (
            windows["max_duration_ms"]
            .max()
        ),
    }


# ============================================================
# DISPLAY RESULTS
# ============================================================

def display_results(results):

    print("\n[3/4] Comparing temporal granularities")
    print("-" * 75)

    result_df = pd.DataFrame(results)

    display_df = result_df.copy()

    display_df["failure_window_rate"] = (
        display_df["failure_window_rate"]
        .map(lambda x: f"{x:.2%}")
    )

    display_df["mean_failure_rate"] = (
        display_df["mean_failure_rate"]
        .map(lambda x: f"{x:.2%}")
    )

    display_df["max_failure_rate"] = (
        display_df["max_failure_rate"]
        .map(lambda x: f"{x:.2%}")
    )

    display_df["mean_events"] = (
        display_df["mean_events"]
        .round(2)
    )

    display_df["median_events"] = (
        display_df["median_events"]
        .round(2)
    )

    display_df["mean_latency"] = (
        display_df["mean_latency"]
        .round(2)
    )

    display_df["max_latency"] = (
        display_df["max_latency"]
        .round(2)
    )

    print(
        display_df.to_string(
            index=False
        )
    )

    return result_df


# ============================================================
# RECOMMENDATION
# ============================================================

def recommend_window(result_df):

    print("\n[4/4] Preliminary recommendation")
    print("-" * 75)

    print(
        "\nCriteria:"
    )

    print(
        "1. Sufficient number of temporal windows"
    )

    print(
        "2. Presence of both normal and failure windows"
    )

    print(
        "3. Reasonable event distribution"
    )

    print(
        "4. Enough temporal resolution for AIOps"
    )

    # --------------------------------------------------------
    # Candidate selection
    # --------------------------------------------------------

    candidates = result_df[
        (
            result_df["windows"] >= 20
        )
        &
        (
            result_df["windows_without_failure"] > 0
        )
        &
        (
            result_df["windows_with_failure"] > 0
        )
    ].copy()

    if candidates.empty:

        print(
            "\nWARNING:"
            "\nNo granularity satisfies all criteria."
        )

        print(
            "\nWe will NOT automatically select a model."
        )

        print(
            "The dataset may require a different "
            "temporal strategy."
        )

        return

    # Prefer finer temporal resolution,
    # but avoid extremely sparse windows.

    candidates["score"] = (
        candidates["windows_without_failure"]
        * 2
        +
        candidates["windows_with_failure"]
    )

    candidates = candidates.sort_values(
        [
            "score",
            "windows",
        ],
        ascending=False,
    )

    best = candidates.iloc[0]

    print(
        f"\nCandidate recommended: "
        f"{best['window_size']}"
    )

    print(
        f"Number of windows: "
        f"{int(best['windows'])}"
    )

    print(
        f"Windows without failure: "
        f"{int(best['windows_without_failure'])}"
    )

    print(
        f"Windows with failure: "
        f"{int(best['windows_with_failure'])}"
    )

    print(
        f"Failure-window rate: "
        f"{best['failure_window_rate']:.2%}"
    )

    print(
        "\nIMPORTANT:"
        "\nThis is a preliminary recommendation."
        "\nThe final window size will be validated "
        "during anomaly detection."
    )


# ============================================================
# MAIN
# ============================================================

def main():

    df = load_data()

    df = prepare_data(df)

    results = []

    for window_size in WINDOW_SIZES:

        result = analyze_granularity(
            df,
            window_size,
        )

        results.append(result)

    result_df = display_results(
        results
    )

    recommend_window(
        result_df
    )

    print("\n" + "=" * 75)
    print("Granularity analysis completed.")
    print("=" * 75)


if __name__ == "__main__":
    main()