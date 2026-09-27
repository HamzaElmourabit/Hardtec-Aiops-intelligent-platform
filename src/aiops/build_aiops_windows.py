"""
HARDTEC - AIOps Window Builder

Builds 1-minute temporal windows from raw observability events.

Purpose:
    Prepare operational features for AIOps anomaly detection.

Important:
    failure_count and failure_rate are observed operational signals.
    They are NOT ground-truth incident labels.
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

OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"

OUTPUT_FILE = OUTPUT_DIR / "aiops_windows.csv"

WINDOW_SIZE = "1min"


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    print("=" * 70)
    print("HARDTEC - AIOps Window Builder")
    print("=" * 70)

    print("\n[1/7] Loading raw dataset...")
    print(f"Input: {INPUT_FILE}")

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"\nERROR: Input file not found:\n{INPUT_FILE}\n"
        )

    df = pd.read_csv(INPUT_FILE)

    print(f"Rows    : {len(df):,}")
    print(f"Columns : {len(df.columns)}")

    return df


# ============================================================
# VALIDATION
# ============================================================

def validate_data(df):

    print("\n[2/7] Validating dataset...")

    required_columns = [
        "timestamp",
        "event_type",
        "category",
        "severity",
        "status",
        "duration_ms",
        "path",
        "status_code",
        "level",
    ]

    missing_columns = [
        col
        for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "Missing required columns:\n"
            + "\n".join(
                f" - {col}"
                for col in missing_columns
            )
        )

    # --------------------------------------------------------
    # Timestamp
    # --------------------------------------------------------

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        errors="coerce",
        utc=True,
    )

    invalid_timestamps = (
        df["timestamp"]
        .isna()
        .sum()
    )

    if invalid_timestamps > 0:

        print(
            f"WARNING: removing "
            f"{invalid_timestamps} invalid timestamps."
        )

        df = df.dropna(
            subset=["timestamp"]
        ).copy()

    # --------------------------------------------------------
    # Numeric
    # --------------------------------------------------------

    df["duration_ms"] = pd.to_numeric(
        df["duration_ms"],
        errors="coerce",
    )

    if "event_count" in df.columns:

        df["event_count"] = pd.to_numeric(
            df["event_count"],
            errors="coerce",
        ).fillna(1)

    else:

        df["event_count"] = 1

    # --------------------------------------------------------
    # Categorical
    # --------------------------------------------------------

    categorical_columns = [
        "event_type",
        "category",
        "severity",
        "status",
        "level",
        "method",
        "path",
    ]

    for column in categorical_columns:

        if column in df.columns:

            df[column] = (
                df[column]
                .astype("string")
                .fillna("")
            )

    print("Timestamp conversion : OK")
    print("Required columns     : OK")
    print("Numeric conversion   : OK")
    print("Missing values       : handled")

    return df


# ============================================================
# SIGNAL CREATION
# ============================================================

def create_signals(df):

    print("\n[3/7] Creating AIOps operational signals...")

    failure_event_types = {
        "SERVER_ERROR",
        "ROUTING_ERROR",
        "HTTP_EXCEPTION",
        "DB_ERROR",
        "HTTP_ERROR",
    }

    # --------------------------------------------------------
    # Failure signal
    # --------------------------------------------------------

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
    ).fillna(False).astype("int8")

    # --------------------------------------------------------
    # Severity
    # --------------------------------------------------------

    df["high_severity_signal"] = (
        df["severity"]
        .eq("HIGH")
        .fillna(False)
        .astype("int8")
    )

    # --------------------------------------------------------
    # Event types
    # --------------------------------------------------------

    df["server_error_signal"] = (
        df["event_type"]
        .eq("SERVER_ERROR")
        .fillna(False)
        .astype("int8")
    )

    df["routing_error_signal"] = (
        df["event_type"]
        .eq("ROUTING_ERROR")
        .fillna(False)
        .astype("int8")
    )

    df["db_error_signal"] = (
        df["event_type"]
        .eq("DB_ERROR")
        .fillna(False)
        .astype("int8")
    )

    df["http_error_signal"] = (
        df["event_type"]
        .isin([
            "HTTP_ERROR",
            "HTTP_EXCEPTION",
        ])
        .fillna(False)
        .astype("int8")
    )

    df["http_request_signal"] = (
        df["event_type"]
        .eq("INCOMING_REQUEST")
        .fillna(False)
        .astype("int8")
    )

    df["db_query_signal"] = (
        df["event_type"]
        .eq("DB_QUERY")
        .fillna(False)
        .astype("int8")
    )

    # --------------------------------------------------------
    # Latency
    # --------------------------------------------------------

    df["slow_100ms"] = (
        df["duration_ms"]
        .gt(100)
        .fillna(False)
        .astype("int8")
    )

    df["slow_500ms"] = (
        df["duration_ms"]
        .gt(500)
        .fillna(False)
        .astype("int8")
    )

    df["slow_1000ms"] = (
        df["duration_ms"]
        .gt(1000)
        .fillna(False)
        .astype("int8")
    )

    print("Failure signals    : OK")
    print("Severity signals   : OK")
    print("Event signals      : OK")
    print("Latency signals    : OK")

    return df


# ============================================================
# TEMPORAL AGGREGATION
# ============================================================

def create_window_features(df):

    print(
        f"\n[4/7] Creating "
        f"{WINDOW_SIZE} temporal windows..."
    )

    df = (
        df
        .sort_values("timestamp")
        .set_index("timestamp")
    )

    grouped = df.resample(
        WINDOW_SIZE
    )

    windows = grouped.agg(

        # ----------------------------------------------------
        # Volume
        # ----------------------------------------------------

        event_count=(
            "event_count",
            "sum",
        ),

        raw_row_count=(
            "event_type",
            "size",
        ),

        # ----------------------------------------------------
        # Failures
        # ----------------------------------------------------

        failure_count=(
            "failure_signal",
            "sum",
        ),

        high_severity_count=(
            "high_severity_signal",
            "sum",
        ),

        server_error_count=(
            "server_error_signal",
            "sum",
        ),

        routing_error_count=(
            "routing_error_signal",
            "sum",
        ),

        db_error_count=(
            "db_error_signal",
            "sum",
        ),

        http_error_count=(
            "http_error_signal",
            "sum",
        ),

        # ----------------------------------------------------
        # Traffic
        # ----------------------------------------------------

        http_request_count=(
            "http_request_signal",
            "sum",
        ),

        db_query_count=(
            "db_query_signal",
            "sum",
        ),

        # ----------------------------------------------------
        # Latency
        # ----------------------------------------------------

        slow_100ms_count=(
            "slow_100ms",
            "sum",
        ),

        slow_500ms_count=(
            "slow_500ms",
            "sum",
        ),

        slow_1000ms_count=(
            "slow_1000ms",
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

    # --------------------------------------------------------
    # P95 latency
    # --------------------------------------------------------

    windows["p95_duration_ms"] = (
        grouped["duration_ms"]
        .quantile(0.95)
    )

    # --------------------------------------------------------
    # Diversity
    # --------------------------------------------------------

    windows["unique_paths"] = (
        grouped["path"]
        .nunique()
    )

    windows["unique_event_types"] = (
        grouped["event_type"]
        .nunique()
    )

    windows["unique_categories"] = (
        grouped["category"]
        .nunique()
    )

    windows["unique_severities"] = (
        grouped["severity"]
        .nunique()
    )

    # --------------------------------------------------------
    # Remove empty windows
    # --------------------------------------------------------

    windows = windows[
        windows["raw_row_count"] > 0
    ].copy()

    # --------------------------------------------------------
    # Rates
    # --------------------------------------------------------

    windows["failure_rate"] = np.where(
        windows["raw_row_count"] > 0,
        windows["failure_count"]
        / windows["raw_row_count"],
        0.0,
    )

    windows["high_severity_rate"] = np.where(
        windows["raw_row_count"] > 0,
        windows["high_severity_count"]
        / windows["raw_row_count"],
        0.0,
    )

    total_error_events = (
        windows["server_error_count"]
        + windows["routing_error_count"]
        + windows["db_error_count"]
        + windows["http_error_count"]
    )

    windows["error_density"] = np.where(
        windows["raw_row_count"] > 0,
        total_error_events
        / windows["raw_row_count"],
        0.0,
    )

    # --------------------------------------------------------
    # Temporal features
    # --------------------------------------------------------

    windows = windows.reset_index()

    windows = windows.rename(
        columns={
            "timestamp": "window_start"
        }
    )

    windows["hour"] = (
        windows["window_start"]
        .dt.hour
    )

    windows["minute"] = (
        windows["window_start"]
        .dt.minute
    )

    windows["day_of_week"] = (
        windows["window_start"]
        .dt.dayofweek
    )

    windows["day_name"] = (
        windows["window_start"]
        .dt.day_name()
    )

    # --------------------------------------------------------
    # Clean numeric values
    # --------------------------------------------------------

    numeric_columns = (
        windows
        .select_dtypes(
            include=np.number
        )
        .columns
    )

    windows[numeric_columns] = (
        windows[numeric_columns]
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
        .fillna(0)
    )

    windows = (
        windows
        .sort_values("window_start")
        .reset_index(drop=True)
    )

    return windows


# ============================================================
# QUALITY REPORT
# ============================================================

def print_quality_report(windows):

    print("\n[5/7] AIOps window quality report")
    print("-" * 70)

    if windows.empty:

        print(
            "ERROR: No non-empty windows generated."
        )

        return

    print(
        f"Number of 1-minute windows : "
        f"{len(windows):,}"
    )

    print(
        f"First window               : "
        f"{windows['window_start'].min()}"
    )

    print(
        f"Last window                : "
        f"{windows['window_start'].max()}"
    )

    print("\nEvents")

    print(
        f"Total events               : "
        f"{windows['event_count'].sum():,.0f}"
    )

    print(
        f"Total raw rows             : "
        f"{windows['raw_row_count'].sum():,.0f}"
    )

    print(
        f"Failure signals            : "
        f"{windows['failure_count'].sum():,.0f}"
    )

    print("\nWindow distribution")

    print(
        f"Windows with failures      : "
        f"{(windows['failure_count'] > 0).sum():,}"
    )

    print(
        f"Windows without failures   : "
        f"{(windows['failure_count'] == 0).sum():,}"
    )

    print(
        f"Maximum events/window      : "
        f"{windows['event_count'].max():.0f}"
    )

    print(
        f"Mean events/window         : "
        f"{windows['event_count'].mean():.2f}"
    )

    print(
        f"Median events/window       : "
        f"{windows['event_count'].median():.2f}"
    )

    print(
        f"Maximum failure rate       : "
        f"{windows['failure_rate'].max():.2%}"
    )

    print("\nLatency")

    print(
        f"Mean average latency       : "
        f"{windows['avg_duration_ms'].mean():.2f} ms"
    )

    print(
        f"Maximum latency            : "
        f"{windows['max_duration_ms'].max():.2f} ms"
    )

    print(
        f"Maximum P95 latency        : "
        f"{windows['p95_duration_ms'].max():.2f} ms"
    )


# ============================================================
# PREVIEW
# ============================================================

def print_sample(windows):

    print("\n[6/7] Preview of generated windows")
    print("-" * 70)

    columns = [
        "window_start",
        "event_count",
        "raw_row_count",
        "failure_count",
        "failure_rate",
        "server_error_count",
        "routing_error_count",
        "db_error_count",
        "http_error_count",
        "high_severity_count",
        "avg_duration_ms",
        "max_duration_ms",
        "p95_duration_ms",
        "unique_paths",
        "unique_event_types",
    ]

    columns = [
        col
        for col in columns
        if col in windows.columns
    ]

    print(
        windows[
            columns
        ]
        .head(20)
        .to_string(index=False)
    )


# ============================================================
# SAVE
# ============================================================

def save_output(windows):

    print("\n[7/7] Saving AIOps windows...")

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    windows.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print(
        f"Output saved to:\n"
        f"{OUTPUT_FILE}"
    )

    print(
        f"\nOutput rows    : "
        f"{len(windows):,}"
    )

    print(
        f"Output columns : "
        f"{len(windows.columns)}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    df = load_data()

    df = validate_data(df)

    df = create_signals(df)

    windows = create_window_features(df)

    print_quality_report(
        windows
    )

    print_sample(
        windows
    )

    save_output(
        windows
    )

    print("\n" + "=" * 70)
    print(
        "AIOps feature engineering "
        "completed successfully."
    )
    print("=" * 70)

    print(
        "\nNext step:"
    )

    print(
        "Build the AIOps anomaly detection baseline."
    )

    print(
        "\nImportant:"
    )

    print(
        "failure_count and failure_rate are "
        "observed operational signals."
    )

    print(
        "They are NOT ground-truth incident labels."
    )


if __name__ == "__main__":
    main()