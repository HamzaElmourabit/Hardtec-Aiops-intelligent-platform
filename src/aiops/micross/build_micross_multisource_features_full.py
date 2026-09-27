
from pathlib import Path
import re
import sys

import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

GAIA_ROOT = Path(r"C:\Users\khadi\Desktop\GAIA-DataSet")
HARDTEC_ROOT = Path(r"C:\Users\khadi\Desktop\hardtec-intelligent-ticketing")

INPUT_DIR = (
    GAIA_ROOT
    / "MicroSS"
    / "metric_candidates_full"
)

OUTPUT_DIR = (
    HARDTEC_ROOT
    / "data"
    / "processed"
    / "micross"
)

OUTPUT_FILE = (
    OUTPUT_DIR
    / "micross_multisource_features_5m.csv"
)

SUMMARY_FILE = (
    OUTPUT_DIR
    / "micross_multisource_features_5m_summary.csv"
)

FREQUENCY = "5min"

# ============================================================
# OPTIONS
# ============================================================

# For very large CSVs, pandas can process in chunks.
CHUNK_SIZE = 200_000

# Minimum proportion of valid timestamps required
MIN_TIMESTAMP_VALID_RATIO = 0.80

# Minimum proportion of valid numeric values required
MIN_VALUE_VALID_RATIO = 0.50

# ============================================================
# HELPER FUNCTIONS
# ============================================================


def log(message: str) -> None:
    """Simple console logger."""
    print(f"[INFO] {message}", flush=True)


def warn(message: str) -> None:
    """Simple warning logger."""
    print(f"[WARNING] {message}", flush=True)


def normalize_column_name(name: str) -> str:
    """
    Normalize a column name for robust detection.
    """
    return (
        str(name)
        .strip()
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
    )


def detect_timestamp_column(columns) -> str | None:
    """
    Detect timestamp column.
    """
    normalized = {
        normalize_column_name(col): col
        for col in columns
    }

    candidates = [
        "timestamp",
        "time",
        "datetime",
        "date",
        "ts",
    ]

    for candidate in candidates:
        if candidate in normalized:
            return normalized[candidate]

    return None


def detect_value_column(columns) -> str | None:
    """
    Detect the metric value column.
    """
    normalized = {
        normalize_column_name(col): col
        for col in columns
    }

    candidates = [
        "value",
        "values",
        "metric_value",
        "metric",
    ]

    for candidate in candidates:
        if candidate in normalized:
            return normalized[candidate]

    # Fallback:
    # choose a column containing "value"
    for normalized_name, original_name in normalized.items():
        if "value" in normalized_name:
            return original_name

    return None


def convert_gaia_timestamp(series: pd.Series) -> pd.Series:
    """
    Convert GAIA/MicroSS timestamps.

    GAIA raw metric timestamps are Unix epoch milliseconds,
    e.g. 1627747200000 -> 2021-08-01 00:00:00.

    The function also supports timestamps that are already
    datetime strings.
    """

    numeric = pd.to_numeric(series, errors="coerce")

    numeric_ratio = numeric.notna().mean()

    if numeric_ratio >= MIN_TIMESTAMP_VALID_RATIO:
        result = pd.to_datetime(
            numeric,
            unit="ms",
            errors="coerce",
        )
    else:
        result = pd.to_datetime(
            series,
            errors="coerce",
        )

    return result


def sanitize_metric_name(name: str) -> str:
    """
    Convert a logical metric filename into a safe feature prefix.
    """

    name = Path(name).stem

    name = name.replace(
        "_2021-07-01_2021-07-15",
        "",
    )

    name = name.replace(
        "_2021-07-15_2021-07-31",
        "",
    )

    name = name.replace(
        "_2021-08-01_2021-08-31",
        "",
    )

    name = re.sub(
        r"[^a-zA-Z0-9_]+",
        "_",
        name,
    )

    name = re.sub(
        r"_+",
        "_",
        name,
    )

    return name.strip("_").lower()


def read_metric_file(file_path: Path) -> pd.DataFrame:
    """
    Read one metric CSV and return:

        timestamp
        value
    """

    try:
        df = pd.read_csv(
            file_path,
            low_memory=False,
        )
    except Exception as exc:
        raise RuntimeError(
            f"Unable to read {file_path}: {exc}"
        )

    if df.empty:
        raise ValueError(
            f"Empty CSV: {file_path.name}"
        )

    timestamp_col = detect_timestamp_column(
        df.columns
    )

    value_col = detect_value_column(
        df.columns
    )

    if timestamp_col is None:
        raise ValueError(
            f"No timestamp column found in {file_path.name}"
        )

    if value_col is None:
        raise ValueError(
            f"No value column found in {file_path.name}"
        )

    result = pd.DataFrame()

    result["timestamp"] = convert_gaia_timestamp(
        df[timestamp_col]
    )

    result["value"] = pd.to_numeric(
        df[value_col],
        errors="coerce",
    )

    timestamp_valid_ratio = (
        result["timestamp"].notna().mean()
    )

    value_valid_ratio = (
        result["value"].notna().mean()
    )

    if timestamp_valid_ratio < MIN_TIMESTAMP_VALID_RATIO:
        raise ValueError(
            f"Too many invalid timestamps in "
            f"{file_path.name}: "
            f"{timestamp_valid_ratio:.2%}"
        )

    if value_valid_ratio < MIN_VALUE_VALID_RATIO:
        raise ValueError(
            f"Too many invalid values in "
            f"{file_path.name}: "
            f"{value_valid_ratio:.2%}"
        )

    result = result.dropna(
        subset=["timestamp"]
    )

    result = result.sort_values(
        "timestamp"
    )

    result = result.drop_duplicates(
        subset=["timestamp"],
        keep="last",
    )

    return result


def aggregate_metric(
    df: pd.DataFrame,
    metric_name: str,
) -> pd.DataFrame:
    """
    Aggregate one metric on a 5-minute grid.

    Features:
        mean
        std
        min
        max
        count

    These statistics preserve more information than
    a simple mean alone.
    """

    df = df.copy()

    df = df.set_index(
        "timestamp"
    )

    aggregated = (
        df["value"]
        .resample(FREQUENCY)
        .agg(
            [
                "mean",
                "std",
                "min",
                "max",
                "count",
            ]
        )
    )

    aggregated = aggregated.rename(
        columns={
            "mean": f"{metric_name}__mean",
            "std": f"{metric_name}__std",
            "min": f"{metric_name}__min",
            "max": f"{metric_name}__max",
            "count": f"{metric_name}__count",
        }
    )

    return aggregated


def build_global_time_grid(
    metric_frames: list[pd.DataFrame],
) -> pd.DatetimeIndex:
    """
    Build the global 5-minute time grid from all metrics.
    """

    if not metric_frames:
        raise RuntimeError(
            "No metric frames available."
        )

    starts = []
    ends = []

    for frame in metric_frames:
        if frame.empty:
            continue

        starts.append(
            frame.index.min()
        )

        ends.append(
            frame.index.max()
        )

    if not starts or not ends:
        raise RuntimeError(
            "Could not determine global time range."
        )

    global_start = min(starts)
    global_end = max(ends)

    global_start = global_start.floor(
        FREQUENCY
    )

    global_end = global_end.ceil(
        FREQUENCY
    )

    log(
        f"Global start: {global_start}"
    )

    log(
        f"Global end:   {global_end}"
    )

    grid = pd.date_range(
        start=global_start,
        end=global_end,
        freq=FREQUENCY,
    )

    log(
        f"5-minute grid points: {len(grid):,}"
    )

    return grid


# ============================================================
# MAIN
# ============================================================


def main() -> None:

    print("=" * 70)
    print(
        "MICROSS MULTISOURCE 5-MINUTE FEATURE BUILDER"
    )
    print("=" * 70)

    log(
        f"Input directory:\n{INPUT_DIR}"
    )

    log(
        f"Output directory:\n{OUTPUT_DIR}"
    )

    if not INPUT_DIR.exists():
        raise FileNotFoundError(
            f"Input directory does not exist:\n{INPUT_DIR}"
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # 1. Discover CSV files
    # --------------------------------------------------------

    csv_files = sorted(
        INPUT_DIR.glob("*.csv")
    )

    if not csv_files:
        raise FileNotFoundError(
            f"No CSV files found in:\n{INPUT_DIR}"
        )

    log(
        f"CSV files found: {len(csv_files):,}"
    )

    # --------------------------------------------------------
    # 2. Read and aggregate every metric
    # --------------------------------------------------------

    aggregated_frames = []

    summary_rows = []

    successful = 0
    failed = 0

    global_start = None
    global_end = None

    for idx, file_path in enumerate(
        csv_files,
        start=1,
    ):

        print(
            f"[{idx:03d}/{len(csv_files):03d}] "
            f"{file_path.name}",
            flush=True,
        )

        metric_name = sanitize_metric_name(
            file_path.name
        )

        try:

            df = read_metric_file(
                file_path
            )

            if df.empty:
                warn(
                    f"Empty after cleaning: "
                    f"{file_path.name}"
                )
                failed += 1
                continue

            metric_start = df[
                "timestamp"
            ].min()

            metric_end = df[
                "timestamp"
            ].max()

            if (
                global_start is None
                or metric_start < global_start
            ):
                global_start = metric_start

            if (
                global_end is None
                or metric_end > global_end
            ):
                global_end = metric_end

            aggregated = aggregate_metric(
                df,
                metric_name,
            )

            aggregated_frames.append(
                aggregated
            )

            summary_rows.append(
                {
                    "file": file_path.name,
                    "metric_name": metric_name,
                    "raw_rows": len(df),
                    "raw_start": metric_start,
                    "raw_end": metric_end,
                    "5m_rows": len(aggregated),
                    "non_null_mean": int(
                        aggregated[
                            f"{metric_name}__mean"
                        ].notna().sum()
                    ),
                }
            )

            successful += 1

        except Exception as exc:

            failed += 1

            warn(
                f"FAILED: {file_path.name}"
            )

            warn(
                str(exc)
            )

    # --------------------------------------------------------
    # 3. Check successful metrics
    # --------------------------------------------------------

    if not aggregated_frames:
        raise RuntimeError(
            "No metric could be processed."
        )

    log(
        f"Successfully processed: "
        f"{successful:,}"
    )

    log(
        f"Failed: {failed:,}"
    )

    # --------------------------------------------------------
    # 4. Build global grid
    # --------------------------------------------------------

    log(
        "Building global 5-minute grid..."
    )

    global_grid = build_global_time_grid(
        aggregated_frames
    )

    # --------------------------------------------------------
    # 5. Reindex every metric on global grid
    # --------------------------------------------------------

    log(
        "Aligning all metrics on the global grid..."
    )

    aligned_frames = []

    for frame in aggregated_frames:

        aligned = frame.reindex(
            global_grid
        )

        aligned_frames.append(
            aligned
        )

    # --------------------------------------------------------
    # 6. Merge all metrics
    # --------------------------------------------------------

    log(
        "Merging multisource metrics..."
    )

    features = pd.concat(
        aligned_frames,
        axis=1,
    )

    features.index.name = "timestamp"

    # --------------------------------------------------------
    # 7. Remove duplicate columns if any
    # --------------------------------------------------------

    duplicated_columns = (
        features.columns[
            features.columns.duplicated()
        ]
        .tolist()
    )

    if duplicated_columns:

        warn(
            f"Duplicate feature columns detected: "
            f"{len(duplicated_columns)}"
        )

        features = features.loc[
            :,
            ~features.columns.duplicated(),
        ]

    # --------------------------------------------------------
    # 8. Add global temporal features
    # --------------------------------------------------------

    log(
        "Adding temporal features..."
    )

    features["hour"] = (
        features.index.hour
    )

    features["day_of_week"] = (
        features.index.dayofweek
    )

    features["day_of_month"] = (
        features.index.day
    )

    features["is_weekend"] = (
        features.index.dayofweek >= 5
    ).astype(int)

    # --------------------------------------------------------
    # 9. Sort columns
    # --------------------------------------------------------

    features = features.sort_index(
        axis=1
    )

    # Make timestamp the first column
    features = features.reset_index()

    cols = features.columns.tolist()

    cols.remove("timestamp")

    features = features[
        ["timestamp"] + cols
    ]

    # --------------------------------------------------------
    # 10. Save feature matrix
    # --------------------------------------------------------

    log(
        "Saving multisource feature matrix..."
    )

    features.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    # --------------------------------------------------------
    # 11. Save summary
    # --------------------------------------------------------

    summary_df = pd.DataFrame(
        summary_rows
    )

    summary_df.to_csv(
        SUMMARY_FILE,
        index=False,
    )

    # --------------------------------------------------------
    # 12. Statistics
    # --------------------------------------------------------

    metric_feature_columns = [
        col
        for col in features.columns
        if "__" in col
    ]

    mean_columns = [
        col
        for col in features.columns
        if col.endswith("__mean")
    ]

    count_columns = [
        col
        for col in features.columns
        if col.endswith("__count")
    ]

    missing_mean = (
        features[mean_columns]
        .isna()
        .mean()
        .mean()
        if mean_columns
        else np.nan
    )

    # --------------------------------------------------------
    # 13. Final report
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print(
        "MULTISOURCE FEATURE BUILD COMPLETE"
    )
    print("=" * 70)

    print(
        f"Input CSV files:             "
        f"{len(csv_files):,}"
    )

    print(
        f"Successfully processed:      "
        f"{successful:,}"
    )

    print(
        f"Failed:                      "
        f"{failed:,}"
    )

    print(
        f"Feature rows:                "
        f"{len(features):,}"
    )

    print(
        f"Total feature columns:       "
        f"{len(features.columns):,}"
    )

    print(
        f"Metric feature columns:      "
        f"{len(metric_feature_columns):,}"
    )

    print(
        f"Mean metric columns:         "
        f"{len(mean_columns):,}"
    )

    print(
        f"Count metric columns:        "
        f"{len(count_columns):,}"
    )

    print(
        f"Average missingness:         "
        f"{missing_mean:.2%}"
    )

    print(
        f"Global start:                "
        f"{global_start}"
    )

    print(
        f"Global end:                  "
        f"{global_end}"
    )

    print()
    print(
        f"Feature file:\n{OUTPUT_FILE}"
    )

    print(
        f"Summary file:\n{SUMMARY_FILE}"
    )

    print("=" * 70)


if __name__ == "__main__":

    try:
        main()

    except KeyboardInterrupt:

        print(
            "\n[WARNING] Process interrupted by user."
        )

        sys.exit(1)

    except Exception as exc:

        print()
        print(
            "[ERROR] Script failed:"
        )

        print(
            str(exc)
        )

        sys.exit(1)

