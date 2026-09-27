
from pathlib import Path
import re
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(r"C:\Users\khadi\Desktop\GAIA-DataSet")

INPUT_DIR = (
    PROJECT_ROOT
    / "MicroSS"
    / "metric_candidates_all_periods"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "MicroSS"
    / "metric_candidates_full"
)

REPORT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "micross"
)

MISSING_REPORT = REPORT_DIR / "micross_full_metrics_missing.csv"
SUMMARY_REPORT = REPORT_DIR / "micross_full_metrics_summary.csv"


# ============================================================
# PERIODS
# ============================================================

PERIOD_ORDER = [
    "2021-07-01_2021-07-15",
    "2021-07-15_2021-07-31",
    "2021-08-01_2021-08-31",
]


# ============================================================
# REGEX
# ============================================================

PERIOD_SUFFIX_RE = re.compile(
    r"_2021-\d{2}-\d{2}_2021-\d{2}-\d{2}$"
)


# ============================================================
# HELPERS
# ============================================================

def logical_metric_name(filename: str) -> str:
    """
    Converts a period-specific filename into a logical metric name.

    Example:
        service_x_metric_2021-07-01_2021-07-15.csv
    ->
        service_x_metric
    """

    name = Path(filename).stem

    name = PERIOD_SUFFIX_RE.sub("", name)

    return name


def detect_columns(df: pd.DataFrame):
    """
    Detect timestamp and value columns.
    """

    columns_lower = {
        str(col).strip().lower(): col
        for col in df.columns
    }

    timestamp_candidates = [
        "timestamp",
        "time",
        "datetime",
        "date",
        "ts",
    ]

    value_candidates = [
        "value",
        "metric_value",
        "val",
    ]

    timestamp_col = None
    value_col = None

    # --------------------------------------------------------
    # Detect timestamp column
    # --------------------------------------------------------

    for candidate in timestamp_candidates:

        if candidate in columns_lower:

            timestamp_col = columns_lower[candidate]

            break

    # --------------------------------------------------------
    # Detect value column
    # --------------------------------------------------------

    for candidate in value_candidates:

        if candidate in columns_lower:

            value_col = columns_lower[candidate]

            break

    # --------------------------------------------------------
    # Fallback timestamp
    # --------------------------------------------------------

    if timestamp_col is None:

        for col in df.columns:

            if "time" in str(col).lower():

                timestamp_col = col

                break

    # --------------------------------------------------------
    # Fallback numeric value
    # --------------------------------------------------------

    if value_col is None:

        numeric_candidates = []

        for col in df.columns:

            if col == timestamp_col:

                continue

            converted = pd.to_numeric(
                df[col],
                errors="coerce"
            )

            if converted.notna().sum() > 0:

                numeric_candidates.append(col)

        if numeric_candidates:

            value_col = numeric_candidates[0]

    return timestamp_col, value_col


def convert_gaia_timestamp(series: pd.Series) -> pd.Series:
    """
    Convert GAIA MicroSS timestamps into pandas datetime.

    GAIA MicroSS metric files use Unix epoch timestamps
    expressed in milliseconds.

    Example:

        1627747200000
        ->
        2021-08-01 00:00:00

    If timestamps are already datetime-like, they are preserved.
    """

    # --------------------------------------------------------
    # First try numeric conversion
    # --------------------------------------------------------

    numeric = pd.to_numeric(
        series,
        errors="coerce"
    )

    numeric_ratio = numeric.notna().mean()

    # --------------------------------------------------------
    # GAIA raw timestamps are Unix milliseconds
    # --------------------------------------------------------

    if numeric_ratio > 0.95:

        return pd.to_datetime(
            numeric,
            unit="ms",
            errors="coerce"
        )

    # --------------------------------------------------------
    # Fallback for already formatted datetime values
    # --------------------------------------------------------

    return pd.to_datetime(
        series,
        errors="coerce"
    )


def read_metric_file(path: Path):
    """
    Read one metric CSV and return standardized timestamp/value
    dataframe.

    Important:
    GAIA MicroSS timestamps are stored as Unix epoch milliseconds.
    """

    try:

        df = pd.read_csv(path)

    except Exception as exc:

        print(
            f"[ERROR] Cannot read: {path.name}"
        )

        print(
            f"        {exc}"
        )

        return None

    if df.empty:

        print(
            f"[WARNING] Empty file: {path.name}"
        )

        return None

    # --------------------------------------------------------
    # Detect columns
    # --------------------------------------------------------

    timestamp_col, value_col = detect_columns(df)

    if timestamp_col is None or value_col is None:

        print(
            f"[WARNING] Cannot detect columns: {path.name}"
        )

        print(
            f"          Columns: {list(df.columns)}"
        )

        return None

    # --------------------------------------------------------
    # Standardize
    # --------------------------------------------------------

    result = pd.DataFrame()

    # IMPORTANT:
    # GAIA timestamps = Unix epoch milliseconds
    result["timestamp"] = convert_gaia_timestamp(
        df[timestamp_col]
    )

    result["value"] = pd.to_numeric(
        df[value_col],
        errors="coerce"
    )

    # --------------------------------------------------------
    # Remove invalid rows
    # --------------------------------------------------------

    result = result.dropna(
        subset=[
            "timestamp",
            "value"
        ]
    )

    if result.empty:

        print(
            f"[WARNING] No valid rows: {path.name}"
        )

        return None

    # --------------------------------------------------------
    # Sort
    # --------------------------------------------------------

    result = result.sort_values(
        "timestamp"
    )

    return result


# ============================================================
# DISCOVER FILES
# ============================================================

def discover_files():

    print("=" * 70)
    print("DISCOVERING MICROSS METRIC FILES")
    print("=" * 70)

    if not INPUT_DIR.exists():

        raise FileNotFoundError(
            f"Input directory does not exist:\n{INPUT_DIR}"
        )

    files = sorted(
        INPUT_DIR.rglob("*.csv")
    )

    print(
        f"Input directory : {INPUT_DIR}"
    )

    print(
        f"CSV files found : {len(files)}"
    )

    return files


# ============================================================
# BUILD LOGICAL GROUPS
# ============================================================

def build_groups(files):

    groups = {}

    for path in files:

        logical_name = logical_metric_name(
            path.name
        )

        if logical_name not in groups:

            groups[logical_name] = []

        groups[logical_name].append(path)

    return groups


# ============================================================
# PERIOD DETECTION
# ============================================================

def detect_period(path: Path):

    name = path.stem

    match = re.search(
        r"(2021-\d{2}-\d{2}_2021-\d{2}-\d{2})$",
        name
    )

    if match:

        return match.group(1)

    # --------------------------------------------------------
    # Search parent directories
    # --------------------------------------------------------

    for part in path.parts:

        match = re.search(
            r"(2021-\d{2}-\d{2}_2021-\d{2}-\d{2})",
            part
        )

        if match:

            return match.group(1)

    return "unknown"


# ============================================================
# BUILD FULL SERIES
# ============================================================

def build_full_series(groups):

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    summary_rows = []

    missing_rows = []

    print()

    print("=" * 70)
    print("BUILDING FULL LOGICAL METRIC SERIES")
    print("=" * 70)

    total_groups = len(groups)

    print(
        f"Logical metrics found: {total_groups}"
    )

    for index, (logical_name, paths) in enumerate(
        sorted(groups.items()),
        start=1
    ):

        print()

        print(
            f"[{index}/{total_groups}] "
            f"{logical_name}"
        )

        # ----------------------------------------------------
        # Organize files by period
        # ----------------------------------------------------

        period_files = {}

        for path in paths:

            period = detect_period(path)

            if period not in period_files:

                period_files[period] = []

            period_files[period].append(path)

        frames = []

        periods_present = []

        total_input_files = 0

        # ----------------------------------------------------
        # Read periods in chronological order
        # ----------------------------------------------------

        for period in PERIOD_ORDER:

            current_files = period_files.get(
                period,
                []
            )

            if not current_files:

                missing_rows.append(
                    {
                        "logical_metric": logical_name,
                        "period": period,
                        "status": "missing",
                    }
                )

                continue

            periods_present.append(
                period
            )

            for path in current_files:

                total_input_files += 1

                df = read_metric_file(
                    path
                )

                if df is not None:

                    frames.append(df)

        # ----------------------------------------------------
        # No readable data
        # ----------------------------------------------------

        if not frames:

            print(
                "    -> skipped: no readable data"
            )

            summary_rows.append(
                {
                    "logical_metric": logical_name,
                    "input_files": total_input_files,
                    "periods_present": 0,
                    "periods": "",
                    "rows_before_dedup": 0,
                    "rows_after_dedup": 0,
                    "duplicate_timestamps": 0,
                    "start_timestamp": None,
                    "end_timestamp": None,
                }
            )

            continue

        # ----------------------------------------------------
        # Concatenate all periods
        # ----------------------------------------------------

        full_df = pd.concat(
            frames,
            ignore_index=True
        )

        rows_before = len(
            full_df
        )

        # ----------------------------------------------------
        # Remove duplicated timestamps
        # ----------------------------------------------------

        duplicate_count = int(
            full_df["timestamp"]
            .duplicated()
            .sum()
        )

        full_df = (
            full_df
            .drop_duplicates(
                subset=["timestamp"],
                keep="first"
            )
            .sort_values(
                "timestamp"
            )
            .reset_index(
                drop=True
            )
        )

        rows_after = len(
            full_df
        )

        # ----------------------------------------------------
        # Safe filename
        # ----------------------------------------------------

        safe_name = re.sub(
            r'[<>:"/\\|?*]',
            "_",
            logical_name
        )

        output_file = (
            OUTPUT_DIR
            / f"{safe_name}.csv"
        )

        # ----------------------------------------------------
        # Write output
        # ----------------------------------------------------

        full_df.to_csv(
            output_file,
            index=False
        )

        # ----------------------------------------------------
        # Summary
        # ----------------------------------------------------

        summary_rows.append(
            {
                "logical_metric": logical_name,
                "input_files": total_input_files,
                "periods_present": len(
                    periods_present
                ),
                "periods": "|".join(
                    periods_present
                ),
                "rows_before_dedup": rows_before,
                "rows_after_dedup": rows_after,
                "duplicate_timestamps": duplicate_count,
                "start_timestamp": full_df[
                    "timestamp"
                ].min(),
                "end_timestamp": full_df[
                    "timestamp"
                ].max(),
            }
        )

        print(
            f"    files={total_input_files} "
            f"periods={len(periods_present)} "
            f"rows={rows_after:,}"
        )

    # ========================================================
    # REPORTS
    # ========================================================

    summary_df = pd.DataFrame(
        summary_rows
    )

    missing_df = pd.DataFrame(
        missing_rows
    )

    summary_df.to_csv(
        SUMMARY_REPORT,
        index=False
    )

    missing_df.to_csv(
        MISSING_REPORT,
        index=False
    )

    return (
        summary_df,
        missing_df
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()

    print("=" * 70)
    print("MICROSS FULL METRICS BUILDER")
    print("=" * 70)

    files = discover_files()

    groups = build_groups(
        files
    )

    summary_df, missing_df = build_full_series(
        groups
    )

    print()

    print("=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)

    print(
        f"Logical metrics : {len(summary_df)}"
    )

    print(
        f"Output directory: {OUTPUT_DIR}"
    )

    print(
        f"Missing report  : {MISSING_REPORT}"
    )

    print(
        f"Summary report  : {SUMMARY_REPORT}"
    )

    if not summary_df.empty:

        print()

        print(
            "Total input files used:",
            int(
                summary_df[
                    "input_files"
                ].sum()
            )
        )

        print(
            "Total rows after dedup:",
            f"{int(summary_df['rows_after_dedup'].sum()):,}"
        )

        print(
            "Metrics with all 3 periods:",
            int(
                (
                    summary_df[
                        "periods_present"
                    ] == 3
                ).sum()
            )
        )

        print(
            "Metrics missing at least one period:",
            int(
                (
                    summary_df[
                        "periods_present"
                    ] < 3
                ).sum()
            )
        )

        print(
            "Duplicate timestamps removed:",
            int(
                summary_df[
                    "duplicate_timestamps"
                ].sum()
            )
        )

        # ----------------------------------------------------
        # Global temporal range
        # ----------------------------------------------------

        print()

        print(
            "Global start timestamp:",
            summary_df[
                "start_timestamp"
            ].min()
        )

        print(
            "Global end timestamp:",
            summary_df[
                "end_timestamp"
            ].max()
        )

    print()

    if not missing_df.empty:

        print(
            "Missing period combinations:",
            len(missing_df)
        )

        print()

        print(
            missing_df
            .head(30)
            .to_string(
                index=False
            )
        )

    else:

        print(
            "No missing period combinations detected."
        )

    print()

    print("=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":

    main()

