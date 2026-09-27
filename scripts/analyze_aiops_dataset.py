"""
============================================================
HARDTEC - AIOps Dataset Analysis
============================================================

Dataset:
    AIOps Log Monitoring & Failure Detection Dataset

File:
    processed_events_analysis.csv

Purpose:
    - Inspect dataset structure
    - Validate data quality
    - Analyze event distributions
    - Analyze failure/error signals
    - Analyze temporal behavior
    - Identify candidate features for AIOps
    - Prepare the next stage:
        1. Incident Detection
        2. Incident Prediction

IMPORTANT:
    This script DOES NOT create an artificial incident label.
    It only analyzes the data and identifies possible signals.

Author:
    HARDTEC Intelligent Ticketing Project
============================================================
"""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# ============================================================
# CONFIGURATION
# ============================================================

# The script searches these locations in this order.
POSSIBLE_PATHS = [
    Path("processed_events_analysis.csv"),
    Path("data/raw/processed_events_analysis.csv"),
    Path("../data/processed_events_analysis.csv"),
]

OUTPUT_DIR = Path("aiops_analysis")

# Maximum number of values displayed for categorical columns
TOP_N = 20


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def print_title(title: str):
    """Print a formatted section title."""
    print("\n")
    print("=" * 75)
    print(title)
    print("=" * 75)


def print_subtitle(title: str):
    """Print a smaller formatted subtitle."""
    print("\n" + "-" * 65)
    print(title)
    print("-" * 65)


def find_dataset() -> Path:
    """Find the CSV file in common project locations."""

    for path in POSSIBLE_PATHS:
        if path.exists():
            return path.resolve()

    print("\n❌ Dataset not found.")
    print("\nExpected one of these locations:")

    for path in POSSIBLE_PATHS:
        print(f"   - {path}")

    print("\nExample:")
    print("   hardtec-intelligent-ticketing/")
    print("   ├── data/")
    print("   │   └── processed_events_analysis.csv")
    print("   └── scripts/")
    print("       └── analyze_aiops_dataset.py")

    raise FileNotFoundError(
        "processed_events_analysis.csv was not found."
    )


def safe_percentage(value, total):
    """Calculate percentage safely."""
    if total == 0:
        return 0.0

    return round((value / total) * 100, 2)


def save_text(filename: str, content: str):
    """Save text output to the analysis directory."""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    filepath = OUTPUT_DIR / filename

    with open(filepath, "w", encoding="utf-8") as file:
        file.write(content)

    return filepath


# ============================================================
# 1. LOAD DATASET
# ============================================================

print_title("HARDTEC AIOps - DATASET ANALYSIS")

dataset_path = find_dataset()

print(f"\n📁 Dataset found:")
print(f"   {dataset_path}")

print("\n⏳ Loading dataset...")

try:
    df = pd.read_csv(
        dataset_path,
        low_memory=False
    )
except Exception as e:
    print("\n❌ Error while reading CSV:")
    print(e)
    raise

print("\n✅ Dataset loaded successfully.")


# ============================================================
# 2. BASIC INFORMATION
# ============================================================

print_title("1. BASIC DATASET INFORMATION")

rows, columns = df.shape

print(f"\nNumber of rows    : {rows:,}")
print(f"Number of columns : {columns}")

print("\nMemory usage:")

memory_mb = df.memory_usage(deep=True).sum() / (1024 ** 2)

print(f"   {memory_mb:.2f} MB")


print("\nColumn names:")

for index, column in enumerate(df.columns, start=1):
    print(f"   {index:02d}. {column}")


# ============================================================
# 3. DATA TYPES
# ============================================================

print_title("2. DATA TYPES")

dtype_table = pd.DataFrame({
    "column": df.columns,
    "dtype": df.dtypes.astype(str).values,
    "missing": df.isna().sum().values,
    "missing_%": (
        df.isna().mean().values * 100
    ).round(2),
    "unique": [
        df[col].nunique(dropna=True)
        for col in df.columns
    ],
})

print(dtype_table.to_string(index=False))

dtype_table.to_csv(
    OUTPUT_DIR / "column_data_types.csv",
    index=False
) if OUTPUT_DIR.exists() else None


# ============================================================
# 4. MISSING VALUES
# ============================================================

print_title("3. MISSING VALUES")

missing = df.isna().sum()

missing_table = pd.DataFrame({
    "column": df.columns,
    "missing_count": missing.values,
    "missing_percentage": (
        missing.values / len(df) * 100
    ).round(2),
})

missing_table = missing_table.sort_values(
    by="missing_count",
    ascending=False
)

print(missing_table.to_string(index=False))

missing_table.to_csv(
    OUTPUT_DIR / "missing_values.csv"
    if OUTPUT_DIR.exists()
    else Path("missing_values.csv"),
    index=False
)


# ============================================================
# 5. DUPLICATES
# ============================================================

print_title("4. DUPLICATE ANALYSIS")

duplicate_rows = df.duplicated().sum()

print(f"\nExact duplicate rows : {duplicate_rows:,}")

print(
    f"Duplicate percentage : "
    f"{safe_percentage(duplicate_rows, len(df))}%"
)


# ============================================================
# 6. SAMPLE
# ============================================================

print_title("5. FIRST 10 ROWS")

print(df.head(10).to_string())


# ============================================================
# 7. NUMERICAL COLUMNS
# ============================================================

print_title("6. NUMERICAL FEATURES")

numeric_columns = df.select_dtypes(
    include=np.number
).columns.tolist()

print(f"\nNumber of numerical columns: {len(numeric_columns)}")

if numeric_columns:

    numeric_stats = df[numeric_columns].describe().T

    numeric_stats["missing"] = (
        df[numeric_columns].isna().sum()
    )

    print(
        numeric_stats.to_string()
    )

    numeric_stats.to_csv(
        OUTPUT_DIR / "numeric_statistics.csv"
        if OUTPUT_DIR.exists()
        else Path("numeric_statistics.csv")
    )

else:

    print("\nNo numerical columns detected.")


# ============================================================
# 8. CATEGORICAL FEATURES
# ============================================================

print_title("7. CATEGORICAL FEATURES")

categorical_columns = df.select_dtypes(
    include=["object", "category", "bool"]
).columns.tolist()

print(
    f"\nNumber of categorical columns: "
    f"{len(categorical_columns)}"
)

for column in categorical_columns:

    print_subtitle(
        f"{column} - TOP {TOP_N} VALUES"
    )

    counts = (
        df[column]
        .fillna("<NULL>")
        .value_counts()
        .head(TOP_N)
    )

    percentages = (
        counts / len(df) * 100
    ).round(2)

    result = pd.DataFrame({
        "count": counts,
        "percentage": percentages
    })

    print(result.to_string())


# ============================================================
# 9. IMPORTANT AIOPS COLUMNS
# ============================================================

print_title("8. AIOps IMPORTANT SIGNALS")

important_columns = [
    "event_type",
    "category",
    "severity",
    "status",
    "status_code",
    "level",
    "environment",
    "hostname",
    "region",
    "method",
    "path",
]

available_important = [
    column
    for column in important_columns
    if column in df.columns
]

missing_important = [
    column
    for column in important_columns
    if column not in df.columns
]

print("\nAvailable important columns:")

for column in available_important:
    print(f"   ✅ {column}")

print("\nMissing expected columns:")

for column in missing_important:
    print(f"   ⚠️ {column}")


# ============================================================
# 10. DISTRIBUTIONS OF IMPORTANT AIOPS FEATURES
# ============================================================

print_title("9. IMPORTANT FEATURE DISTRIBUTIONS")

for column in available_important:

    print_subtitle(column)

    counts = (
        df[column]
        .fillna("<NULL>")
        .value_counts()
        .head(TOP_N)
    )

    percentages = (
        counts / len(df) * 100
    ).round(2)

    result = pd.DataFrame({
        "count": counts,
        "percentage": percentages
    })

    print(result.to_string())


# ============================================================
# 11. FAILURE / ERROR SIGNAL ANALYSIS
# ============================================================

print_title("10. FAILURE / ERROR SIGNAL ANALYSIS")

failure_keywords = [
    "failure",
    "failed",
    "error",
    "critical",
    "exception",
    "timeout",
    "crash",
    "fatal",
    "server_error",
    "5xx",
]

# ------------------------------------------------------------
# Event type
# ------------------------------------------------------------

if "event_type" in df.columns:

    print_subtitle("Event Types Related To Failure")

    event_type_text = (
        df["event_type"]
        .fillna("")
        .astype(str)
        .str.lower()
    )

    mask = event_type_text.apply(
        lambda value: any(
            keyword in value
            for keyword in failure_keywords
        )
    )

    failure_events = df.loc[mask]

    print(
        f"\nPotential failure/error events: "
        f"{len(failure_events):,}"
    )

    print(
        f"Percentage of dataset: "
        f"{safe_percentage(len(failure_events), len(df))}%"
    )

    if len(failure_events) > 0:

        print("\nDetected values:")

        print(
            df.loc[mask, "event_type"]
            .value_counts()
            .to_string()
        )


# ------------------------------------------------------------
# Category
# ------------------------------------------------------------

if "category" in df.columns:

    print_subtitle("Categories Related To Failure")

    category_text = (
        df["category"]
        .fillna("")
        .astype(str)
        .str.lower()
    )

    mask = category_text.apply(
        lambda value: any(
            keyword in value
            for keyword in failure_keywords
        )
    )

    print(
        f"\nPotential failure categories: "
        f"{mask.sum():,}"
    )

    if mask.sum() > 0:

        print(
            df.loc[mask, "category"]
            .value_counts()
            .to_string()
        )


# ------------------------------------------------------------
# Severity
# ------------------------------------------------------------

if "severity" in df.columns:

    print_subtitle("Severity Distribution")

    severity_counts = (
        df["severity"]
        .fillna("<NULL>")
        .value_counts()
    )

    severity_table = pd.DataFrame({
        "count": severity_counts,
        "percentage": (
            severity_counts / len(df) * 100
        ).round(2)
    })

    print(
        severity_table.to_string()
    )


# ------------------------------------------------------------
# Status
# ------------------------------------------------------------

if "status" in df.columns:

    print_subtitle("Status Distribution")

    status_counts = (
        df["status"]
        .fillna("<NULL>")
        .value_counts()
    )

    status_table = pd.DataFrame({
        "count": status_counts,
        "percentage": (
            status_counts / len(df) * 100
        ).round(2)
    })

    print(
        status_table.to_string()
    )


# ============================================================
# 12. HTTP STATUS CODE ANALYSIS
# ============================================================

print_title("11. HTTP STATUS CODE ANALYSIS")

if "status_code" in df.columns:

    status_code_numeric = pd.to_numeric(
        df["status_code"],
        errors="coerce"
    )

    valid_status = status_code_numeric.notna()

    print(
        f"\nValid HTTP status codes: "
        f"{valid_status.sum():,}"
    )

    # 2xx
    http_2xx = (
        status_code_numeric.between(200, 299)
    ).sum()

    # 3xx
    http_3xx = (
        status_code_numeric.between(300, 399)
    ).sum()

    # 4xx
    http_4xx = (
        status_code_numeric.between(400, 499)
    ).sum()

    # 5xx
    http_5xx = (
        status_code_numeric.between(500, 599)
    ).sum()

    http_table = pd.DataFrame({
        "class": [
            "2xx Success",
            "3xx Redirect",
            "4xx Client Error",
            "5xx Server Error",
        ],
        "count": [
            http_2xx,
            http_3xx,
            http_4xx,
            http_5xx,
        ],
    })

    http_table["percentage"] = (
        http_table["count"] / len(df) * 100
    ).round(2)

    print(
        http_table.to_string(index=False)
    )

else:

    print("\nstatus_code column not available.")


# ============================================================
# 13. PERFORMANCE / LATENCY ANALYSIS
# ============================================================

print_title("12. PERFORMANCE / LATENCY ANALYSIS")

if "duration_ms" in df.columns:

    duration = pd.to_numeric(
        df["duration_ms"],
        errors="coerce"
    )

    print("\nDuration statistics:")

    print(
        duration.describe().to_string()
    )

    print("\nLatency thresholds:")

    thresholds = [100, 250, 500, 1000, 2000, 5000]

    for threshold in thresholds:

        count = (
            duration > threshold
        ).sum()

        print(
            f"   > {threshold:>4} ms : "
            f"{count:,} "
            f"({safe_percentage(count, len(df))}%)"
        )

else:

    print("\nduration_ms column not available.")


# ============================================================
# 14. TIMESTAMP ANALYSIS
# ============================================================

print_title("13. TEMPORAL ANALYSIS")

timestamp_column = None

for candidate in [
    "timestamp",
    "received_at",
    "date",
]:

    if candidate in df.columns:

        timestamp_column = candidate
        break


if timestamp_column:

    print(
        f"\nTimestamp column selected: "
        f"{timestamp_column}"
    )

    timestamps = pd.to_datetime(
        df[timestamp_column],
        errors="coerce",
        utc=True
    )

    valid_timestamps = timestamps.notna()

    print(
        f"\nValid timestamps: "
        f"{valid_timestamps.sum():,}"
    )

    if valid_timestamps.sum() > 0:

        print(
            f"\nMinimum timestamp: "
            f"{timestamps.min()}"
        )

        print(
            f"Maximum timestamp: "
            f"{timestamps.max()}"
        )

        time_span = (
            timestamps.max()
            - timestamps.min()
        )

        print(
            f"Time span: {time_span}"
        )

        # Temporal components
        temporal_df = pd.DataFrame({
            "timestamp": timestamps
        })

        temporal_df["hour"] = (
            temporal_df["timestamp"].dt.hour
        )

        temporal_df["day_of_week"] = (
            temporal_df["timestamp"].dt.day_name()
        )

        temporal_df["date"] = (
            temporal_df["timestamp"].dt.date
        )

        temporal_df["month"] = (
            temporal_df["timestamp"].dt.to_period("M")
            .astype(str)
        )

        print_subtitle("Events By Hour")

        print(
            temporal_df["hour"]
            .value_counts()
            .sort_index()
            .to_string()
        )

        print_subtitle("Events By Day Of Week")

        print(
            temporal_df["day_of_week"]
            .value_counts()
            .to_string()
        )

        print_subtitle("Events By Date")

        daily_events = (
            temporal_df["date"]
            .value_counts()
            .sort_index()
        )

        print(
            daily_events.to_string()
        )

        daily_events.to_csv(
            OUTPUT_DIR / "daily_event_counts.csv"
            if OUTPUT_DIR.exists()
            else Path("daily_event_counts.csv")
        )

    else:

        print(
            "\n⚠️ No valid timestamps detected."
        )

else:

    print(
        "\n⚠️ No timestamp column found."
    )


# ============================================================
# 15. PROVIDED TEMPORAL FEATURES
# ============================================================

print_title("14. PROVIDED TEMPORAL FEATURES")

temporal_columns = [
    "hour",
    "date",
    "day_of_week",
    "month",
    "week",
    "day",
]

for column in temporal_columns:

    if column in df.columns:

        print_subtitle(column)

        print(
            df[column]
            .fillna("<NULL>")
            .value_counts()
            .head(TOP_N)
            .to_string()
        )


# ============================================================
# 16. HOST / INSTANCE ANALYSIS
# ============================================================

print_title("15. INFRASTRUCTURE ANALYSIS")

for column in [
    "hostname",
    "instance_id",
    "process_id",
    "region",
    "environment",
    "project",
]:

    if column in df.columns:

        unique_count = (
            df[column]
            .nunique(dropna=True)
        )

        print(
            f"\n{column}: "
            f"{unique_count:,} unique values"
        )

        print(
            df[column]
            .fillna("<NULL>")
            .value_counts()
            .head(TOP_N)
            .to_string()
        )


# ============================================================
# 17. MESSAGE ANALYSIS
# ============================================================

print_title("16. MESSAGE / LOG CONTENT ANALYSIS")

if "message" in df.columns:

    message = (
        df["message"]
        .fillna("")
        .astype(str)
    )

    message_length = (
        message.str.len()
    )

    print(
        "\nMessage length statistics:"
    )

    print(
        message_length.describe().to_string()
    )

    print(
        "\nEmpty messages:"
    )

    empty_messages = (
        message.str.strip() == ""
    ).sum()

    print(
        f"   {empty_messages:,} "
        f"({safe_percentage(empty_messages, len(df))}%)"
    )

    print("\nSample messages:")

    for value in (
        message[message.str.strip() != ""]
        .head(10)
    ):

        print(
            f"   - {value[:250]}"
        )


# ============================================================
# 18. FAILURE CORRELATION ANALYSIS
# ============================================================

print_title("17. FAILURE SIGNAL CORRELATION")

failure_mask = pd.Series(
    False,
    index=df.index
)

# status
if "status" in df.columns:

    status_text = (
        df["status"]
        .fillna("")
        .astype(str)
        .str.lower()
    )

    failure_mask |= status_text.isin([
        "failure",
        "failed",
        "error",
        "critical",
        "failure_event",
    ])

# event type
if "event_type" in df.columns:

    event_text = (
        df["event_type"]
        .fillna("")
        .astype(str)
        .str.lower()
    )

    failure_mask |= event_text.apply(
        lambda value: any(
            keyword in value
            for keyword in failure_keywords
        )
    )

# status code
if "status_code" in df.columns:

    status_code_numeric = pd.to_numeric(
        df["status_code"],
        errors="coerce"
    )

    failure_mask |= (
        status_code_numeric >= 500
    )

print(
    f"\nRows containing at least one "
    f"obvious failure/error signal:"
)

print(
    f"   {failure_mask.sum():,} "
    f"({safe_percentage(failure_mask.sum(), len(df))}%)"
)


# ============================================================
# 19. FAILURE SIGNALS BY CATEGORY
# ============================================================

if failure_mask.sum() > 0:

    for column in [
        "event_type",
        "category",
        "severity",
        "status",
        "level",
        "environment",
        "hostname",
    ]:

        if column in df.columns:

            print_subtitle(
                f"{column} among failure/error events"
            )

            print(
                df.loc[
                    failure_mask,
                    column
                ]
                .fillna("<NULL>")
                .value_counts()
                .head(TOP_N)
                .to_string()
            )


# ============================================================
# 20. TEMPORAL FAILURE ANALYSIS
# ============================================================

print_title("18. TEMPORAL FAILURE ANALYSIS")

if timestamp_column and failure_mask.sum() > 0:

    timestamps = pd.to_datetime(
        df[timestamp_column],
        errors="coerce",
        utc=True
    )

    temporal_failure = pd.DataFrame({
        "timestamp": timestamps,
        "failure": failure_mask
    })

    temporal_failure = (
        temporal_failure
        .dropna(subset=["timestamp"])
        .set_index("timestamp")
    )

    hourly = (
        temporal_failure["failure"]
        .resample("1h")
        .agg(
            events="count",
            failures="sum"
        )
    )

    hourly["failure_rate_%"] = (
        hourly["failures"]
        / hourly["events"]
        * 100
    ).round(2)

    print(
        "\nHourly failure statistics:"
    )

    print(
        hourly.tail(30).to_string()
    )

    hourly.to_csv(
        OUTPUT_DIR / "hourly_failure_analysis.csv"
        if OUTPUT_DIR.exists()
        else Path("hourly_failure_analysis.csv")
    )

    # Highest failure periods
    print_subtitle(
        "Periods With Highest Failure Counts"
    )

    highest_failure_hours = (
        hourly
        .sort_values(
            "failures",
            ascending=False
        )
        .head(20)
    )

    print(
        highest_failure_hours.to_string()
    )


# ============================================================
# 21. POTENTIAL FEATURES FOR INCIDENT DETECTION
# ============================================================

print_title("19. POTENTIAL FEATURES FOR INCIDENT DETECTION")

detection_features = [
    "event_type",
    "category",
    "severity",
    "status",
    "status_code",
    "level",
    "duration_ms",
    "hour",
    "day_of_week",
    "month",
    "hostname",
    "instance_id",
    "region",
    "environment",
    "method",
    "path",
]

available_detection_features = [
    column
    for column in detection_features
    if column in df.columns
]

print(
    "\nPotential detection features:"
)

for feature in available_detection_features:

    print(
        f"   ✓ {feature}"
    )


# ============================================================
# 22. POTENTIAL FEATURES FOR INCIDENT PREDICTION
# ============================================================

print_title("20. POTENTIAL FEATURES FOR INCIDENT PREDICTION")

prediction_features = [
    "timestamp",
    "event_type",
    "category",
    "severity",
    "status",
    "status_code",
    "duration_ms",
    "hostname",
    "instance_id",
    "process_id",
    "region",
    "environment",
    "hour",
    "day_of_week",
    "week",
    "month",
]

available_prediction_features = [
    column
    for column in prediction_features
    if column in df.columns
]

print(
    "\nPotential prediction features:"
)

for feature in available_prediction_features:

    print(
        f"   ✓ {feature}"
    )


# ============================================================
# 23. DATA LEAKAGE WARNING
# ============================================================

print_title("21. DATA LEAKAGE CHECK")

print(
    """
IMPORTANT:

For Incident Prediction, we must NOT use information
that only becomes available AFTER the incident occurs.

Examples of potentially dangerous variables:

    - post-incident status
    - final resolution information
    - future event information
    - labels derived from the future
    - fields that directly reveal the failure

The prediction task must respect time:

    Past Window
        ↓
    Features
        ↓
    Prediction
        ↓
    Future Window

Example:

    t-3   t-2   t-1
      \\    |    /
       \\   |   /
        MODEL
          |
          ↓
    Incident at t+1
"""
)


# ============================================================
# 24. DATASET READINESS REPORT
# ============================================================

print_title("22. AIOps READINESS SUMMARY")

print("\nDataset:")
print(f"   Rows       : {rows:,}")
print(f"   Columns    : {columns}")
print(f"   Memory     : {memory_mb:.2f} MB")

print("\nTemporal information:")

if timestamp_column:
    print(
        f"   ✓ Timestamp column: {timestamp_column}"
    )
else:
    print(
        "   ❌ No timestamp column"
    )

print("\nFailure signals:")

if failure_mask.sum() > 0:

    print(
        f"   ✓ Failure/error signals detected: "
        f"{failure_mask.sum():,}"
    )

else:

    print(
        "   ⚠️ No obvious failure signal detected"
    )

print("\nAIOps candidate capabilities:")

print(
    "   ✓ Log/event monitoring"
)

print(
    "   ✓ Failure/error analysis"
)

if timestamp_column:

    print(
        "   ✓ Temporal analysis"
    )

    print(
        "   ✓ Time-window based prediction possible"
    )

else:

    print(
        "   ❌ Temporal prediction requires timestamps"
    )

print(
    "   ✓ Anomaly detection candidate"
)

print(
    "   ✓ Failure prediction candidate"
)


# ============================================================
# 25. SAVE ANALYSIS SUMMARY
# ============================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

summary_lines = []

summary_lines.append(
    "HARDTEC AIOps Dataset Analysis"
)

summary_lines.append(
    "=" * 50
)

summary_lines.append(
    f"Dataset: {dataset_path}"
)

summary_lines.append(
    f"Rows: {rows:,}"
)

summary_lines.append(
    f"Columns: {columns}"
)

summary_lines.append(
    f"Memory MB: {memory_mb:.2f}"
)

summary_lines.append(
    f"Duplicate rows: {duplicate_rows:,}"
)

summary_lines.append(
    f"Timestamp column: {timestamp_column}"
)

summary_lines.append(
    f"Failure/error signal rows: {failure_mask.sum():,}"
)

summary_lines.append(
    f"Failure/error signal percentage: "
    f"{safe_percentage(failure_mask.sum(), len(df))}%"
)

summary_lines.append("")

summary_lines.append(
    "Available detection features:"
)

summary_lines.extend(
    [
        f" - {feature}"
        for feature in available_detection_features
    ]
)

summary_lines.append("")

summary_lines.append(
    "Available prediction features:"
)

summary_lines.extend(
    [
        f" - {feature}"
        for feature in available_prediction_features
    ]
)

summary_path = save_text(
    "aiops_analysis_summary.txt",
    "\n".join(summary_lines)
)


# ============================================================
# 26. FINAL MESSAGE
# ============================================================

print_title("ANALYSIS FINISHED")

print(
    f"""
✅ Analysis completed successfully.

Dataset:
    {dataset_path}

Results directory:
    {OUTPUT_DIR.resolve()}

Important output:
    {summary_path.resolve()}

NEXT STEP:
    Send me the complete terminal output.

Do NOT train an AIOps model yet.

We first need to determine from the real data:

    1. What exactly represents a failure?
    2. Whether anomaly detection is appropriate.
    3. Whether a reliable temporal prediction target
       can be constructed without data leakage.
    4. Which features should be used.
    5. Which ML model should be selected.

Then we will implement:

    HARDTEC AIOps
        │
        ├── Incident Detection
        │       └── Anomaly Detection
        │
        └── Incident Prediction
                └── Temporal ML

without inventing labels.
"""
)

print("\n")
print("=" * 75)
print("END OF ANALYSIS")
print("=" * 75)

