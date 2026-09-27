from pathlib import Path
import re
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]

FAULT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "micross"
    / "micross_fault_events.csv"
)

METRIC_DIR = PROJECT_ROOT / "MicroSS" / "metric_selected"

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "micross"
    / "micross_prediction_dataset.csv"
)

TIME_OFFSET_HOURS = 1

WINDOW_MINUTES = 5

HORIZONS = [5, 10, 15]

# Past-only rolling windows
ROLLING_SHORT = 3       # 15 minutes
ROLLING_LONG = 12       # 60 minutes

MIN_HISTORY_POINTS = 3


# ============================================================
# HELPERS
# ============================================================

def extract_service(filename: str):
    """
    Extract service name from filenames such as:

    dbservice1_0.0.0.4_docker_cpu_total_norm_pct_...
    """

    services = [
        "dbservice1",
        "dbservice2",
        "webservice1",
        "webservice2",
        "redisservice1",
        "redisservice2",
        "mobservice1",
        "mobservice2",
        "logservice1",
        "logservice2",
    ]

    for service in services:
        if filename.startswith(service + "_"):
            return service

    return None


def extract_metric_type(filename: str):
    filename_lower = filename.lower()

    if "docker_cpu_total_norm_pct" in filename_lower:
        return "cpu"

    if "docker_memory_usage_pct" in filename_lower:
        return "memory"

    return None


def robust_z_from_past(series: pd.Series, window: int = 12):
    """
    Past-only robust z-score.

    IMPORTANT:
    The current value is compared only with previous observations.
    This avoids temporal leakage.
    """

    past = series.shift(1)

    median = past.rolling(
        window=window,
        min_periods=MIN_HISTORY_POINTS
    ).median()

    mad = (
        past
        .rolling(
            window=window,
            min_periods=MIN_HISTORY_POINTS
        )
        .apply(
            lambda x: np.median(np.abs(x - np.median(x))),
            raw=True
        )
    )

    denominator = 1.4826 * mad

    z = (series - median) / denominator.replace(0, np.nan)

    return z.replace([np.inf, -np.inf], np.nan)


# ============================================================
# LOAD FAULTS
# ============================================================

print("=" * 70)
print("MICROSS - FAST PREDICTION DATASET BUILDER")
print("=" * 70)

print("\nProject root:")
print(PROJECT_ROOT)

print("\nFault file:")
print(FAULT_FILE)

print("\nMetric directory:")
print(METRIC_DIR)


print("\n[1] Loading faults...")

faults = pd.read_csv(FAULT_FILE)

faults["fault_start"] = pd.to_datetime(
    faults["fault_start"],
    errors="coerce"
)

faults["fault_end"] = pd.to_datetime(
    faults["fault_end"],
    errors="coerce"
)

# Keep original timestamps untouched
faults["aligned_fault_start"] = (
    faults["fault_start"]
    + pd.Timedelta(hours=TIME_OFFSET_HOURS)
)

faults["aligned_fault_end"] = (
    faults["fault_end"]
    + pd.Timedelta(hours=TIME_OFFSET_HOURS)
)

# Remove suspicious duration rows
if "duration_suspicious" in faults.columns:
    faults = faults[
        faults["duration_suspicious"].fillna(False) == False
    ].copy()

faults = faults.dropna(
    subset=[
        "service",
        "aligned_fault_start",
        "aligned_fault_end"
    ]
).copy()

faults = faults.sort_values(
    ["service", "aligned_fault_start"]
).reset_index(drop=True)

print(f"Valid faults: {len(faults):,}")
print(f"Timestamp working offset: +{TIME_OFFSET_HOURS}h")

print("\nFaults by service:")
print(
    faults["service"]
    .value_counts()
    .sort_index()
    .to_string()
)


# ============================================================
# LOAD AND AGGREGATE METRICS
# ============================================================

print("\n" + "=" * 70)
print("[2] Loading and aggregating metrics")
print("=" * 70)

metric_files = sorted(
    METRIC_DIR.glob("*.csv")
)

print(f"\nMetric files found: {len(metric_files)}")

if len(metric_files) == 0:
    raise FileNotFoundError(
        f"No metric CSV files found in {METRIC_DIR}"
    )


all_metric_windows = []

for i, file in enumerate(metric_files, start=1):

    service = extract_service(file.name)
    metric_type = extract_metric_type(file.name)

    if service is None or metric_type is None:
        print(
            f"  SKIP {file.name}"
            f" -> service={service}, metric={metric_type}"
        )
        continue

    print(
        f"  [{i:02d}/{len(metric_files)}] "
        f"{service:14s} | {metric_type:6s} | {file.name}"
    )

    df = pd.read_csv(
        file,
        usecols=["timestamp", "value"]
    )

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        unit="ms",
        utc=True,
        errors="coerce"
    )

    df["value"] = pd.to_numeric(
        df["value"],
        errors="coerce"
    )

    df = df.dropna(
        subset=["timestamp", "value"]
    )

    if df.empty:
        continue

    # Convert UTC timestamp to naive UTC representation
    df["timestamp"] = (
        df["timestamp"]
        .dt.tz_convert("UTC")
        .dt.tz_localize(None)
    )

    df = df.sort_values("timestamp")

    # --------------------------------------------------------
    # 5-minute aggregation
    # --------------------------------------------------------

    df = df.set_index("timestamp")

    agg = (
        df["value"]
        .resample(f"{WINDOW_MINUTES}min")
        .agg(
            [
                "count",
                "mean",
                "std",
                "min",
                "max",
                "first",
                "last",
            ]
        )
    )

    agg = agg.rename(
        columns={
            "count": f"{metric_type}_count",
            "mean": f"{metric_type}_mean",
            "std": f"{metric_type}_std",
            "min": f"{metric_type}_min",
            "max": f"{metric_type}_max",
            "first": f"{metric_type}_first",
            "last": f"{metric_type}_last",
        }
    )

    # Intra-window percentage change
    agg[f"{metric_type}_change_pct"] = (
        (
            agg[f"{metric_type}_last"]
            - agg[f"{metric_type}_first"]
        )
        / agg[f"{metric_type}_first"].abs().replace(0, np.nan)
        * 100
    )

    agg = agg.reset_index()

    agg["service"] = service

    all_metric_windows.append(agg)


if not all_metric_windows:
    raise RuntimeError(
        "No metric data could be loaded."
    )


metrics = pd.concat(
    all_metric_windows,
    ignore_index=True
)

print("\nAggregated metric rows:")
print(f"{len(metrics):,}")


# ============================================================
# BUILD SERVICE-LEVEL CPU + MEMORY TABLE
# ============================================================

print("\n" + "=" * 70)
print("[3] Building CPU + Memory windows")
print("=" * 70)

# We have one row per service/time/metric type.
# Pivot to one row per service + 5-minute window.

metrics = metrics.sort_values(
    ["service", "timestamp"]
)

index_cols = [
    "service",
    "timestamp"
]

value_cols = [
    c
    for c in metrics.columns
    if c not in index_cols
]

# Since each service/time has CPU and Memory,
# aggregate duplicated rows defensively.

metrics = (
    metrics
    .groupby(index_cols, as_index=False)[value_cols]
    .mean()
)

metrics = metrics.sort_values(
    ["service", "timestamp"]
).reset_index(drop=True)


# ============================================================
# PAST-ONLY FEATURES
# ============================================================

print("\n[4] Creating past-only temporal features...")

result_parts = []

for service, group in metrics.groupby(
    "service",
    sort=False
):

    group = group.sort_values(
        "timestamp"
    ).copy()

    # --------------------------------------------------------
    # CPU features
    # --------------------------------------------------------

    for metric_type in ["cpu", "memory"]:

        last_col = f"{metric_type}_last"

        if last_col not in group.columns:
            continue

        series = group[last_col]

        # Previous value
        group[
            f"{metric_type}_lag1"
        ] = series.shift(1)

        # Change relative to previous window
        group[
            f"{metric_type}_delta_5m"
        ] = (
            series
            - series.shift(1)
        )

        group[
            f"{metric_type}_pct_change_5m"
        ] = (
            (
                series
                - series.shift(1)
            )
            / series.shift(1).abs().replace(0, np.nan)
            * 100
        )

        # Past-only rolling statistics
        past = series.shift(1)

        group[
            f"{metric_type}_rolling_mean_15m"
        ] = (
            past
            .rolling(
                ROLLING_SHORT,
                min_periods=MIN_HISTORY_POINTS
            )
            .mean()
        )

        group[
            f"{metric_type}_rolling_std_15m"
        ] = (
            past
            .rolling(
                ROLLING_SHORT,
                min_periods=MIN_HISTORY_POINTS
            )
            .std()
        )

        group[
            f"{metric_type}_rolling_mean_60m"
        ] = (
            past
            .rolling(
                ROLLING_LONG,
                min_periods=MIN_HISTORY_POINTS
            )
            .mean()
        )

        group[
            f"{metric_type}_rolling_std_60m"
        ] = (
            past
            .rolling(
                ROLLING_LONG,
                min_periods=MIN_HISTORY_POINTS
            )
            .std()
        )

        group[
            f"{metric_type}_rolling_min_15m"
        ] = (
            past
            .rolling(
                ROLLING_SHORT,
                min_periods=MIN_HISTORY_POINTS
            )
            .min()
        )

        group[
            f"{metric_type}_rolling_max_15m"
        ] = (
            past
            .rolling(
                ROLLING_SHORT,
                min_periods=MIN_HISTORY_POINTS
            )
            .max()
        )

        # Past-only robust anomaly score
        group[
            f"{metric_type}_robust_z"
        ] = robust_z_from_past(
            series,
            window=ROLLING_LONG
        )

        group[
            f"{metric_type}_is_anomaly"
        ] = (
            group[f"{metric_type}_robust_z"].abs()
            >= 3
        ).astype(int)

    result_parts.append(group)


metrics_features = pd.concat(
    result_parts,
    ignore_index=True
)

metrics_features = metrics_features.sort_values(
    ["service", "timestamp"]
).reset_index(drop=True)

print(
    f"Rows after temporal features: "
    f"{len(metrics_features):,}"
)


# ============================================================
# FAULT TARGET GENERATION
# ============================================================

print("\n" + "=" * 70)
print("[5] Generating future incident targets")
print("=" * 70)


def add_fault_targets(service_df, fault_df):

    service_df = service_df.sort_values(
        "timestamp"
    ).copy()

    fault_df = fault_df.sort_values(
        "aligned_fault_start"
    ).copy()

    if fault_df.empty:

        for horizon in HORIZONS:
            service_df[
                f"incident_next_{horizon}m"
            ] = 0

        service_df[
            "incident_active_now"
        ] = 0

        service_df[
            "minutes_to_next_incident"
        ] = np.nan

        return service_df

    fault_starts = (
        fault_df["aligned_fault_start"]
        .values
        .astype("datetime64[ns]")
    )

    fault_ends = (
        fault_df["aligned_fault_end"]
        .values
        .astype("datetime64[ns]")
    )

    times = (
        service_df["timestamp"]
        .values
        .astype("datetime64[ns]")
    )

    # --------------------------------------------------------
    # Next fault after current observation
    # --------------------------------------------------------

    next_idx = np.searchsorted(
        fault_starts,
        times,
        side="right"
    )

    valid_next = (
        next_idx < len(fault_starts)
    )

    next_start = np.full(
        len(times),
        np.datetime64("NaT"),
        dtype="datetime64[ns]"
    )

    next_start[valid_next] = (
        fault_starts[next_idx[valid_next]]
    )

    delta_minutes = (
        (
            next_start
            - times
        )
        / np.timedelta64(1, "m")
    ).astype(float)

    service_df[
        "minutes_to_next_incident"
    ] = delta_minutes

    for horizon in HORIZONS:

        service_df[
            f"incident_next_{horizon}m"
        ] = (
            (delta_minutes > 0)
            & (delta_minutes <= horizon)
        ).astype(int)

    # --------------------------------------------------------
    # Current active fault
    # --------------------------------------------------------

    previous_idx = np.searchsorted(
        fault_starts,
        times,
        side="right"
    ) - 1

    active = np.zeros(
        len(times),
        dtype=int
    )

    valid_prev = previous_idx >= 0

    prev_starts = np.full(
        len(times),
        np.datetime64("NaT"),
        dtype="datetime64[ns]"
    )

    prev_ends = np.full(
        len(times),
        np.datetime64("NaT"),
        dtype="datetime64[ns]"
    )

    prev_starts[valid_prev] = (
        fault_starts[
            previous_idx[valid_prev]
        ]
    )

    prev_ends[valid_prev] = (
        fault_ends[
            previous_idx[valid_prev]
        ]
    )

    active = (
        valid_prev
        & (times >= prev_starts)
        & (times < prev_ends)
    ).astype(int)

    service_df[
        "incident_active_now"
    ] = active

    return service_df


target_parts = []

for service, service_df in metrics_features.groupby(
    "service",
    sort=False
):

    service_faults = faults[
        faults["service"] == service
    ].copy()

    service_df = add_fault_targets(
        service_df,
        service_faults
    )

    target_parts.append(
        service_df
    )


dataset = pd.concat(
    target_parts,
    ignore_index=True
)


# ============================================================
# DATA QUALITY FILTERING
# ============================================================

print("\n" + "=" * 70)
print("[6] Data quality filtering")
print("=" * 70)

# We require both CPU and Memory in the current window.

required_columns = [
    "cpu_count",
    "memory_count",
    "cpu_last",
    "memory_last",
]

missing_required = [
    c
    for c in required_columns
    if c not in dataset.columns
]

if missing_required:
    raise RuntimeError(
        f"Missing required columns: {missing_required}"
    )


before = len(dataset)

dataset = dataset[
    (dataset["cpu_count"].fillna(0) > 0)
    & (dataset["memory_count"].fillna(0) > 0)
].copy()

after = len(dataset)

print(
    f"Rows before metric coverage filter: {before:,}"
)

print(
    f"Rows after CPU + Memory filter: {after:,}"
)

print(
    f"Rows removed: {before - after:,}"
)


# ------------------------------------------------------------
# Do not use active incident windows as prediction samples
# ------------------------------------------------------------

before_active = len(dataset)

dataset = dataset[
    dataset["incident_active_now"] == 0
].copy()

print(
    f"Rows removed because incident already active: "
    f"{before_active - len(dataset):,}"
)


# ============================================================
# CLEAN NUMERIC FEATURES
# ============================================================

print("\n[7] Cleaning final dataset...")

# Boolean -> int
for col in dataset.columns:

    if dataset[col].dtype == bool:
        dataset[col] = dataset[col].astype(int)


# Replace infinities
dataset = dataset.replace(
    [np.inf, -np.inf],
    np.nan
)


# Sort
dataset = dataset.sort_values(
    ["service", "timestamp"]
).reset_index(drop=True)


# ============================================================
# SAVE
# ============================================================

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)

dataset.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# REPORT
# ============================================================

print("\n" + "=" * 70)
print("DATASET CREATED")
print("=" * 70)

print("\nOutput:")
print(OUTPUT_FILE)

print(f"\nShape: {dataset.shape}")

print(
    f"Services: "
    f"{dataset['service'].nunique()}"
)

print(
    f"Time range: "
    f"{dataset['timestamp'].min()} "
    f"-> "
    f"{dataset['timestamp'].max()}"
)


print("\nTarget distributions:")

for horizon in HORIZONS:

    col = f"incident_next_{horizon}m"

    positives = int(
        dataset[col].sum()
    )

    total = len(dataset)

    rate = (
        positives / total * 100
        if total > 0
        else 0
    )

    print(
        f"  {col:22s}: "
        f"{positives:6,} positives "
        f"({rate:.2f}%)"
    )


print("\nCurrent active incidents:")
print(
    dataset[
        "incident_active_now"
    ].value_counts()
    .sort_index()
    .to_string()
)


print("\nRows by service:")

print(
    dataset["service"]
    .value_counts()
    .sort_index()
    .to_string()
)


print("\nMissing values in main features:")

main_features = [
    "cpu_last",
    "memory_last",
    "cpu_robust_z",
    "memory_robust_z",
    "cpu_pct_change_5m",
    "memory_pct_change_5m",
    "incident_next_5m",
    "incident_next_10m",
    "incident_next_15m",
]

print(
    dataset[
        [
            c
            for c in main_features
            if c in dataset.columns
        ]
    ]
    .isna()
    .sum()
    .to_string()
)


print("\n" + "=" * 70)
print("DONE")
print("=" * 70)