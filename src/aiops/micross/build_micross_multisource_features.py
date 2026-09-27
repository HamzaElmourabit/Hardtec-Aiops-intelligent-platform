from pathlib import Path
import re
import gc
import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]

METRIC_DIR = (
    PROJECT_ROOT
    / "MicroSS"
    / "metric_candidates"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "micross"
)

OUTPUT_FILE = (
    OUTPUT_DIR
    / "micross_multisource_features_5m.csv"
)

# MicroSS metrics are sampled at different frequencies.
# We normalize them to 5-minute windows.
RESAMPLE_RULE = "5min"

# Working hypothesis established from previous alignment tests.
# This does NOT modify original RUN timestamps.
TIMESTAMP_OFFSET_HOURS = 1

ROBUST_Z_THRESHOLD = 3.0


# ============================================================
# SERVICES
# ============================================================

SERVICES = [
    "dbservice1",
    "dbservice2",
    "logservice1",
    "logservice2",
    "mobservice1",
    "mobservice2",
    "redisservice1",
    "redisservice2",
    "webservice1",
    "webservice2",
]


# ============================================================
# METRIC IDENTIFICATION
# ============================================================

def detect_service(filename: str) -> str:

    name = filename.lower()

    for service in SERVICES:
        if service in name:
            return service

    return "SYSTEM"


def detect_metric_group(filename: str) -> str:

    name = filename.lower()

    if "network" in name:
        return "network"

    if "diskio" in name:
        return "disk_io"

    if "docker_cpu" in name:
        return "cpu"

    if "system_cpu" in name:
        return "cpu"

    if "_cpu_" in name:
        return "cpu"

    if "docker_memory" in name:
        return "memory"

    if "system_memory" in name:
        return "memory"

    if "load" in name:
        return "load"

    if "filesystem" in name:
        return "filesystem"

    return "other"


def safe_metric_name(filename: str) -> str:

    name = Path(filename).stem.lower()

    # --------------------------------------------------------
    # Remove date range
    # --------------------------------------------------------

    name = re.sub(
        r"_2021-08-01_2021-08-31$",
        "",
        name,
    )

    # --------------------------------------------------------
    # IMPORTANT
    #
    # For SYSTEM metrics we KEEP the IP identifiers.
    #
    # Example:
    #
    # system_0.0.0.2_0.0.0.2_system_cpu_idle_norm_pct
    #
    # system_0.0.0.3_0.0.0.3_system_cpu_idle_norm_pct
    #
    # These are different system sources and must not
    # become the same feature.
    # --------------------------------------------------------

    if name.startswith("system_"):

        # Keep the IP information.
        name = re.sub(
            r"[^a-zA-Z0-9]+",
            "_",
            name,
        )

    else:

        # ----------------------------------------------------
        # Service metrics
        #
        # For service metrics the Docker IP is not useful
        # as a feature identifier, so we remove it.
        # ----------------------------------------------------

        name = re.sub(
            r"(?:\d{1,3}\.){3}\d{1,3}",
            "",
            name,
        )

        name = re.sub(
            r"[^a-zA-Z0-9]+",
            "_",
            name,
        )

    # Normalize repeated underscores.

    name = re.sub(
        r"_+",
        "_",
        name,
    )

    return name.strip("_")


# ============================================================
# ROBUST Z
# ============================================================

def compute_robust_z(series: pd.Series) -> pd.Series:

    median = series.median()

    values = series.dropna().to_numpy()

    if len(values) == 0:

        return pd.Series(
            np.nan,
            index=series.index,
            dtype="float32",
        )

    mad = np.median(
        np.abs(values - median)
    )

    if not np.isfinite(mad) or mad == 0:

        return pd.Series(
            0.0,
            index=series.index,
            dtype="float32",
        )

    z = (
        0.6745
        * (series - median)
        / mad
    )

    return z.astype("float32")


# ============================================================
# PROCESS ONE METRIC
# ============================================================

def process_metric(file_path: Path):

    try:

        df = pd.read_csv(
            file_path,
            usecols=["timestamp", "value"],
        )

    except Exception as exc:

        print(
            f"      ERROR reading file: {exc}"
        )

        return None

    if df.empty:
        return None

    # ========================================================
    # NUMERIC CONVERSION
    # ========================================================

    df["timestamp"] = pd.to_numeric(
        df["timestamp"],
        errors="coerce",
    )

    df["value"] = pd.to_numeric(
        df["value"],
        errors="coerce",
    )

    df = df.dropna(
        subset=[
            "timestamp",
            "value",
        ]
    )

    if df.empty:
        return None

    # ========================================================
    # TIMESTAMP
    # ========================================================

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        unit="ms",
        utc=True,
        errors="coerce",
    )

    df = df.dropna(
        subset=["timestamp"]
    )

    if df.empty:
        return None

    # ========================================================
    # WORKING ALIGNMENT OFFSET
    # ========================================================

    df["timestamp"] = (
        df["timestamp"]
        + pd.Timedelta(
            hours=TIMESTAMP_OFFSET_HOURS
        )
    )

    df = df.set_index(
        "timestamp"
    )

    # ========================================================
    # 5-MINUTE AGGREGATION
    # ========================================================

    agg = (
        df["value"]
        .resample(RESAMPLE_RULE)
        .agg(
            [
                "mean",
                "std",
                "min",
                "max",
                "last",
                "count",
            ]
        )
    )

    del df
    gc.collect()

    if agg.empty:
        return None

    agg = agg.dropna(
        subset=[
            "mean",
            "last",
        ]
    )

    if agg.empty:
        return None

    # ========================================================
    # METADATA
    # ========================================================

    service = detect_service(
        file_path.name
    )

    group = detect_metric_group(
        file_path.name
    )

    metric_name = safe_metric_name(
        file_path.name
    )

    # ========================================================
    # FEATURES
    # ========================================================

    last = agg["last"]

    previous = last.shift(1)

    delta = (
        last
        - previous
    )

    pct_change = (
        last
        .pct_change()
        .replace(
            [
                np.inf,
                -np.inf,
            ],
            np.nan,
        )
        .clip(
            -10,
            10,
        )
    )

    robust_z = compute_robust_z(
        last
    )

    anomaly = (
        robust_z.abs()
        >= ROBUST_Z_THRESHOLD
    ).astype(
        "int8"
    )

    # ========================================================
    # RESULT
    # ========================================================

    result = pd.DataFrame(
        {
            "timestamp": agg.index,

            "service": service,

            "metric_group": group,

            "metric_name": metric_name,

            "mean": agg["mean"]
                .astype("float32"),

            "std": agg["std"]
                .fillna(0)
                .astype("float32"),

            "min": agg["min"]
                .astype("float32"),

            "max": agg["max"]
                .astype("float32"),

            "last": last
                .astype("float32"),

            "count": agg["count"]
                .astype("int32"),

            "delta": delta
                .astype("float32"),

            "pct_change": pct_change
                .astype("float32"),

            "robust_z": robust_z,

            "is_anomaly": anomaly,
        }
    )

    del agg
    del last
    del previous
    del delta
    del pct_change
    del robust_z
    del anomaly

    gc.collect()

    return result


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print(
        "MICROSS - MULTI-SOURCE FEATURE BUILDER "
        "(RAM OPTIMIZED)"
    )
    print("=" * 70)
    print()

    print("Project root:")
    print(PROJECT_ROOT)

    print()
    print("Metric directory:")
    print(METRIC_DIR)

    print()
    print("Output:")
    print(OUTPUT_FILE)

    print()
    print(
        "Timestamp working offset: "
        f"+{TIMESTAMP_OFFSET_HOURS} hour"
    )

    print()
    print(
        f"Aggregation: {RESAMPLE_RULE}"
    )

    print()

    if not METRIC_DIR.exists():

        raise FileNotFoundError(
            f"Metric directory not found: "
            f"{METRIC_DIR}"
        )

    files = sorted(
        METRIC_DIR.rglob("*.csv")
    )

    print(
        f"Metric files found: {len(files)}"
    )

    if not files:

        raise RuntimeError(
            "No metric CSV files found."
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ========================================================
    # PROCESS ONE FILE AT A TIME
    # ========================================================

    metric_tables = []

    for index, file_path in enumerate(
        files,
        start=1,
    ):

        print(
            f"[{index:03d}/{len(files):03d}] "
            f"{file_path.name}"
        )

        result = process_metric(
            file_path
        )

        if result is None:

            print(
                "      -> skipped"
            )

            continue

        # ----------------------------------------------------
        # Metadata
        # ----------------------------------------------------

        service = result[
            "service"
        ].iloc[0]

        metric_name = result[
            "metric_name"
        ].iloc[0]

        # ----------------------------------------------------
        # Compact feature table
        # ----------------------------------------------------

        compact = result[
            [
                "timestamp",
                "service",
                "last",
                "delta",
                "pct_change",
                "robust_z",
                "is_anomaly",
            ]
        ].copy()

        # ----------------------------------------------------
        # Unique prefix
        # ----------------------------------------------------

        prefix = (
            f"{service}__"
            f"{metric_name}"
        )

        compact.rename(
            columns={
                "last":
                    f"{prefix}__last",

                "delta":
                    f"{prefix}__delta",

                "pct_change":
                    f"{prefix}__pct_change",

                "robust_z":
                    f"{prefix}__robust_z",

                "is_anomaly":
                    f"{prefix}__is_anomaly",
            },
            inplace=True,
        )

        # ----------------------------------------------------
        # Timestamp as index
        # ----------------------------------------------------

        compact = compact.drop(
            columns=[
                "service"
            ]
        )

        compact = compact.set_index(
            "timestamp"
        )

        metric_tables.append(
            compact
        )

        del result
        del compact

        gc.collect()

    # ========================================================
    # ALL METRICS PROCESSED
    # ========================================================

    print()
    print("=" * 70)
    print(
        "ALL METRICS PROCESSED"
    )
    print("=" * 70)
    print()

    print(
        f"Valid metric tables: "
        f"{len(metric_tables)}"
    )

    # ========================================================
    # SEPARATE SERVICE / SYSTEM
    # ========================================================

    service_tables = {
        service: []
        for service in SERVICES
    }

    system_tables = []

    for table in metric_tables:

        first_column = table.columns[0]

        service = first_column.split(
            "__",
            1
        )[0]

        if service == "SYSTEM":

            system_tables.append(
                table
            )

        elif service in service_tables:

            service_tables[
                service
            ].append(
                table
            )

    del metric_tables
    gc.collect()

    # ========================================================
    # HELPER
    # ========================================================

    def merge_tables(tables):

        if not tables:
            return None

        base = tables[0].copy()

        for table in tables[1:]:

            # Safety check against duplicate columns
            overlapping = (
                base.columns
                .intersection(
                    table.columns
                )
            )

            if len(overlapping) > 0:

                # This should normally never happen
                # after the SYSTEM naming correction.
                print(
                    "      WARNING: duplicate "
                    "columns detected:"
                )

                for column in overlapping:
                    print(
                        f"         {column}"
                    )

                # Rename duplicates using a suffix.
                table = table.rename(
                    columns={
                        column:
                        f"{column}__duplicate"
                        for column in overlapping
                    }
                )

            base = base.join(
                table,
                how="outer",
            )

            del table

            gc.collect()

        return base

    # ========================================================
    # SYSTEM FEATURES
    # ========================================================

    print()
    print(
        "Building SYSTEM-level feature table..."
    )

    system_base = merge_tables(
        system_tables
    )

    del system_tables
    gc.collect()

    if system_base is not None:

        system_base = (
            system_base
            .sort_index()
        )

        print(
            "SYSTEM shape:",
            system_base.shape
        )

    # ========================================================
    # SERVICE TABLES
    # ========================================================

    print()
    print(
        "Building service-level tables..."
    )

    service_results = []

    for service in SERVICES:

        print(
            f"   {service}"
        )

        tables = service_tables[
            service
        ]

        if not tables:

            print(
                "      -> no metrics"
            )

            continue

        service_base = merge_tables(
            tables
        )

        del service_tables[
            service
        ]

        gc.collect()

        if service_base is None:
            continue

        # ----------------------------------------------------
        # Add service identifier
        # ----------------------------------------------------

        service_base[
            "service"
        ] = service

        service_base = (
            service_base
            .reset_index()
        )

        service_results.append(
            service_base
        )

        del service_base

        gc.collect()

    del service_tables
    gc.collect()

    # ========================================================
    # COMBINE SERVICES
    # ========================================================

    print()
    print(
        "Combining service tables..."
    )

    if not service_results:

        raise RuntimeError(
            "No service metric table created."
        )

    services_df = pd.concat(
        service_results,
        ignore_index=True,
    )

    del service_results
    gc.collect()

    print(
        "Service matrix shape:",
        services_df.shape
    )

    # ========================================================
    # ATTACH SYSTEM FEATURES
    # ========================================================

    if system_base is not None:

        print()
        print(
            "Attaching SYSTEM features..."
        )

        system_base = (
            system_base
            .reset_index()
        )

        # ----------------------------------------------------
        # SYSTEM features are global infrastructure signals.
        #
        # They are replicated for every service at the same
        # timestamp.
        # ----------------------------------------------------

        services_df = services_df.merge(
            system_base,
            on="timestamp",
            how="left",
        )

        del system_base
        gc.collect()

    # ========================================================
    # SORT
    # ========================================================

    print()
    print(
        "Sorting final dataset..."
    )

    services_df = services_df.sort_values(
        [
            "service",
            "timestamp",
        ]
    )

    # ========================================================
    # REMOVE DUPLICATES
    # ========================================================

    services_df = (
        services_df
        .drop_duplicates(
            subset=[
                "timestamp",
                "service",
            ],
            keep="first",
        )
    )

    # ========================================================
    # SAVE
    # ========================================================

    print()
    print(
        "Saving final dataset..."
    )

    services_df.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    print()
    print("=" * 70)
    print(
        "DATASET CREATED"
    )
    print("=" * 70)
    print()

    print(
        "Output:"
    )

    print(
        OUTPUT_FILE
    )

    print()

    print(
        "Shape:",
        services_df.shape
    )

    print()

    print(
        "Services:"
    )

    print(
        services_df[
            "service"
        ]
        .value_counts()
        .sort_index()
    )

    print()

    print(
        "Time range:"
    )

    print(
        "Start:",
        services_df[
            "timestamp"
        ].min(),
    )

    print(
        "End:",
        services_df[
            "timestamp"
        ].max(),
    )

    print()

    print(
        "Total feature columns:",
        len(
            services_df.columns
        ),
    )

    print()

    print(
        "Memory usage:"
    )

    print(
        f"{services_df.memory_usage(deep=True).sum() / 1024**2:.2f} MB"
    )

    print()

    print(
        "First columns:"
    )

    for column in (
        services_df.columns[:20]
    ):

        print(
            f"  {column}"
        )

    print()

    print(
        "MULTI-SOURCE FEATURE BUILD COMPLETE."
    )

    print()


if __name__ == "__main__":
    main()