
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

HARDTEC_ROOT = Path(
    r"C:\Users\khadi\Desktop\hardtec-intelligent-ticketing"
)

FEATURE_FILE = (
    HARDTEC_ROOT
    / "data"
    / "processed"
    / "micross"
    / "micross_multisource_features_5m.csv"
)

FAULT_FILE = (
    HARDTEC_ROOT
    / "data"
    / "processed"
    / "micross"
    / "micross_fault_events.csv"
)

OUTPUT_DIR = (
    HARDTEC_ROOT
    / "data"
    / "processed"
    / "micross"
)

ANALYSIS_FILE = (
    OUTPUT_DIR
    / "micross_multisource_incident_analysis.csv"
)

SUMMARY_FILE = (
    OUTPUT_DIR
    / "micross_multisource_incident_analysis_summary.csv"
)

EVENT_FILE = (
    OUTPUT_DIR
    / "micross_incident_labels_5m.csv"
)


# ============================================================
# ALIGNMENT
# ============================================================

# IMPORTANT:
# +1h reste une hypothèse de travail.
# Les timestamps originaux ne sont jamais modifiés.

FAULT_TIME_OFFSET = pd.Timedelta(hours=1)


# ============================================================
# TIMESTAMP CONVERSION
# ============================================================

def convert_timestamp(series: pd.Series) -> pd.Series:
    """
    Convertit automatiquement :
    - Unix timestamps en millisecondes
    - timestamps texte
    """

    numeric = pd.to_numeric(
        series,
        errors="coerce",
    )

    numeric_ratio = numeric.notna().mean()

    if numeric_ratio > 0.95:

        return pd.to_datetime(
            numeric,
            unit="ms",
            errors="coerce",
        )

    return pd.to_datetime(
        series,
        errors="coerce",
    )


# ============================================================
# FIND TIMESTAMP COLUMN
# ============================================================

def find_timestamp_column(
    df: pd.DataFrame,
) -> str:

    candidates = [
        "timestamp",
        "time",
        "datetime",
        "date",
        "event_time",
        "start_time",
    ]

    lower_map = {
        str(col).lower(): col
        for col in df.columns
    }

    for candidate in candidates:

        if candidate in lower_map:

            return lower_map[candidate]

    raise ValueError(
        "No timestamp column found. "
        f"Available columns: {list(df.columns)}"
    )


def find_fault_timestamp_column(
    df: pd.DataFrame,
) -> str:

    candidates = [
        "fault_start",
        "start_time",
        "start",
        "timestamp",
        "event_time",
        "fault_time",
        "datetime",
    ]

    lower_map = {
        str(col).lower(): col
        for col in df.columns
    }

    for candidate in candidates:

        if candidate in lower_map:

            return lower_map[candidate]

    for col in df.columns:

        name = str(col).lower()

        if (
            "start" in name
            and (
                "time" in name
                or "date" in name
            )
        ):

            return col

    raise ValueError(
        "No fault timestamp column found. "
        f"Available columns: {list(df.columns)}"
    )


# ============================================================
# LOAD FEATURES
# ============================================================

def load_features() -> pd.DataFrame:

    print(
        "[INFO] Loading multisource feature matrix:"
    )

    print(FEATURE_FILE)

    if not FEATURE_FILE.exists():

        raise FileNotFoundError(
            f"Feature file not found:\n{FEATURE_FILE}"
        )

    df = pd.read_csv(
        FEATURE_FILE,
        low_memory=False,
    )

    print(
        f"[INFO] Feature matrix shape: "
        f"{df.shape}"
    )

    timestamp_col = find_timestamp_column(
        df
    )

    print(
        f"[INFO] Timestamp column: "
        f"{timestamp_col}"
    )

    df["timestamp"] = convert_timestamp(
        df[timestamp_col]
    )

    df = df.dropna(
        subset=["timestamp"]
    )

    df = df.sort_values(
        "timestamp"
    )

    df = df.drop_duplicates(
        subset=["timestamp"],
        keep="first",
    )

    df = df.set_index(
        "timestamp"
    )

    return df


# ============================================================
# LOAD FAULT EVENTS
# ============================================================

def load_fault_events() -> pd.DataFrame:

    print(
        "\n[INFO] Loading MicroSS fault events:"
    )

    print(FAULT_FILE)

    if not FAULT_FILE.exists():

        raise FileNotFoundError(
            f"Fault file not found:\n{FAULT_FILE}"
        )

    faults = pd.read_csv(
        FAULT_FILE,
        low_memory=False,
    )

    print(
        f"[INFO] Fault event table shape: "
        f"{faults.shape}"
    )

    print("[INFO] Fault columns:")

    print(list(faults.columns))

    timestamp_col = find_fault_timestamp_column(
        faults
    )

    print(
        f"[INFO] Fault timestamp column: "
        f"{timestamp_col}"
    )

    faults[
        "fault_timestamp_original"
    ] = convert_timestamp(
        faults[timestamp_col]
    )

    faults = faults.dropna(
        subset=[
            "fault_timestamp_original"
        ]
    )

    faults = faults.sort_values(
        "fault_timestamp_original"
    )

    faults[
        "fault_timestamp_aligned"
    ] = (
        faults[
            "fault_timestamp_original"
        ]
        + FAULT_TIME_OFFSET
    )

    return faults


# ============================================================
# FILTER FAULTS
# ============================================================

def filter_faults_to_feature_range(
    faults: pd.DataFrame,
    feature_index: pd.DatetimeIndex,
) -> pd.DataFrame:

    feature_start = feature_index.min()

    feature_end = feature_index.max()

    print(
        "\n[INFO] Filtering aligned fault events "
        "to feature time range..."
    )

    print(
        "[INFO] Feature range:"
    )

    print(
        f"       {feature_start}"
    )

    print(
        f"       {feature_end}"
    )

    aligned = faults[
        "fault_timestamp_aligned"
    ]

    mask = (
        aligned >= feature_start
    ) & (
        aligned <= feature_end
    )

    filtered = faults.loc[
        mask
    ].copy()

    print(
        f"[INFO] Fault events before filtering: "
        f"{len(faults):,}"
    )

    print(
        f"[INFO] Fault events inside feature range: "
        f"{len(filtered):,}"
    )

    print(
        f"[INFO] Fault events outside feature range: "
        f"{len(faults) - len(filtered):,}"
    )

    if len(filtered) == 0:

        raise ValueError(
            "No aligned fault events are inside "
            "the feature time range."
        )

    return filtered


# ============================================================
# CREATE INCIDENT LABELS
# ============================================================

def create_incident_labels(
    feature_index: pd.DatetimeIndex,
    faults: pd.DataFrame,
) -> pd.DataFrame:

    print(
        "\n[INFO] Creating incident labels..."
    )

    labels = pd.DataFrame(
        index=feature_index
    )

    labels[
        "incident_now"
    ] = 0

    labels[
        "incident_next_5m"
    ] = 0

    labels[
        "incident_next_10m"
    ] = 0

    labels[
        "incident_next_15m"
    ] = 0

    labels[
        "incident_next_30m"
    ] = 0

    labels[
        "minutes_to_next_fault"
    ] = np.nan

    # --------------------------------------------------------
    # Get sorted unique fault timestamps
    # --------------------------------------------------------

    fault_times = (
        pd.Series(
            faults[
                "fault_timestamp_aligned"
            ]
        )
        .dropna()
        .sort_values()
        .drop_duplicates()
        .reset_index(
            drop=True
        )
    )

    print(
        f"[INFO] Unique aligned fault timestamps: "
        f"{len(fault_times):,}"
    )

    if fault_times.empty:

        warnings.warn(
            "No fault timestamps available."
        )

        return labels

    # --------------------------------------------------------
    # Convert to DatetimeIndex
    # --------------------------------------------------------

    fault_index = pd.DatetimeIndex(
        fault_times
    )

    feature_index = pd.DatetimeIndex(
        feature_index
    )

    # --------------------------------------------------------
    # INCIDENT NOW
    # --------------------------------------------------------
    #
    # Every fault is assigned to its 5-minute bucket.
    #
    # Example:
    #
    # 12:06:52
    #      ↓
    # 12:05:00
    #
    # --------------------------------------------------------

    fault_buckets = fault_index.floor(
        "5min"
    )

    fault_buckets = fault_buckets[
        fault_buckets.isin(
            feature_index
        )
    ]

    labels.loc[
        fault_buckets,
        "incident_now",
    ] = 1

    # --------------------------------------------------------
    # FUTURE FAULT
    # --------------------------------------------------------
    #
    # For every feature window t:
    #
    # next fault = first fault STRICTLY AFTER t
    #
    # We use searchsorted directly on timestamps.
    # --------------------------------------------------------

    next_positions = (
        fault_index.searchsorted(
            feature_index,
            side="right",
        )
    )

    has_next_fault = (
        next_positions
        < len(fault_index)
    )

    next_fault_times = pd.Series(
        pd.NaT,
        index=feature_index,
        dtype="datetime64[ns]",
    )

    valid_positions = (
        next_positions[
            has_next_fault
        ]
    )

    valid_feature_positions = np.where(
        has_next_fault
    )[0]

    if len(valid_positions) > 0:

        next_fault_times.iloc[
            valid_feature_positions
        ] = fault_index[
            valid_positions
        ]

    # --------------------------------------------------------
    # MINUTES TO NEXT FAULT
    # --------------------------------------------------------

    delta_minutes = (
        next_fault_times
        - pd.Series(
            feature_index,
            index=feature_index,
        )
    ).dt.total_seconds() / 60.0

    labels[
        "minutes_to_next_fault"
    ] = delta_minutes.values

    # --------------------------------------------------------
    # FUTURE HORIZONS
    # --------------------------------------------------------

    horizons = {
        "incident_next_5m": 5,
        "incident_next_10m": 10,
        "incident_next_15m": 15,
        "incident_next_30m": 30,
    }

    for column, minutes in horizons.items():

        labels[column] = (
            labels[
                "minutes_to_next_fault"
            ]
            > 0
        ) & (
            labels[
                "minutes_to_next_fault"
            ]
            <= minutes
        )

        labels[column] = (
            labels[column]
            .astype(int)
        )

    # --------------------------------------------------------
    # DIAGNOSTIC
    # --------------------------------------------------------

    print(
        "\n[INFO] Incident label counts:"
    )

    for column in [
        "incident_now",
        "incident_next_5m",
        "incident_next_10m",
        "incident_next_15m",
        "incident_next_30m",
    ]:

        count = int(
            labels[column].sum()
        )

        rate = (
            count / len(labels)
            if len(labels)
            else 0
        )

        print(
            f"       {column}: "
            f"{count:,} "
            f"({rate:.2%})"
        )

    # --------------------------------------------------------
    # DIAGNOSTIC: NEXT FAULT DISTRIBUTION
    # --------------------------------------------------------

    valid_delta = (
        labels[
            "minutes_to_next_fault"
        ]
        .dropna()
    )

    print(
        "\n[INFO] Minutes to next fault:"
    )

    if not valid_delta.empty:

        print(
            valid_delta.describe()
        )

        print(
            "\n[INFO] Windows with next fault "
            "within:"
        )

        for minutes in [
            1,
            2,
            5,
            10,
            15,
            30,
            60,
        ]:

            count = int(
                (
                    valid_delta
                    <= minutes
                ).sum()
            )

            print(
                f"       <= {minutes:>2} min: "
                f"{count:,}"
            )

    else:

        print(
            "       No future faults found."
        )

    # --------------------------------------------------------
    # DIAGNOSTIC: SHOW EXAMPLES
    # --------------------------------------------------------

    diagnostic = labels[
        labels[
            "minutes_to_next_fault"
        ].notna()
    ].copy()

    diagnostic = diagnostic[
        diagnostic[
            "minutes_to_next_fault"
        ] <= 30
    ].head(10)

    print(
        "\n[INFO] Example future incident windows:"
    )

    if not diagnostic.empty:

        print(
            diagnostic[
                [
                    "minutes_to_next_fault",
                    "incident_next_5m",
                    "incident_next_10m",
                    "incident_next_15m",
                    "incident_next_30m",
                ]
            ].to_string()
        )

    else:

        print(
            "       NONE FOUND."
        )

    # --------------------------------------------------------
    # CONSISTENCY CHECK
    # --------------------------------------------------------

    n5 = labels[
        "incident_next_5m"
    ].sum()

    n10 = labels[
        "incident_next_10m"
    ].sum()

    n15 = labels[
        "incident_next_15m"
    ].sum()

    n30 = labels[
        "incident_next_30m"
    ].sum()

    if not (
        n5 <= n10 <= n15 <= n30
    ):

        raise RuntimeError(
            "Invalid horizon labels: "
            "5m <= 10m <= 15m <= 30m "
            "is violated."
        )

    return labels


# ============================================================
# IDENTIFY METRIC COLUMNS
# ============================================================

def identify_metric_columns(
    df: pd.DataFrame,
) -> list[str]:

    metric_columns = []

    excluded = {
        "incident_now",
        "incident_next_5m",
        "incident_next_10m",
        "incident_next_15m",
        "incident_next_30m",
        "minutes_to_next_fault",
        "hour",
        "day_of_week",
        "day_of_month",
        "is_weekend",
    }

    for col in df.columns:

        name = str(col)

        if name in excluded:
            continue

        if (
            "__mean" in name
            or "__std" in name
            or "__min" in name
            or "__max" in name
            or "__count" in name
        ):

            metric_columns.append(
                name
            )

    return metric_columns


# ============================================================
# BUILD ANOMALY FEATURES
# ============================================================

def build_preincident_features(
    df: pd.DataFrame,
    labels: pd.DataFrame,
) -> pd.DataFrame:

    print(
        "\n[INFO] Building multisource anomaly indicators..."
    )

    result = labels.copy()

    metric_columns = identify_metric_columns(
        df
    )

    print(
        f"[INFO] Metric feature columns: "
        f"{len(metric_columns)}"
    )

    metric_data = df[
        metric_columns
    ].copy()

    metric_data = metric_data.apply(
        pd.to_numeric,
        errors="coerce",
    )

    # --------------------------------------------------------
    # Robust statistics
    # --------------------------------------------------------
    #
    # Exploratory analysis only.
    #
    # For ML, these statistics must be calculated using
    # training/past data only.
    # --------------------------------------------------------

    medians = metric_data.median(
        axis=0,
        skipna=True,
    )

    mads = (
        metric_data
        - medians
    ).abs().median(
        axis=0,
        skipna=True,
    )

    mads = mads.replace(
        0,
        np.nan,
    )

    robust_z = (
        metric_data
        - medians
    ).div(
        1.4826 * mads,
        axis="columns",
    )

    abs_robust_z = robust_z.abs()

    # --------------------------------------------------------
    # GLOBAL ANOMALIES
    # --------------------------------------------------------

    result[
        "multisource_anomaly_count"
    ] = (
        abs_robust_z > 3
    ).sum(
        axis=1
    )

    result[
        "multisource_anomaly_ratio"
    ] = (
        abs_robust_z > 3
    ).mean(
        axis=1
    )

    result[
        "multisource_high_anomaly_count"
    ] = (
        abs_robust_z > 5
    ).sum(
        axis=1
    )

    # --------------------------------------------------------
    # DOMAIN ANOMALIES
    # --------------------------------------------------------

    domains = {
        "network": [
            c
            for c in metric_columns
            if "network" in c.lower()
        ],
        "disk": [
            c
            for c in metric_columns
            if "diskio" in c.lower()
        ],
        "system": [
            c
            for c in metric_columns
            if "system_" in c.lower()
        ],
    }

    for domain, columns in domains.items():

        if not columns:
            continue

        domain_z = abs_robust_z[
            columns
        ]

        result[
            f"{domain}_anomaly_count"
        ] = (
            domain_z > 3
        ).sum(
            axis=1
        )

        result[
            f"{domain}_anomaly_ratio"
        ] = (
            domain_z > 3
        ).mean(
            axis=1
        )

    # --------------------------------------------------------
    # GLOBAL CHANGE
    # --------------------------------------------------------

    mean_columns = [
        c
        for c in metric_columns
        if "__mean" in c
    ]

    print(
        f"[INFO] Mean metric columns: "
        f"{len(mean_columns)}"
    )

    if mean_columns:

        current = metric_data[
            mean_columns
        ]

        previous = current.shift(
            1
        )

        pct_change = (
            current
            - previous
        ).div(
            previous.abs()
            + 1e-9
        )

        result[
            "global_abs_change_mean"
        ] = pct_change.abs().mean(
            axis=1
        )

        result[
            "global_abs_change_median"
        ] = pct_change.abs().median(
            axis=1
        )

        result[
            "global_large_change_count"
        ] = (
            pct_change.abs()
            > 0.50
        ).sum(
            axis=1
        )

    # --------------------------------------------------------
    # ANOMALY PRESSURE
    # --------------------------------------------------------

    result[
        "anomaly_pressure_15m"
    ] = (
        result[
            "multisource_anomaly_ratio"
        ]
        .rolling(
            window=3,
            min_periods=1,
        )
        .mean()
    )

    result[
        "anomaly_pressure_30m"
    ] = (
        result[
            "multisource_anomaly_ratio"
        ]
        .rolling(
            window=6,
            min_periods=1,
        )
        .mean()
    )

    result[
        "anomaly_pressure_change"
    ] = (
        result[
            "anomaly_pressure_15m"
        ]
        - result[
            "anomaly_pressure_15m"
        ].shift(1)
    )

    return result


# ============================================================
# SUMMARY
# ============================================================

def build_summary(
    analysis: pd.DataFrame,
    all_faults: pd.DataFrame,
    aligned_faults: pd.DataFrame,
) -> pd.DataFrame:

    print(
        "\n[INFO] Building incident analysis summary..."
    )

    rows = []

    total_windows = len(
        analysis
    )

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    rows.append(
        {
            "category": "dataset",
            "metric": "feature_windows",
            "value": total_windows,
        }
    )

    rows.append(
        {
            "category": "dataset",
            "metric": "fault_events_total",
            "value": len(all_faults),
        }
    )

    rows.append(
        {
            "category": "dataset",
            "metric": "fault_events_in_feature_range",
            "value": len(aligned_faults),
        }
    )

    rows.append(
        {
            "category": "dataset",
            "metric": "fault_events_outside_feature_range",
            "value": (
                len(all_faults)
                - len(aligned_faults)
            ),
        }
    )

    rows.append(
        {
            "category": "dataset",
            "metric": "unique_aligned_faults_in_feature_range",
            "value": aligned_faults[
                "fault_timestamp_aligned"
            ].nunique(),
        }
    )

    # --------------------------------------------------------
    # Alignment
    # --------------------------------------------------------

    rows.append(
        {
            "category": "alignment",
            "metric": "fault_offset_hours",
            "value": 1,
        }
    )

    rows.append(
        {
            "category": "alignment",
            "metric": "status",
            "value": "working hypothesis",
        }
    )

    # --------------------------------------------------------
    # Targets
    # --------------------------------------------------------

    targets = [
        "incident_now",
        "incident_next_5m",
        "incident_next_10m",
        "incident_next_15m",
        "incident_next_30m",
    ]

    for target in targets:

        positives = int(
            analysis[target].sum()
        )

        rate = (
            positives / total_windows
            if total_windows
            else np.nan
        )

        rows.append(
            {
                "category": "target",
                "metric": target,
                "value": positives,
            }
        )

        rows.append(
            {
                "category": "target_rate",
                "metric": target,
                "value": rate,
            }
        )

    # --------------------------------------------------------
    # Preincident comparison
    # --------------------------------------------------------

    incident_mask = (
        analysis[
            "incident_next_15m"
        ]
        == 1
    )

    nonincident_mask = (
        analysis[
            "incident_next_15m"
        ]
        == 0
    )

    comparison_columns = [
        "multisource_anomaly_count",
        "multisource_anomaly_ratio",
        "multisource_high_anomaly_count",
        "network_anomaly_count",
        "network_anomaly_ratio",
        "disk_anomaly_count",
        "disk_anomaly_ratio",
        "system_anomaly_count",
        "system_anomaly_ratio",
        "global_abs_change_mean",
        "global_abs_change_median",
        "global_large_change_count",
        "anomaly_pressure_15m",
        "anomaly_pressure_30m",
        "anomaly_pressure_change",
    ]

    for column in comparison_columns:

        if column not in analysis.columns:
            continue

        incident_mean = analysis.loc[
            incident_mask,
            column,
        ].mean()

        nonincident_mean = analysis.loc[
            nonincident_mask,
            column,
        ].mean()

        difference = (
            incident_mean
            - nonincident_mean
        )

        rows.append(
            {
                "category": "preincident_comparison",
                "metric": column,
                "value": difference,
                "incident_mean": incident_mean,
                "nonincident_mean": nonincident_mean,
            }
        )

    # --------------------------------------------------------
    # Data quality
    # --------------------------------------------------------

    rows.append(
        {
            "category": "data_quality",
            "metric": "average_missingness",
            "value": analysis.isna().mean().mean(),
        }
    )

    return pd.DataFrame(
        rows
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)

    print(
        "MICROSS MULTISOURCE INCIDENT ANALYSIS"
    )

    print("=" * 70)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # 1. Features
    # --------------------------------------------------------

    features = load_features()

    print(
        "[INFO] Feature time range:"
    )

    print(
        f"       {features.index.min()}"
    )

    print(
        f"       {features.index.max()}"
    )

    # --------------------------------------------------------
    # 2. Faults
    # --------------------------------------------------------

    all_faults = load_fault_events()

    print(
        "\n[INFO] Original fault range:"
    )

    print(
        f"       "
        f"{all_faults['fault_timestamp_original'].min()}"
    )

    print(
        f"       "
        f"{all_faults['fault_timestamp_original'].max()}"
    )

    print(
        "\n[INFO] Aligned fault range (+1h):"
    )

    print(
        f"       "
        f"{all_faults['fault_timestamp_aligned'].min()}"
    )

    print(
        f"       "
        f"{all_faults['fault_timestamp_aligned'].max()}"
    )

    # --------------------------------------------------------
    # 3. Filter
    # --------------------------------------------------------

    aligned_faults = (
        filter_faults_to_feature_range(
            all_faults,
            features.index,
        )
    )

    print(
        "\n[INFO] Filtered aligned fault range:"
    )

    print(
        f"       "
        f"{aligned_faults['fault_timestamp_aligned'].min()}"
    )

    print(
        f"       "
        f"{aligned_faults['fault_timestamp_aligned'].max()}"
    )

    # --------------------------------------------------------
    # 4. Labels
    # --------------------------------------------------------

    labels = create_incident_labels(
        features.index,
        aligned_faults,
    )

    # --------------------------------------------------------
    # 5. Anomaly features
    # --------------------------------------------------------

    analysis = build_preincident_features(
        features,
        labels,
    )

    # --------------------------------------------------------
    # 6. Temporal features
    # --------------------------------------------------------

    for column in [
        "hour",
        "day_of_week",
        "day_of_month",
        "is_weekend",
    ]:

        if column in features.columns:

            analysis[column] = features[
                column
            ]

    # --------------------------------------------------------
    # 7. Reset index
    # --------------------------------------------------------

    analysis = analysis.reset_index()

    columns = [
        "timestamp"
    ] + [
        c
        for c in analysis.columns
        if c != "timestamp"
    ]

    analysis = analysis[
        columns
    ]

    # --------------------------------------------------------
    # 8. Summary
    # --------------------------------------------------------

    summary = build_summary(
        analysis,
        all_faults,
        aligned_faults,
    )

    # --------------------------------------------------------
    # 9. Event labels
    # --------------------------------------------------------

    event_labels = analysis[
        [
            "timestamp",
            "incident_now",
            "incident_next_5m",
            "incident_next_10m",
            "incident_next_15m",
            "incident_next_30m",
            "minutes_to_next_fault",
        ]
    ].copy()

    # --------------------------------------------------------
    # 10. Save
    # --------------------------------------------------------

    print(
        "\n[INFO] Saving incident analysis..."
    )

    analysis.to_csv(
        ANALYSIS_FILE,
        index=False,
    )

    summary.to_csv(
        SUMMARY_FILE,
        index=False,
    )

    event_labels.to_csv(
        EVENT_FILE,
        index=False,
    )

    # --------------------------------------------------------
    # 11. Final report
    # --------------------------------------------------------

    print("\n" + "=" * 70)

    print(
        "MULTISOURCE INCIDENT ANALYSIS COMPLETE"
    )

    print("=" * 70)

    print(
        f"Feature windows:                  "
        f"{len(analysis):,}"
    )

    print(
        f"Fault events total:               "
        f"{len(all_faults):,}"
    )

    print(
        f"Fault events in feature range:    "
        f"{len(aligned_faults):,}"
    )

    print(
        f"Fault events outside range:       "
        f"{len(all_faults) - len(aligned_faults):,}"
    )

    print(
        f"Analysis columns:                 "
        f"{len(analysis.columns):,}"
    )

    for column in [
        "incident_now",
        "incident_next_5m",
        "incident_next_10m",
        "incident_next_15m",
        "incident_next_30m",
    ]:

        count = int(
            analysis[column].sum()
        )

        rate = (
            count / len(analysis)
            if len(analysis)
            else 0
        )

        print(
            f"{column:<34}"
            f"{count:>7,} "
            f"({rate:.2%})"
        )

    print(
        f"Average missingness:              "
        f"{analysis.isna().mean().mean():.2%}"
    )

    print(
        "\nWorking fault alignment hypothesis: "
        "+1 hour"
    )

    print(
        "WARNING: +1h is not proven timezone "
        "synchronization."
    )

    print(
        "\nAnalysis file:"
    )

    print(ANALYSIS_FILE)

    print(
        "\nSummary file:"
    )

    print(SUMMARY_FILE)

    print(
        "\nEvent labels file:"
    )

    print(EVENT_FILE)

    print("=" * 70)


if __name__ == "__main__":
    main()

