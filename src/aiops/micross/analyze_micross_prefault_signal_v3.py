
from pathlib import Path
import numpy as np
import pandas as pd


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

PREDICTION_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "micross"
    / "micross_prediction_dataset.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "micross"
)

OUTPUT_FILE = OUTPUT_DIR / "micross_prefault_signal_v3.csv"
SUMMARY_FILE = OUTPUT_DIR / "micross_prefault_signal_v3_summary.csv"
ANOMALY_RATE_FILE = OUTPUT_DIR / "micross_prefault_signal_v3_anomaly_rates.csv"
COVERAGE_FILE = OUTPUT_DIR / "micross_prefault_signal_v3_coverage.csv"


# Working hypothesis established during MicroSS alignment
TIMESTAMP_OFFSET_HOURS = 1


# Pre-fault windows
WINDOWS = {
    "T_minus_15m": (15, 10),
    "T_minus_10m": (10, 5),
    "T_minus_5m": (5, 0),
}


FEATURES = [
    "cpu_last",
    "memory_last",
    "cpu_pct_change_5m",
    "memory_pct_change_5m",
    "cpu_robust_z",
    "memory_robust_z",
    "cpu_is_anomaly",
    "memory_is_anomaly",
]


ANOMALY_Z_THRESHOLD = 3.0


# ============================================================
# HELPERS
# ============================================================

def mad(series):
    """
    Median Absolute Deviation.
    """
    series = pd.Series(series).dropna()

    if len(series) == 0:
        return np.nan

    median = series.median()

    return np.median(
        np.abs(series - median)
    )


def safe_mean(series):
    series = pd.Series(series).dropna()

    if len(series) == 0:
        return np.nan

    return series.mean()


def safe_median(series):
    series = pd.Series(series).dropna()

    if len(series) == 0:
        return np.nan

    return series.median()


def cliff_delta(x, y):
    """
    Cliff's Delta.

    Positive value:
        x tends to be larger than y.

    Negative value:
        x tends to be smaller than y.
    """

    x = pd.Series(x).dropna().to_numpy()
    y = pd.Series(y).dropna().to_numpy()

    if len(x) == 0 or len(y) == 0:
        return np.nan

    # Limit extremely large comparisons
    # to avoid unnecessary memory consumption.
    if len(x) * len(y) > 2_000_000:

        rng = np.random.default_rng(42)

        max_sample = 1000

        if len(x) > max_sample:
            x = rng.choice(
                x,
                size=max_sample,
                replace=False,
            )

        if len(y) > max_sample:
            y = rng.choice(
                y,
                size=max_sample,
                replace=False,
            )

    diff = x[:, None] - y[None, :]

    greater = np.sum(diff > 0)
    lower = np.sum(diff < 0)

    return (greater - lower) / (len(x) * len(y))


def normalize_timestamp_series(series):
    """
    Convert timestamps to:
        UTC
        then timezone-naive

    This guarantees that all comparisons use
    the same timestamp representation.
    """

    return (
        pd.to_datetime(
            series,
            errors="coerce",
            utc=True,
        )
        .dt
        .tz_localize(None)
    )


def normalize_timestamp_scalar(value):
    """
    Normalize one timestamp to UTC-naive.
    """

    ts = pd.to_datetime(
        value,
        errors="coerce",
        utc=True,
    )

    if pd.isna(ts):
        return pd.NaT

    return ts.tz_localize(None)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("=" * 70)
    print("MICROSS - PRE-FAULT SIGNAL ANALYSIS V3")
    print("=" * 70)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ========================================================
    # 1. LOAD FAULT EVENTS
    # ========================================================

    print("\n[1] Loading fault events...")

    if not FAULT_FILE.exists():
        raise FileNotFoundError(
            f"Fault file not found:\n{FAULT_FILE}"
        )

    faults = pd.read_csv(
        FAULT_FILE
    )

    print(
        f"Total fault rows: {len(faults):,}"
    )

    # --------------------------------------------------------
    # Normalize original timestamps
    # --------------------------------------------------------

    faults["fault_start"] = normalize_timestamp_series(
        faults["fault_start"]
    )

    faults["fault_end"] = normalize_timestamp_series(
        faults["fault_end"]
    )

    # --------------------------------------------------------
    # Remove suspicious duration events
    # --------------------------------------------------------

    if "duration_suspicious" in faults.columns:

        suspicious_mask = (
            faults["duration_suspicious"]
            .astype(bool)
        )

        print(
            "Suspicious duration rows excluded:",
            suspicious_mask.sum(),
        )

        faults = faults.loc[
            ~suspicious_mask
        ].copy()

    else:

        print(
            "Suspicious duration column not found."
        )

    # --------------------------------------------------------
    # Working +1 hour alignment
    # --------------------------------------------------------

    faults["aligned_fault_start"] = (
        faults["fault_start"]
        + pd.Timedelta(
            hours=TIMESTAMP_OFFSET_HOURS
        )
    )

    faults["aligned_fault_end"] = (
        faults["fault_end"]
        + pd.Timedelta(
            hours=TIMESTAMP_OFFSET_HOURS
        )
    )

    # Explicitly ensure timezone-naive
    faults["aligned_fault_start"] = (
        normalize_timestamp_series(
            faults["aligned_fault_start"]
        )
    )

    faults["aligned_fault_end"] = (
        normalize_timestamp_series(
            faults["aligned_fault_end"]
        )
    )

    faults = faults.dropna(
        subset=[
            "aligned_fault_start",
            "aligned_fault_end",
            "service",
        ]
    ).copy()

    print(
        f"Valid faults: {len(faults):,}"
    )

    print(
        f"Working timestamp offset: "
        f"+{TIMESTAMP_OFFSET_HOURS} hour"
    )

    # Give each fault a stable ID
    if "fault_id" not in faults.columns:

        faults["fault_id"] = np.arange(
            1,
            len(faults) + 1,
        )

    # --------------------------------------------------------
    # Timestamp sanity check
    # --------------------------------------------------------

    print("\nTimestamp normalization:")

    print(
        "  fault_start timezone:",
        faults["fault_start"].dt.tz,
    )

    print(
        "  aligned_fault_start timezone:",
        faults["aligned_fault_start"].dt.tz,
    )

    # ========================================================
    # 2. LOAD PREDICTION DATASET
    # ========================================================

    print("\n[2] Loading prediction dataset...")

    if not PREDICTION_FILE.exists():
        raise FileNotFoundError(
            f"Prediction dataset not found:\n"
            f"{PREDICTION_FILE}"
        )

    df = pd.read_csv(
        PREDICTION_FILE
    )

    # --------------------------------------------------------
    # Normalize prediction timestamps
    # --------------------------------------------------------

    df["timestamp"] = normalize_timestamp_series(
        df["timestamp"]
    )

    df["service"] = (
        df["service"]
        .astype(str)
        .str.strip()
    )

    print(
        f"Prediction rows: {len(df):,}"
    )

    print(
        f"Services: {df['service'].nunique()}"
    )

    print("\nFeatures used:")

    for feature in FEATURES:
        print(f"  - {feature}")

    # ========================================================
    # 3. VALIDATE REQUIRED COLUMNS
    # ========================================================

    required_fault_columns = [
        "fault_id",
        "service",
        "aligned_fault_start",
        "aligned_fault_end",
    ]

    required_data_columns = [
        "service",
        "timestamp",
    ] + FEATURES

    missing_fault = [
        c
        for c in required_fault_columns
        if c not in faults.columns
    ]

    missing_data = [
        c
        for c in required_data_columns
        if c not in df.columns
    ]

    if missing_fault:

        raise ValueError(
            "Missing fault columns:\n"
            + "\n".join(
                f"  - {c}"
                for c in missing_fault
            )
        )

    if missing_data:

        raise ValueError(
            "Missing prediction dataset columns:\n"
            + "\n".join(
                f"  - {c}"
                for c in missing_data
            )
        )

    # ========================================================
    # 4. SORT DATA
    # ========================================================

    df = df.sort_values(
        [
            "service",
            "timestamp",
        ]
    ).reset_index(
        drop=True
    )

    faults = faults.sort_values(
        [
            "service",
            "aligned_fault_start",
        ]
    ).reset_index(
        drop=True
    )

    # ========================================================
    # 5. BUILD FAULT EXCLUSION ZONES
    # ========================================================

    print("\n[3] Building fault exclusion zones...")

    exclusion_zones = []

    for _, row in faults.iterrows():

        start = row["aligned_fault_start"]
        end = row["aligned_fault_end"]

        exclusion_zones.append(
            {
                "fault_id": row["fault_id"],
                "service": row["service"],
                "start": start - pd.Timedelta(minutes=15),
                "end": end + pd.Timedelta(minutes=15),
            }
        )

    exclusion_df = pd.DataFrame(
        exclusion_zones
    )

    print(
        f"Fault exclusion zones: "
        f"{len(exclusion_df):,}"
    )

    # ========================================================
    # 6. PREPARE SERVICE INDEXES
    # ========================================================

    print("\n[4] Preparing service indexes...")

    service_groups = {}

    for service, group in df.groupby(
        "service",
        sort=False,
    ):

        service_groups[service] = {
            "timestamps": group[
                "timestamp"
            ].to_numpy(),

            "data": group,
        }

    print(
        f"Service indexes created: "
        f"{len(service_groups)}"
    )

    # ========================================================
    # 7. BUILD PRE-FAULT WINDOWS
    # ========================================================

    print("\n[5] Building pre-fault windows...")

    prefault_records = []

    coverage_records = []

    total_faults = len(faults)

    for fault_index, fault in faults.iterrows():

        fault_id = fault["fault_id"]
        service = fault["service"]

        fault_start = fault[
            "aligned_fault_start"
        ]

        fault_end = fault[
            "aligned_fault_end"
        ]

        if service not in service_groups:

            coverage_records.append(
                {
                    "fault_id": fault_id,
                    "service": service,
                    "fault_start": fault_start,
                    "has_pre_15m": False,
                    "has_pre_10m": False,
                    "has_pre_5m": False,
                }
            )

            continue

        service_df = service_groups[
            service
        ]["data"]

        service_min = service_df[
            "timestamp"
        ].min()

        service_max = service_df[
            "timestamp"
        ].max()

        has_pre_15 = False
        has_pre_10 = False
        has_pre_5 = False

        # ----------------------------------------------------
        # Check if enough historical data exists
        # ----------------------------------------------------

        for window_name, (
            minutes_before,
            minutes_after,
        ) in WINDOWS.items():

            window_start = (
                fault_start
                - pd.Timedelta(
                    minutes=minutes_before
                )
            )

            window_end = (
                fault_start
                - pd.Timedelta(
                    minutes=minutes_after
                )
            )

            # No data possible
            if window_end < service_min:

                continue

            if window_start > service_max:

                continue

            mask = (
                (service_df["timestamp"] >= window_start)
                &
                (service_df["timestamp"] < window_end)
            )

            window_df = service_df.loc[
                mask,
                [
                    "timestamp"
                ] + FEATURES,
            ].copy()

            if window_df.empty:

                continue

            if window_name == "T_minus_15m":
                has_pre_15 = True

            elif window_name == "T_minus_10m":
                has_pre_10 = True

            elif window_name == "T_minus_5m":
                has_pre_5 = True

            # ------------------------------------------------
            # Build aggregate record
            # ------------------------------------------------

            record = {
                "fault_id": fault_id,
                "service": service,
                "fault_type": fault.get(
                    "fault_type",
                    "unknown",
                ),
                "fault_start": fault_start,
                "fault_end": fault_end,
                "window": window_name,
                "window_start": window_start,
                "window_end": window_end,
                "n_rows": len(window_df),
            }

            for feature in FEATURES:

                values = pd.to_numeric(
                    window_df[feature],
                    errors="coerce",
                ).dropna()

                record[
                    f"{feature}_mean"
                ] = safe_mean(values)

                record[
                    f"{feature}_median"
                ] = safe_median(values)

                record[
                    f"{feature}_mad"
                ] = mad(values)

            # ------------------------------------------------
            # CPU anomaly rate
            # ------------------------------------------------

            cpu_z = pd.to_numeric(
                window_df["cpu_robust_z"],
                errors="coerce",
            )

            memory_z = pd.to_numeric(
                window_df["memory_robust_z"],
                errors="coerce",
            )

            cpu_valid = cpu_z.notna()

            memory_valid = memory_z.notna()

            if cpu_valid.sum() > 0:

                record[
                    "cpu_z_anomaly_rate"
                ] = (
                    (
                        cpu_z.loc[
                            cpu_valid
                        ].abs()
                        >= ANOMALY_Z_THRESHOLD
                    )
                    .mean()
                )

            else:

                record[
                    "cpu_z_anomaly_rate"
                ] = np.nan

            if memory_valid.sum() > 0:

                record[
                    "memory_z_anomaly_rate"
                ] = (
                    (
                        memory_z.loc[
                            memory_valid
                        ].abs()
                        >= ANOMALY_Z_THRESHOLD
                    )
                    .mean()
                )

            else:

                record[
                    "memory_z_anomaly_rate"
                ] = np.nan

            # Existing anomaly flags
            cpu_flag = pd.to_numeric(
                window_df["cpu_is_anomaly"],
                errors="coerce",
            )

            memory_flag = pd.to_numeric(
                window_df["memory_is_anomaly"],
                errors="coerce",
            )

            record[
                "cpu_flag_anomaly_rate"
            ] = cpu_flag.mean()

            record[
                "memory_flag_anomaly_rate"
            ] = memory_flag.mean()

            prefault_records.append(
                record
            )

        coverage_records.append(
            {
                "fault_id": fault_id,
                "service": service,
                "fault_start": fault_start,
                "fault_end": fault_end,
                "has_pre_15m": has_pre_15,
                "has_pre_10m": has_pre_10,
                "has_pre_5m": has_pre_5,
            }
        )

        if (
            (fault_index + 1) % 250 == 0
            or fault_index == total_faults - 1
        ):

            print(
                f"  Processed "
                f"{fault_index + 1:,}/"
                f"{total_faults:,} faults"
            )

    prefault_df = pd.DataFrame(
        prefault_records
    )

    coverage_df = pd.DataFrame(
        coverage_records
    )

    print(
        "\nPre-fault records:",
        f"{len(prefault_df):,}",
    )

    # ========================================================
    # 8. CHECK COVERAGE
    # ========================================================

    print("\n[6] Pre-fault coverage")

    if not coverage_df.empty:

        for col in [
            "has_pre_15m",
            "has_pre_10m",
            "has_pre_5m",
        ]:

            count = coverage_df[
                col
            ].sum()

            pct = (
                count
                / len(coverage_df)
                * 100
            )

            print(
                f"{col:15s}: "
                f"{count:4d}/"
                f"{len(coverage_df):4d} "
                f"({pct:6.2f}%)"
            )

    if prefault_df.empty:

        print(
            "\nWARNING: No pre-fault records found."
        )

        print(
            "Check timestamp alignment and metric coverage."
        )

        coverage_df.to_csv(
            COVERAGE_FILE,
            index=False,
        )

        print(
            "\nCoverage saved to:"
        )

        print(
            COVERAGE_FILE
        )

        raise SystemExit(0)

    # ========================================================
    # 9. BUILD MATCHED CONTROL WINDOWS
    # ========================================================

    print("\n[7] Building matched control windows...")

    control_records = []

    rng = np.random.default_rng(42)

    # --------------------------------------------------------
    # Prepare fault intervals per service
    # --------------------------------------------------------

    faults_by_service = {}

    for service, group in faults.groupby(
        "service",
        sort=False,
    ):

        starts = group[
            "aligned_fault_start"
        ].sort_values().to_numpy()

        ends = group[
            "aligned_fault_end"
        ].sort_values().to_numpy()

        faults_by_service[service] = (
            starts,
            ends,
        )

    # --------------------------------------------------------
    # Candidate controls
    #
    # For each fault, choose control windows from the
    # prediction dataset:
    #
    #   same service
    #   same approximate time-of-day
    #   not near another fault
    #   at least 60 min away from target fault
    #
    # Maximum 3 controls.
    # --------------------------------------------------------

    for fault_index, fault in faults.iterrows():

        fault_id = fault["fault_id"]
        service = fault["service"]
        fault_start = fault[
            "aligned_fault_start"
        ]

        if service not in service_groups:
            continue

        service_df = service_groups[
            service
        ]["data"]

        # Need enough history
        service_min = service_df[
            "timestamp"
        ].min()

        if fault_start - service_min < pd.Timedelta(
            minutes=60
        ):
            continue

        # ----------------------------------------------------
        # Same time-of-day ±30 minutes
        # ----------------------------------------------------

        candidate = service_df.copy()

        candidate["time_diff_minutes"] = (
            (
                (
                    candidate["timestamp"].dt.hour
                    * 60
                    + candidate["timestamp"].dt.minute
                )
                -
                (
                    fault_start.hour * 60
                    + fault_start.minute
                )
            )
            .abs()
        )

        # Circular time difference
        candidate["time_diff_minutes"] = np.minimum(
            candidate["time_diff_minutes"],
            1440
            - candidate["time_diff_minutes"],
        )

        candidate = candidate.loc[
            candidate[
                "time_diff_minutes"
            ] <= 30
        ].copy()

        # ----------------------------------------------------
        # At least 60 minutes from target fault
        # ----------------------------------------------------

        candidate = candidate.loc[
            (
                candidate["timestamp"]
                - fault_start
            ).abs()
            >= pd.Timedelta(
                minutes=60
            )
        ].copy()

        # ----------------------------------------------------
        # Exclude points near ANY fault
        # for the same service
        # ----------------------------------------------------

        starts, ends = faults_by_service[
            service
        ]

        timestamps = (
            candidate["timestamp"]
            .to_numpy()
        )

        invalid = np.zeros(
            len(candidate),
            dtype=bool,
        )

        # Efficient interval check using searchsorted.
        # A candidate is near a fault if:
        #
        # fault_start - 15m <= timestamp
        # <= fault_end + 15m
        #
        expanded_starts = (
            starts
            - np.timedelta64(
                15,
                "m",
            )
        )

        expanded_ends = (
            ends
            + np.timedelta64(
                15,
                "m",
            )
        )

        positions = np.searchsorted(
            expanded_starts,
            timestamps,
            side="right",
        ) - 1

        valid_positions = (
            positions >= 0
        )

        idx = np.where(
            valid_positions
        )[0]

        if len(idx) > 0:

            p = positions[idx]

            invalid[idx] = (
                timestamps[idx]
                <= expanded_ends[p]
            )

        candidate = candidate.loc[
            ~invalid
        ].copy()

        if candidate.empty:
            continue

        # ----------------------------------------------------
        # Select maximum 3 controls
        # on different dates where possible.
        # ----------------------------------------------------

        candidate["date"] = (
            candidate["timestamp"].dt.date
        )

        candidate = candidate.sort_values(
            [
                "time_diff_minutes",
                "timestamp",
            ]
        )

        selected = []

        used_dates = set()

        for _, control in candidate.iterrows():

            control_date = control[
                "timestamp"
            ].date()

            if control_date in used_dates:
                continue

            selected.append(
                control
            )

            used_dates.add(
                control_date
            )

            if len(selected) >= 3:
                break

        for control in selected:

            control_ts = control[
                "timestamp"
            ]

            # Use the same T-minus-5m style
            # anchor structure as the primary
            # short pre-fault analysis.
            control_window_start = (
                control_ts
                - pd.Timedelta(
                    minutes=5
                )
            )

            control_window_end = (
                control_ts
            )

            control_mask = (
                (
                    service_df["timestamp"]
                    >= control_window_start
                )
                &
                (
                    service_df["timestamp"]
                    < control_window_end
                )
            )

            control_df = service_df.loc[
                control_mask,
                [
                    "timestamp"
                ] + FEATURES,
            ].copy()

            if control_df.empty:
                continue

            record = {
                "fault_id": fault_id,
                "service": service,
                "fault_type": fault.get(
                    "fault_type",
                    "unknown",
                ),
                "window": "CONTROL_5m",
                "control_timestamp": control_ts,
                "window_start": control_window_start,
                "window_end": control_window_end,
                "n_rows": len(control_df),
            }

            for feature in FEATURES:

                values = pd.to_numeric(
                    control_df[feature],
                    errors="coerce",
                ).dropna()

                record[
                    f"{feature}_mean"
                ] = safe_mean(values)

                record[
                    f"{feature}_median"
                ] = safe_median(values)

                record[
                    f"{feature}_mad"
                ] = mad(values)

            cpu_z = pd.to_numeric(
                control_df["cpu_robust_z"],
                errors="coerce",
            )

            memory_z = pd.to_numeric(
                control_df["memory_robust_z"],
                errors="coerce",
            )

            if cpu_z.notna().sum() > 0:

                record[
                    "cpu_z_anomaly_rate"
                ] = (
                    (
                        cpu_z.dropna().abs()
                        >= ANOMALY_Z_THRESHOLD
                    )
                    .mean()
                )

            else:

                record[
                    "cpu_z_anomaly_rate"
                ] = np.nan

            if memory_z.notna().sum() > 0:

                record[
                    "memory_z_anomaly_rate"
                ] = (
                    (
                        memory_z.dropna().abs()
                        >= ANOMALY_Z_THRESHOLD
                    )
                    .mean()
                )

            else:

                record[
                    "memory_z_anomaly_rate"
                ] = np.nan

            cpu_flag = pd.to_numeric(
                control_df["cpu_is_anomaly"],
                errors="coerce",
            )

            memory_flag = pd.to_numeric(
                control_df["memory_is_anomaly"],
                errors="coerce",
            )

            record[
                "cpu_flag_anomaly_rate"
            ] = cpu_flag.mean()

            record[
                "memory_flag_anomaly_rate"
            ] = memory_flag.mean()

            control_records.append(
                record
            )

    control_df_final = pd.DataFrame(
        control_records
    )

    print(
        "Control records:",
        f"{len(control_df_final):,}",
    )

    # ========================================================
    # 10. SAVE RAW ANALYSIS DATA
    # ========================================================

    print("\n[8] Combining analysis records...")

    if not prefault_df.empty:

        prefault_df[
            "sample_type"
        ] = "PRE_FAULT"

    if not control_df_final.empty:

        control_df_final[
            "sample_type"
        ] = "CONTROL"

    combined = pd.concat(
        [
            prefault_df,
            control_df_final,
        ],
        ignore_index=True,
        sort=False,
    )

    combined.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print(
        "Analysis file saved:"
    )

    print(
        OUTPUT_FILE
    )

    # ========================================================
    # 11. PRE-FAULT SUMMARY
    # ========================================================

    print("\n[9] Building statistical summary...")

    summary_records = []

    feature_summary_columns = []

    for feature in FEATURES:

        feature_summary_columns.extend(
            [
                f"{feature}_mean",
                f"{feature}_median",
                f"{feature}_mad",
            ]
        )

    # --------------------------------------------------------
    # Compare each pre-fault window against controls
    # --------------------------------------------------------

    for window_name in WINDOWS.keys():

        pre = combined.loc[
            (
                combined["sample_type"]
                == "PRE_FAULT"
            )
            &
            (
                combined["window"]
                == window_name
            )
        ].copy()

        if pre.empty:
            continue

        control = combined.loc[
            combined["sample_type"]
            == "CONTROL"
        ].copy()

        for feature in FEATURES:

            value_col = (
                f"{feature}_median"
            )

            if value_col not in pre.columns:
                continue

            x = pd.to_numeric(
                pre[value_col],
                errors="coerce",
            ).dropna()

            y = pd.to_numeric(
                control[value_col],
                errors="coerce",
            ).dropna()

            if len(x) == 0 or len(y) == 0:
                continue

            delta = cliff_delta(
                x,
                y,
            )

            summary_records.append(
                {
                    "window": window_name,
                    "feature": feature,
                    "pre_fault_n": len(x),
                    "control_n": len(y),
                    "pre_fault_mean": x.mean(),
                    "control_mean": y.mean(),
                    "pre_fault_median": x.median(),
                    "control_median": y.median(),
                    "pre_fault_mad": mad(x),
                    "control_mad": mad(y),
                    "cliffs_delta": delta,
                    "absolute_effect": abs(delta),
                }
            )

    summary_df = pd.DataFrame(
        summary_records
    )

    summary_df.to_csv(
        SUMMARY_FILE,
        index=False,
    )

    print(
        "Summary saved:"
    )

    print(
        SUMMARY_FILE
    )

    # ========================================================
    # 12. ANOMALY RATE ANALYSIS
    # ========================================================

    print("\n[10] Building anomaly-rate analysis...")

    anomaly_records = []

    for window_name in WINDOWS.keys():

        pre = combined.loc[
            (
                combined["sample_type"]
                == "PRE_FAULT"
            )
            &
            (
                combined["window"]
                == window_name
            )
        ]

        if pre.empty:
            continue

        control = combined.loc[
            combined["sample_type"]
            == "CONTROL"
        ]

        # CPU
        pre_cpu = pd.to_numeric(
            pre["cpu_z_anomaly_rate"],
            errors="coerce",
        ).dropna()

        ctrl_cpu = pd.to_numeric(
            control["cpu_z_anomaly_rate"],
            errors="coerce",
        ).dropna()

        # Memory
        pre_mem = pd.to_numeric(
            pre["memory_z_anomaly_rate"],
            errors="coerce",
        ).dropna()

        ctrl_mem = pd.to_numeric(
            control["memory_z_anomaly_rate"],
            errors="coerce",
        ).dropna()

        anomaly_records.append(
            {
                "window": window_name,
                "cpu_pre_fault_n": len(pre_cpu),
                "cpu_control_n": len(ctrl_cpu),
                "cpu_pre_fault_anomaly_rate_mean":
                    pre_cpu.mean()
                    if len(pre_cpu)
                    else np.nan,
                "cpu_control_anomaly_rate_mean":
                    ctrl_cpu.mean()
                    if len(ctrl_cpu)
                    else np.nan,
                "cpu_pre_fault_anomaly_rate_median":
                    pre_cpu.median()
                    if len(pre_cpu)
                    else np.nan,
                "cpu_control_anomaly_rate_median":
                    ctrl_cpu.median()
                    if len(ctrl_cpu)
                    else np.nan,
                "memory_pre_fault_n": len(pre_mem),
                "memory_control_n": len(ctrl_mem),
                "memory_pre_fault_anomaly_rate_mean":
                    pre_mem.mean()
                    if len(pre_mem)
                    else np.nan,
                "memory_control_anomaly_rate_mean":
                    ctrl_mem.mean()
                    if len(ctrl_mem)
                    else np.nan,
                "memory_pre_fault_anomaly_rate_median":
                    pre_mem.median()
                    if len(pre_mem)
                    else np.nan,
                "memory_control_anomaly_rate_median":
                    ctrl_mem.median()
                    if len(ctrl_mem)
                    else np.nan,
            }
        )

    anomaly_rate_df = pd.DataFrame(
        anomaly_records
    )

    anomaly_rate_df.to_csv(
        ANOMALY_RATE_FILE,
        index=False,
    )

    print(
        "Anomaly-rate file saved:"
    )

    print(
        ANOMALY_RATE_FILE
    )

    # ========================================================
    # 13. SAVE COVERAGE
    # ========================================================

    coverage_df.to_csv(
        COVERAGE_FILE,
        index=False,
    )

    print(
        "Coverage file saved:"
    )

    print(
        COVERAGE_FILE
    )

    # ========================================================
    # 14. TERMINAL SUMMARY
    # ========================================================

    print("\n" + "=" * 70)
    print("PRE-FAULT ANALYSIS COMPLETE")
    print("=" * 70)

    print(
        f"\nValid faults analyzed : "
        f"{len(faults):,}"
    )

    print(
        f"Pre-fault records     : "
        f"{len(prefault_df):,}"
    )

    print(
        f"Control records       : "
        f"{len(control_df_final):,}"
    )

    print(
        f"Working offset        : "
        f"+{TIMESTAMP_OFFSET_HOURS} hour"
    )

    print("\nPRE-FAULT COVERAGE")

    if not coverage_df.empty:

        for col in [
            "has_pre_15m",
            "has_pre_10m",
            "has_pre_5m",
        ]:

            count = coverage_df[
                col
            ].sum()

            pct = (
                count
                / len(coverage_df)
                * 100
            )

            print(
                f"{col:15s}: "
                f"{count:4d}/"
                f"{len(coverage_df):4d} "
                f"({pct:6.2f}%)"
            )

    print("\nANOMALY RATE COMPARISON")

    if not anomaly_rate_df.empty:

        display_cols = [
            "window",
            "cpu_pre_fault_anomaly_rate_mean",
            "cpu_control_anomaly_rate_mean",
            "memory_pre_fault_anomaly_rate_mean",
            "memory_control_anomaly_rate_mean",
        ]

        print(
            anomaly_rate_df[
                display_cols
            ].to_string(
                index=False
            )
        )

    print("\nSTRONGEST EFFECTS")

    if not summary_df.empty:

        strongest = (
            summary_df
            .sort_values(
                "absolute_effect",
                ascending=False,
            )
            .head(15)
        )

        print(
            strongest[
                [
                    "window",
                    "feature",
                    "pre_fault_n",
                    "control_n",
                    "pre_fault_median",
                    "control_median",
                    "cliffs_delta",
                ]
            ].to_string(
                index=False
            )
        )

    print("\nOUTPUT FILES")

    print(
        f"  1. {OUTPUT_FILE}"
    )

    print(
        f"  2. {SUMMARY_FILE}"
    )

    print(
        f"  3. {ANOMALY_RATE_FILE}"
    )

    print(
        f"  4. {COVERAGE_FILE}"
    )

    print("\nIMPORTANT METHODOLOGICAL NOTES")

    print(
        "  - +1h is a working timestamp-alignment hypothesis."
    )

    print(
        "  - Fault events with suspicious durations are excluded."
    )

    print(
        "  - Pre-fault windows contain only past observations."
    )

    print(
        "  - Control windows are selected outside fault exclusion zones."
    )

    print(
        "  - This analysis evaluates signal availability,"
    )

    print(
        "    not predictive-model accuracy."
    )

    print(
        "  - Robust-z features in the prediction dataset are"
    )

    print(
        "    assumed to have been computed using past-only information."
    )

    print("=" * 70)

