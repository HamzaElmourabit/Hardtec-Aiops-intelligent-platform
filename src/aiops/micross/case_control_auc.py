
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]

FEATURE_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "micross"
    / "micross_multisource_features_5m.csv"
)

FAULT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "micross"
    / "micross_fault_events.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "micross"
)

OUTPUT_FILE = (
    OUTPUT_DIR
    / "micross_case_control_auc.csv"
)

TOP_FILE = (
    OUTPUT_DIR
    / "micross_top_auc_signals.csv"
)


# ============================================================
# PARAMETERS
# ============================================================

OFFSET_HOURS = 1

# Horizons before the fault.
HORIZONS = {
    "T_minus_15m": 15,
    "T_minus_10m": 10,
    "T_minus_5m": 5,
}

# A control observation must be at least this far from a fault.
CONTROL_EXCLUSION_MINUTES = 15

# Minimum number of valid observations.
MIN_OBS = 20

# Keep the best signals.
TOP_N = 50


# ============================================================
# HEADER
# ============================================================

print()
print("=" * 75)
print("MICROSS - CASE-CONTROL + ROC-AUC")
print("=" * 75)


# ============================================================
# LOAD FAULTS
# ============================================================

print()
print("Loading faults...")

faults = pd.read_csv(
    FAULT_FILE
)

print(
    f"Raw faults: {len(faults)}"
)


# ------------------------------------------------------------
# Remove suspicious duration faults.
# ------------------------------------------------------------

if "duration_suspicious" in faults.columns:

    faults = faults[
        faults["duration_suspicious"]
        .fillna(False)
        == False
    ].copy()


# ------------------------------------------------------------
# Find start column.
# ------------------------------------------------------------

if "fault_start" not in faults.columns:

    possible = [
        "start",
        "timestamp",
    ]

    start_col = None

    for col in possible:

        if col in faults.columns:

            start_col = col
            break

    if start_col is None:

        raise RuntimeError(
            "Impossible de trouver fault_start."
        )

    faults["fault_start"] = faults[start_col]


# ------------------------------------------------------------
# Timestamp.
# ------------------------------------------------------------

faults["fault_start"] = pd.to_datetime(
    faults["fault_start"],
    errors="coerce",
    utc=True,
)


faults = faults.dropna(
    subset=["fault_start"]
).copy()


# ------------------------------------------------------------
# Apply +1h working alignment.
# ------------------------------------------------------------

faults["aligned_fault_start"] = (
    faults["fault_start"]
    + pd.Timedelta(
        hours=OFFSET_HOURS
    )
)


# ------------------------------------------------------------
# Service.
# ------------------------------------------------------------

faults["service"] = (
    faults["service"]
    .astype(str)
    .str.strip()
)


faults = faults.sort_values(
    "aligned_fault_start"
).reset_index(
    drop=True
)


print(
    f"Valid faults: {len(faults)}"
)

print(
    f"Working offset: +{OFFSET_HOURS}h"
)


# ============================================================
# UNIQUE FAULT WINDOWS
# ============================================================

faults["fault_window"] = (
    faults["aligned_fault_start"]
    .dt.floor("5min")
)


fault_events = (
    faults[
        [
            "service",
            "aligned_fault_start",
            "fault_window",
        ]
    ]
    .drop_duplicates(
        subset=[
            "service",
            "fault_window",
        ]
    )
    .copy()
)


print(
    f"Unique service/fault windows: {len(fault_events)}"
)


# ============================================================
# LOAD MULTI-SOURCE FEATURES
# ============================================================

print()
print("Loading feature dataset...")

df = pd.read_csv(
    FEATURE_FILE
)

print(
    f"Feature dataset shape: {df.shape}"
)


df["timestamp"] = pd.to_datetime(
    df["timestamp"],
    errors="coerce",
    utc=True,
)


df = df.dropna(
    subset=["timestamp"]
).copy()


df["service"] = (
    df["service"]
    .astype(str)
    .str.strip()
)


df = df.sort_values(
    [
        "service",
        "timestamp",
    ]
).reset_index(
    drop=True
)


# ============================================================
# SELECT RAW SIGNAL FEATURES
# ============================================================

signal_columns = []

for col in df.columns:

    if col in {
        "timestamp",
        "service",
    }:

        continue


    if not pd.api.types.is_numeric_dtype(
        df[col]
    ):

        continue


    # --------------------------------------------------------
    # IMPORTANT:
    #
    # We use raw level, delta and percentage change.
    #
    # We DO NOT use robust_z because the previous analysis
    # demonstrated that full-series robust-z can explode when
    # the MAD is close to zero.
    # --------------------------------------------------------

    if (
        "__last" in col
        or "__delta" in col
        or "__pct_change" in col
    ):

        signal_columns.append(
            col
        )


print(
    f"Candidate signals: {len(signal_columns)}"
)


# ============================================================
# CREATE FAULT LABELS
# ============================================================

print()
print("Creating fault/control labels...")


# ------------------------------------------------------------
# A timestamp is a CONTROL candidate if there is no fault
# within +/- 15 minutes for the same service.
# ------------------------------------------------------------

df["near_fault"] = False


for service in df["service"].unique():

    service_faults = (
        fault_events.loc[
            fault_events["service"] == service,
            "fault_window",
        ]
        .sort_values()
        .values
    )


    if len(service_faults) == 0:

        continue


    mask = (
        df["service"]
        == service
    )


    timestamps = (
        df.loc[
            mask,
            "timestamp"
        ]
        .values
        .astype(
            "datetime64[ns]"
        )
    )


    fault_times = (
        service_faults
        .astype(
            "datetime64[ns]"
        )
    )


    positions = np.searchsorted(
        fault_times,
        timestamps,
        side="left",
    )


    left_idx = np.clip(
        positions - 1,
        0,
        len(fault_times) - 1,
    )


    right_idx = np.clip(
        positions,
        0,
        len(fault_times) - 1,
    )


    left_distance = np.abs(
        timestamps
        - fault_times[left_idx]
    )


    right_distance = np.abs(
        timestamps
        - fault_times[right_idx]
    )


    minimum_distance = np.minimum(
        left_distance,
        right_distance,
    )


    excluded = (
        minimum_distance
        <= np.timedelta64(
            CONTROL_EXCLUSION_MINUTES,
            "m",
        )
    )


    df.loc[
        mask,
        "near_fault"
    ] = excluded


normal_df = df[
    ~df["near_fault"]
].copy()


print(
    f"Normal control rows: {len(normal_df)}"
)

print(
    f"Excluded rows: {df['near_fault'].sum()}"
)


# ============================================================
# BUILD PRE-FAULT SAMPLE
# ============================================================

print()
print("Building pre-fault samples...")


pre_tables = []


for horizon_name, minutes in HORIZONS.items():

    temp = fault_events[
        [
            "service",
            "aligned_fault_start",
            "fault_window",
        ]
    ].copy()


    temp["timestamp"] = (
        temp["fault_window"]
        - pd.Timedelta(
            minutes=minutes
        )
    )


    temp["horizon"] = (
        horizon_name
    )


    # --------------------------------------------------------
    # Exact join on the 5-minute grid.
    # --------------------------------------------------------

    merged = temp.merge(
        df,
        on=[
            "service",
            "timestamp",
        ],
        how="left",
    )


    pre_tables.append(
        merged
    )


pre_df = pd.concat(
    pre_tables,
    ignore_index=True
)


print(
    f"Pre-fault rows: {len(pre_df)}"
)


# ============================================================
# ROC-AUC ANALYSIS
# ============================================================

print()
print("=" * 75)
print("RUNNING CASE-CONTROL ROC-AUC")
print("=" * 75)


results = []


# ------------------------------------------------------------
# We calculate AUC for every signal and every horizon.
#
# AUC interpretation:
#
# 0.50 = random
# >0.50 = higher values tend to indicate pre-fault
# <0.50 = lower values tend to indicate pre-fault
#
# We use:
#
# discriminative_auc = max(AUC, 1-AUC)
#
# because a predictive signal can increase OR decrease.
# ------------------------------------------------------------


for index, feature in enumerate(
    signal_columns,
    start=1
):

    if index % 25 == 0:

        print(
            f"  processed {index}/{len(signal_columns)}"
        )


    normal_values = pd.to_numeric(
        normal_df[feature],
        errors="coerce",
    )


    normal_values = normal_values.replace(
        [
            np.inf,
            -np.inf,
        ],
        np.nan,
    )


    # --------------------------------------------------------
    # Sample normal observations.
    #
    # Limit to 20,000 for speed while retaining enough
    # observations for stable AUC estimation.
    # --------------------------------------------------------

    normal_values = (
        normal_values
        .dropna()
    )


    if len(normal_values) > 20000:

        normal_values = (
            normal_values
            .sample(
                20000,
                random_state=42,
            )
        )


    if len(normal_values) < MIN_OBS:

        continue


    normal_median = (
        normal_values
        .median()
    )


    normal_q25 = (
        normal_values
        .quantile(0.25)
    )


    normal_q75 = (
        normal_values
        .quantile(0.75)
    )


    normal_iqr = (
        normal_q75
        - normal_q25
    )


    if (
        not np.isfinite(
            normal_iqr
        )
        or normal_iqr <= 1e-12
    ):

        normal_iqr = np.nan


    result = {
        "feature": feature,
        "normal_n": len(normal_values),
        "normal_median": normal_median,
        "normal_iqr": normal_iqr,
    }


    horizon_auc = []


    for horizon_name in HORIZONS:

        fault_values = pd.to_numeric(
            pre_df.loc[
                pre_df["horizon"]
                == horizon_name,
                feature,
            ],
            errors="coerce",
        )


        fault_values = fault_values.replace(
            [
                np.inf,
                -np.inf,
            ],
            np.nan,
        ).dropna()


        if len(fault_values) < MIN_OBS:

            result[
                f"{horizon_name}_n"
            ] = len(fault_values)

            result[
                f"{horizon_name}_auc"
            ] = np.nan

            result[
                f"{horizon_name}_discriminative_auc"
            ] = np.nan

            result[
                f"{horizon_name}_median"
            ] = np.nan

            result[
                f"{horizon_name}_median_shift"
            ] = np.nan

            continue


        # ----------------------------------------------------
        # To avoid massive arrays, sample the fault group if
        # necessary.
        # ----------------------------------------------------

        if len(fault_values) > 20000:

            fault_values = (
                fault_values
                .sample(
                    20000,
                    random_state=42,
                )
            )


        y = np.concatenate(
            [
                np.zeros(
                    len(normal_values)
                ),
                np.ones(
                    len(fault_values)
                ),
            ]
        )


        x = np.concatenate(
            [
                normal_values.to_numpy(),
                fault_values.to_numpy(),
            ]
        )


        # ----------------------------------------------------
        # Remove any remaining invalid values.
        # ----------------------------------------------------

        valid = np.isfinite(
            x
        )


        x = x[valid]
        y = y[valid]


        if (
            len(np.unique(y))
            < 2
        ):

            auc = np.nan

        else:

            try:

                auc = roc_auc_score(
                    y,
                    x,
                )

            except Exception:

                auc = np.nan


        if np.isfinite(auc):

            discriminative_auc = max(
                auc,
                1.0 - auc,
            )

        else:

            discriminative_auc = np.nan


        fault_median = (
            fault_values
            .median()
        )


        median_shift = (
            fault_median
            - normal_median
        )


        result[
            f"{horizon_name}_n"
        ] = len(fault_values)


        result[
            f"{horizon_name}_auc"
        ] = auc


        result[
            f"{horizon_name}_discriminative_auc"
        ] = discriminative_auc


        result[
            f"{horizon_name}_median"
        ] = fault_median


        result[
            f"{horizon_name}_median_shift"
        ] = median_shift


        if np.isfinite(
            discriminative_auc
        ):

            horizon_auc.append(
                discriminative_auc
            )


    # --------------------------------------------------------
    # Aggregate predictive strength.
    #
    # Stronger weight near the fault:
    #
    # T-15 : 20%
    # T-10 : 30%
    # T-5  : 50%
    # --------------------------------------------------------

    weighted_auc_values = []
    weighted_auc_weights = []


    for horizon_name, weight in [
        ("T_minus_15m", 0.20),
        ("T_minus_10m", 0.30),
        ("T_minus_5m", 0.50),
    ]:

        value = result.get(
            f"{horizon_name}_discriminative_auc",
            np.nan,
        )


        if np.isfinite(value):

            weighted_auc_values.append(
                value
            )

            weighted_auc_weights.append(
                weight
            )


    if weighted_auc_values:

        weighted_auc = np.average(
            weighted_auc_values,
            weights=weighted_auc_weights,
        )

    else:

        weighted_auc = np.nan


    # --------------------------------------------------------
    # Persistence.
    #
    # Number of horizons with AUC >= 0.60.
    # --------------------------------------------------------

    persistence = 0


    for horizon_name in HORIZONS:

        value = result.get(
            f"{horizon_name}_discriminative_auc",
            np.nan,
        )


        if (
            np.isfinite(value)
            and value >= 0.60
        ):

            persistence += 1


    # --------------------------------------------------------
    # Strong signal indicator.
    #
    # AUC >= 0.70 is considered meaningful.
    # --------------------------------------------------------

    strong_horizons = 0


    for horizon_name in HORIZONS:

        value = result.get(
            f"{horizon_name}_discriminative_auc",
            np.nan,
        )


        if (
            np.isfinite(value)
            and value >= 0.70
        ):

            strong_horizons += 1


    # --------------------------------------------------------
    # Final ranking score.
    #
    # AUC is primary.
    # Persistence gives a small bonus.
    # --------------------------------------------------------

    if np.isfinite(
        weighted_auc
    ):

        final_score = (
            weighted_auc
            + 0.02 * persistence
            + 0.03 * strong_horizons
        )

    else:

        final_score = np.nan


    result[
        "weighted_discriminative_auc"
    ] = weighted_auc


    result[
        "persistence"
    ] = persistence


    result[
        "strong_horizons"
    ] = strong_horizons


    result[
        "final_signal_score"
    ] = final_score


    results.append(
        result
    )


# ============================================================
# RESULTS DATAFRAME
# ============================================================

results_df = pd.DataFrame(
    results
)


results_df = results_df[
    results_df[
        "final_signal_score"
    ].notna()
].copy()


results_df = results_df.sort_values(
    [
        "final_signal_score",
        "T_minus_5m_discriminative_auc",
        "T_minus_10m_discriminative_auc",
    ],
    ascending=False,
).reset_index(
    drop=True
)


results_df["rank"] = (
    np.arange(
        1,
        len(results_df) + 1
    )
)


# ============================================================
# SAVE RESULTS
# ============================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


results_df.to_csv(
    OUTPUT_FILE,
    index=False,
)


results_df.head(
    TOP_N
).to_csv(
    TOP_FILE,
    index=False,
)


# ============================================================
# FINAL REPORT
# ============================================================

print()
print("=" * 75)
print("CASE-CONTROL + AUC COMPLETE")
print("=" * 75)

print()

print(
    f"Signals analyzed: {len(results_df)}"
)

print(
    f"Normal control rows: {len(normal_df)}"
)

print(
    f"Pre-fault rows: {len(pre_df)}"
)

print()

print(
    "Full ranking:"
)

print(
    OUTPUT_FILE
)

print()

print(
    f"Top {TOP_N}:"
)

print(
    TOP_FILE
)

print()
print("=" * 75)
print("TOP 30 SIGNALS")
print("=" * 75)


display_columns = [
    "rank",
    "feature",
    "T_minus_15m_discriminative_auc",
    "T_minus_10m_discriminative_auc",
    "T_minus_5m_discriminative_auc",
    "weighted_discriminative_auc",
    "persistence",
    "strong_horizons",
    "final_signal_score",
]


print(
    results_df[
        display_columns
    ]
    .head(30)
    .to_string(
        index=False
    )
)


print()
print("=" * 75)
print("AUC INTERPRETATION")
print("=" * 75)

print(
    "0.50  = random"
)

print(
    "0.55  = very weak"
)

print(
    "0.60  = potentially useful"
)

print(
    "0.70  = strong"
)

print(
    "0.80+ = very strong"
)

print()

print(
    "NOTE: discriminative AUC uses max(AUC, 1-AUC),"
)

print(
    "because a useful signal may increase OR decrease"
)

print(
    "before an incident."
)

print()
print("CASE-CONTROL ANALYSIS FINISHED.")

