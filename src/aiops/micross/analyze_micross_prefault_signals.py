from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

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

FEATURE_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "micross"
    / "micross_multisource_features_5m.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "micross"
)

FULL_OUTPUT = (
    OUTPUT_DIR
    / "micross_matched_case_control_auc.csv"
)

TOP_OUTPUT = (
    OUTPUT_DIR
    / "micross_matched_top_auc_signals.csv"
)

TIME_OFFSET_HOURS = 1
GRID_MINUTES = 5

HORIZONS = [15, 10, 5]

CONTROL_PER_CASE = 5
CONTROL_EXCLUSION_MINUTES = 15
CONTROL_TIME_TOLERANCE_MINUTES = 1

MIN_CASES = 10
MIN_VALID_VALUES = 20


# ============================================================
# HELPERS
# ============================================================

def robust_effect(case_values, control_values):
    """
    Robust effect size based on control median and IQR.
    """

    case_values = pd.Series(case_values).dropna()
    control_values = pd.Series(control_values).dropna()

    if len(case_values) == 0 or len(control_values) < MIN_VALID_VALUES:
        return np.nan

    control_median = control_values.median()
    q1 = control_values.quantile(0.25)
    q3 = control_values.quantile(0.75)

    iqr = q3 - q1

    if not np.isfinite(iqr) or iqr == 0:
        return np.nan

    return (
        case_values.median() - control_median
    ) / iqr


def discriminative_auc(case_values, control_values):
    """
    AUC independent of direction.

    max(AUC, 1-AUC)
    means that both positive and negative relationships
    are considered discriminative.
    """

    case_values = pd.Series(case_values).dropna()
    control_values = pd.Series(control_values).dropna()

    if len(case_values) < MIN_CASES:
        return np.nan

    if len(control_values) < MIN_CASES:
        return np.nan

    y = np.concatenate(
        [
            np.ones(len(case_values)),
            np.zeros(len(control_values)),
        ]
    )

    x = np.concatenate(
        [
            case_values.values,
            control_values.values,
        ]
    )

    if len(np.unique(x)) < 2:
        return np.nan

    try:
        auc = roc_auc_score(y, x)
        return max(auc, 1.0 - auc)
    except Exception:
        return np.nan


def closest_row(service_df, target_time):
    """
    Find the metric row closest to target_time,
    with a maximum tolerance.
    """

    if service_df.empty:
        return None

    distances = (
        service_df["timestamp"] - target_time
    ).abs()

    idx = distances.idxmin()

    if distances.loc[idx] <= pd.Timedelta(
        minutes=CONTROL_TIME_TOLERANCE_MINUTES
    ):
        return service_df.loc[idx]

    return None


def build_control_index(service_df, fault_times):
    """
    Mark timestamps that are too close to known faults.

    Controls must be outside +/- CONTROL_EXCLUSION_MINUTES
    from every fault of the same service.
    """

    if service_df.empty:
        return service_df.copy()

    result = service_df.copy()

    excluded = np.zeros(len(result), dtype=bool)

    timestamps = result["timestamp"].values

    for fault_time in fault_times:

        left = (
            fault_time
            - pd.Timedelta(
                minutes=CONTROL_EXCLUSION_MINUTES
            )
        )

        right = (
            fault_time
            + pd.Timedelta(
                minutes=CONTROL_EXCLUSION_MINUTES
            )
        )

        excluded |= (
            (timestamps >= left)
            & (timestamps <= right)
        )

    result["_is_control"] = ~excluded

    return result


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 75)
    print("MICROSS - MATCHED CASE-CONTROL + ROC-AUC")
    print("=" * 75)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ========================================================
    # LOAD FAULTS
    # ========================================================

    print("\nLoading faults...")

    faults = pd.read_csv(
        FAULT_FILE
    )

    print(
        f"Raw faults: {len(faults):,}"
    )

    # Keep only valid-duration faults
    if "duration_suspicious" in faults.columns:

        faults = faults[
            faults["duration_suspicious"] == False
        ].copy()

    print(
        f"Valid faults: {len(faults):,}"
    )

    faults["fault_start"] = (
        pd.to_datetime(
            faults["fault_start"],
            utc=True,
            errors="coerce",
        )
        .dt.tz_localize(None)
    )

    faults = faults.dropna(
        subset=[
            "fault_start",
            "service",
        ]
    ).copy()

    # ========================================================
    # WORKING TIMESTAMP ALIGNMENT
    # ========================================================

    print(
        f"Working offset: "
        f"+{TIME_OFFSET_HOURS}h"
    )

    faults["aligned_fault_start"] = (
        faults["fault_start"]
        + pd.Timedelta(
            hours=TIME_OFFSET_HOURS
        )
    )

    # IMPORTANT:
    # Metrics are sampled on a 5-minute grid.
    # Therefore the exact fault timestamp must be
    # projected onto this grid.

    faults["fault_grid_time"] = (
        faults["aligned_fault_start"]
        .dt.floor(
            f"{GRID_MINUTES}min"
        )
    )

    # One fault window per service/grid timestamp
    faults = faults.drop_duplicates(
        subset=[
            "service",
            "fault_grid_time",
        ]
    ).reset_index(drop=True)

    print(
        "Unique service/fault windows: "
        f"{len(faults):,}"
    )

    # ========================================================
    # LOAD MULTISOURCE FEATURES
    # ========================================================

    print("\nLoading feature dataset...")

    features = pd.read_csv(
        FEATURE_FILE
    )

    print(
        f"Feature dataset shape: "
        f"{features.shape}"
    )

    # --------------------------------------------------------
    # TIMESTAMP NORMALIZATION
    # --------------------------------------------------------

    features["timestamp"] = (
        pd.to_datetime(
            features["timestamp"],
            utc=True,
            errors="coerce",
        )
        .dt.tz_localize(None)
    )

    features = features.dropna(
        subset=["timestamp"]
    ).copy()

    # ========================================================
    # SERVICE DETECTION
    # ========================================================

    if "service" in features.columns:

        service_column = "service"

    else:

        # The multisource file normally does not need this
        # because service is encoded in the feature names.
        #
        # We create a service table from prefixes.

        service_column = None

    # ========================================================
    # IDENTIFY RAW SIGNAL FEATURES
    # ========================================================

    excluded_columns = {
        "timestamp",
        "service",
        "fault_id",
        "block_id",
        "window_start",
        "window_end",
    }

    candidate_features = []

    for column in features.columns:

        if column in excluded_columns:
            continue

        if (
            column.endswith("__last")
            or column.endswith("__delta")
            or column.endswith("__pct_change")
        ):

            candidate_features.append(column)

    print(
        f"Candidate signals: "
        f"{len(candidate_features):,}"
    )

    if not candidate_features:
        raise RuntimeError(
            "No candidate signal features found."
        )

    # ========================================================
    # PREPARE SERVICE TABLES
    # ========================================================

    print(
        "\nPreparing service-level data..."
    )

    service_names = sorted(
        faults["service"]
        .dropna()
        .astype(str)
        .unique()
    )

    service_tables = {}

    for service in service_names:

        service_prefix = (
            str(service).lower()
            + "__"
        )

        service_columns = [
            c
            for c in candidate_features
            if c.lower().startswith(
                service_prefix
            )
        ]

        if not service_columns:
            continue

        cols = [
            "timestamp"
        ] + service_columns

        service_df = features[
            cols
        ].copy()

        service_df = (
            service_df
            .sort_values("timestamp")
            .drop_duplicates(
                subset=["timestamp"]
            )
            .reset_index(drop=True)
        )

        service_tables[service] = service_df

    print(
        f"Service tables prepared: "
        f"{len(service_tables)}"
    )

    # ========================================================
    # CONTROL INDEX
    # ========================================================

    print(
        "\nBuilding service-matched controls..."
    )

    control_tables = {}

    for service, service_df in service_tables.items():

        service_faults = faults[
            faults["service"] == service
        ]

        fault_times = list(
            service_faults[
                "fault_grid_time"
            ]
        )

        control_tables[service] = (
            build_control_index(
                service_df,
                fault_times,
            )
        )

    # ========================================================
    # BUILD CASE-CONTROL SAMPLES
    # ========================================================

    print("\n" + "=" * 75)
    print(
        "BUILDING MATCHED CASE-CONTROL SAMPLES"
    )
    print("=" * 75)

    sample_rows = []

    total_faults = len(faults)

    for fault_idx, fault in faults.iterrows():

        service = str(
            fault["service"]
        )

        if service not in service_tables:
            continue

        service_df = service_tables[
            service
        ]

        control_df = control_tables[
            service
        ]

        fault_time = fault[
            "fault_grid_time"
        ]

        # ----------------------------------------------------
        # One case for each horizon
        # ----------------------------------------------------

        for horizon in HORIZONS:

            target_time = (
                fault_time
                - pd.Timedelta(
                    minutes=horizon
                )
            )

            # ------------------------------------------------
            # CASE
            # ------------------------------------------------

            case_row = closest_row(
                service_df,
                target_time,
            )

            if case_row is None:
                continue

            # ------------------------------------------------
            # CONTROLS
            # ------------------------------------------------

            available_controls = (
                control_df[
                    control_df["_is_control"]
                ].copy()
            )

            if available_controls.empty:
                continue

            # Prefer controls from the same hour
            same_hour = (
                available_controls[
                    available_controls[
                        "timestamp"
                    ].dt.hour
                    == target_time.hour
                ]
                .copy()
            )

            if len(same_hour) >= CONTROL_PER_CASE:

                control_pool = same_hour

            else:

                control_pool = (
                    available_controls
                )

            # ------------------------------------------------
            # Nearest controls in time
            # ------------------------------------------------

            control_pool = (
                control_pool
                .assign(
                    _distance=(
                        control_pool[
                            "timestamp"
                        ]
                        - target_time
                    ).abs()
                )
                .sort_values(
                    "_distance"
                )
                .head(
                    CONTROL_PER_CASE
                )
            )

            # ------------------------------------------------
            # Store samples
            # ------------------------------------------------

            for feature in candidate_features:

                case_value = case_row[
                    feature
                ]

                if pd.isna(case_value):
                    continue

                controls = control_pool[
                    feature
                ].dropna()

                if len(controls) == 0:
                    continue

                row = {
                    "fault_index": fault_idx,
                    "service": service,
                    "horizon_minutes": horizon,
                    "fault_grid_time": fault_time,
                    "target_time": target_time,
                    "feature": feature,
                    "case_value": case_value,
                }

                for control_idx, value in enumerate(
                    controls.values
                ):

                    row[
                        f"control_{control_idx + 1}"
                    ] = value

                sample_rows.append(
                    row
                )

        if (
            (fault_idx + 1) % 100 == 0
        ):

            print(
                f"  processed faults "
                f"{fault_idx + 1}/"
                f"{total_faults}"
            )

    # ========================================================
    # SAMPLE CHECK
    # ========================================================

    samples = pd.DataFrame(
        sample_rows
    )

    print(
        f"\nMatched sample rows: "
        f"{len(samples):,}"
    )

    if samples.empty:

        print(
            "\nERROR: No matched samples."
        )

        print(
            "\nTimestamp diagnostic:"
        )

        print(
            "Feature timestamps:"
        )

        print(
            features["timestamp"]
            .head()
            .tolist()
        )

        print(
            "\nFault grid timestamps:"
        )

        print(
            faults[
                "fault_grid_time"
            ]
            .head()
            .tolist()
        )

        print(
            "\nServices in faults:"
        )

        print(
            sorted(
                faults["service"]
                .astype(str)
                .unique()
            )
        )

        print(
            "\nPrepared service tables:"
        )

        print(
            sorted(
                service_tables.keys()
            )
        )

        raise RuntimeError(
            "No matched samples were created. "
            "Check timestamp/service alignment."
        )

    # ========================================================
    # COMPUTE AUC PER FEATURE / HORIZON
    # ========================================================

    print("\n" + "=" * 75)
    print(
        "RUNNING MATCHED CASE-CONTROL ROC-AUC"
    )
    print("=" * 75)

    results = []

    grouped = samples.groupby(
        [
            "feature",
            "horizon_minutes",
        ]
    )

    for (feature, horizon), group in grouped:

        case_values = (
            group["case_value"]
            .dropna()
        )

        control_values = []

        for i in range(
            1,
            CONTROL_PER_CASE + 1,
        ):

            column = (
                f"control_{i}"
            )

            if column in group.columns:

                control_values.extend(
                    group[column]
                    .dropna()
                    .tolist()
                )

        control_values = pd.Series(
            control_values
        )

        auc = discriminative_auc(
            case_values,
            control_values,
        )

        effect = robust_effect(
            case_values,
            control_values,
        )

        results.append(
            {
                "feature": feature,
                "horizon_minutes": horizon,
                "auc": auc,
                "effect": effect,
                "n_cases": len(case_values),
                "n_controls": len(
                    control_values
                ),
            }
        )

    results_df = pd.DataFrame(
        results
    )

    # ========================================================
    # PIVOT HORIZONS
    # ========================================================

    print(
        "\nAggregating signal performance..."
    )

    final_rows = []

    for feature, group in results_df.groupby(
        "feature"
    ):

        row = {
            "feature": feature
        }

        auc_by_horizon = {}

        effect_by_horizon = {}

        for horizon in HORIZONS:

            sub = group[
                group["horizon_minutes"]
                == horizon
            ]

            if sub.empty:
                auc = np.nan
                effect = np.nan
                n_cases = 0
                n_controls = 0

            else:

                auc = sub.iloc[0]["auc"]
                effect = sub.iloc[0]["effect"]
                n_cases = sub.iloc[0]["n_cases"]
                n_controls = sub.iloc[0]["n_controls"]

            auc_by_horizon[
                horizon
            ] = auc

            effect_by_horizon[
                horizon
            ] = effect

            row[
                f"auc_tminus_{horizon}m"
            ] = auc

            row[
                f"effect_tminus_{horizon}m"
            ] = effect

            row[
                f"cases_tminus_{horizon}m"
            ] = n_cases

            row[
                f"controls_tminus_{horizon}m"
            ] = n_controls

        # ----------------------------------------------------
        # Weighted AUC
        # More importance to T-5
        # ----------------------------------------------------

        values = []

        weights = {
            15: 0.20,
            10: 0.30,
            5: 0.50,
        }

        for horizon, weight in weights.items():

            auc = auc_by_horizon[
                horizon
            ]

            if pd.notna(auc):

                values.append(
                    auc * weight
                )

        if values:

            weighted_auc = sum(
                values
            )

        else:

            weighted_auc = np.nan

        row[
            "weighted_auc"
        ] = weighted_auc

        # ----------------------------------------------------
        # Persistence
        # ----------------------------------------------------

        valid_aucs = [
            x
            for x in auc_by_horizon.values()
            if pd.notna(x)
        ]

        persistence = sum(
            x >= 0.60
            for x in valid_aucs
        )

        strong_horizons = sum(
            x >= 0.70
            for x in valid_aucs
        )

        row[
            "persistence"
        ] = persistence

        row[
            "strong_horizons"
        ] = strong_horizons

        # ----------------------------------------------------
        # Effect consistency
        # ----------------------------------------------------

        valid_effects = [
            x
            for x in effect_by_horizon.values()
            if pd.notna(x)
        ]

        if len(valid_effects) >= 2:

            signs = [
                np.sign(x)
                for x in valid_effects
                if x != 0
            ]

            if signs:

                positive = sum(
                    x > 0
                    for x in signs
                )

                negative = sum(
                    x < 0
                    for x in signs
                )

                consistency = max(
                    positive,
                    negative,
                ) / len(signs)

            else:

                consistency = 0.0

        else:

            consistency = 0.0

        row[
            "effect_consistency"
        ] = consistency

        # ----------------------------------------------------
        # Final ranking score
        # ----------------------------------------------------

        if pd.notna(
            weighted_auc
        ):

            auc_component = (
                weighted_auc - 0.50
            ) * 2.0

            auc_component = max(
                0.0,
                auc_component,
            )

        else:

            auc_component = 0.0

        persistence_component = (
            persistence / 3.0
        )

        consistency_component = (
            consistency
        )

        score = (
            0.60 * auc_component
            + 0.25 * persistence_component
            + 0.15 * consistency_component
        )

        row[
            "signal_score"
        ] = score

        final_rows.append(
            row
        )

    ranking = pd.DataFrame(
        final_rows
    )

    # ========================================================
    # SORT
    # ========================================================

    ranking = ranking.sort_values(
        [
            "signal_score",
            "weighted_auc",
            "persistence",
        ],
        ascending=False,
    ).reset_index(
        drop=True
    )

    ranking[
        "rank"
    ] = np.arange(
        1,
        len(ranking) + 1,
    )

    # ========================================================
    # SAVE
    # ========================================================

    results_df.to_csv(
        FULL_OUTPUT,
        index=False,
    )

    ranking.to_csv(
        TOP_OUTPUT,
        index=False,
    )

    # ========================================================
    # REPORT
    # ========================================================

    print("\n" + "=" * 75)
    print(
        "MATCHED CASE-CONTROL + AUC COMPLETE"
    )
    print("=" * 75)

    print(
        f"Signals analyzed: "
        f"{len(ranking):,}"
    )

    print(
        f"Matched samples: "
        f"{len(samples):,}"
    )

    print(
        "\nTOP 30 MATCHED SIGNALS"
    )

    display_columns = [
        "rank",
        "feature",
        "auc_tminus_15m",
        "auc_tminus_10m",
        "auc_tminus_5m",
        "weighted_auc",
        "persistence",
        "strong_horizons",
        "effect_consistency",
        "signal_score",
    ]

    print(
        ranking[
            display_columns
        ]
        .head(30)
        .to_string(
            index=False
        )
    )

    # ========================================================
    # INTERPRETATION
    # ========================================================

    print("\n" + "=" * 75)
    print("INTERPRETATION")
    print("=" * 75)

    if ranking.empty:

        print(
            "No signals could be ranked."
        )

    else:

        best_auc = ranking[
            "weighted_auc"
        ].max()

        signals_60 = (
            ranking[
                "weighted_auc"
            ] >= 0.60
        ).sum()

        signals_70 = (
            ranking[
                "weighted_auc"
            ] >= 0.70
        ).sum()

        print(
            f"Best weighted AUC: "
            f"{best_auc:.4f}"
        )

        print(
            f"Signals with AUC >= 0.60: "
            f"{signals_60}"
        )

        print(
            f"Signals with AUC >= 0.70: "
            f"{signals_70}"
        )

        if best_auc >= 0.70:

            print(
                "\nStrong discriminative signal detected."
            )

        elif best_auc >= 0.60:

            print(
                "\nModerate discriminative signal detected."
            )

        else:

            print(
                "\nNo strong pre-fault discriminative "
                "signal detected."
            )

    print(
        "\nFull results:"
    )

    print(
        FULL_OUTPUT
    )

    print(
        "\nTop signals:"
    )

    print(
        TOP_OUTPUT
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()