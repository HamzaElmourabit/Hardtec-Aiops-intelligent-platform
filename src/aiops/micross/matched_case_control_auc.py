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


# IMPORTANT
# The multisource feature dataset was ALREADY built with +1h.
# Therefore we do NOT shift the feature timestamps again.
#
# We only apply +1h to the original RUN fault timestamps.
FAULT_OFFSET_HOURS = 1

GRID_MINUTES = 5

HORIZONS = [15, 10, 5]

CONTROL_PER_CASE = 5

CONTROL_EXCLUSION_MINUTES = 15

TIME_TOLERANCE_MINUTES = 2

MIN_CASES = 10


# ============================================================
# HELPERS
# ============================================================

def normalize_timestamp(series):
    """
    Convert timestamps to UTC and then remove timezone information.

    This guarantees that both RUN and metric timestamps use
    the same tz-naive representation.
    """

    return (
        pd.to_datetime(
            series,
            utc=True,
            errors="coerce",
        )
        .dt.tz_localize(None)
    )


def discriminative_auc(case_values, control_values):
    """
    Direction-independent ROC-AUC.

    max(AUC, 1-AUC)
    """

    case_values = (
        pd.Series(case_values)
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )

    control_values = (
        pd.Series(control_values)
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )

    if len(case_values) < MIN_CASES:
        return np.nan

    if len(control_values) < MIN_CASES:
        return np.nan

    x = np.concatenate(
        [
            case_values.values,
            control_values.values,
        ]
    )

    y = np.concatenate(
        [
            np.ones(len(case_values)),
            np.zeros(len(control_values)),
        ]
    )

    if len(np.unique(x)) < 2:
        return np.nan

    try:

        auc = roc_auc_score(
            y,
            x,
        )

        return max(
            auc,
            1.0 - auc,
        )

    except Exception:

        return np.nan


def robust_effect(case_values, control_values):

    case_values = (
        pd.Series(case_values)
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )

    control_values = (
        pd.Series(control_values)
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )

    if len(case_values) < MIN_CASES:
        return np.nan

    if len(control_values) < MIN_CASES:
        return np.nan

    q1 = control_values.quantile(0.25)
    q3 = control_values.quantile(0.75)

    iqr = q3 - q1

    if not np.isfinite(iqr) or iqr == 0:
        return np.nan

    return (
        case_values.median()
        - control_values.median()
    ) / iqr


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
    # 1. LOAD FAULTS
    # ========================================================

    print("\nLoading faults...")

    faults = pd.read_csv(
        FAULT_FILE
    )

    print(
        f"Raw faults: {len(faults):,}"
    )

    # Keep valid durations only
    if "duration_suspicious" in faults.columns:

        faults = faults[
            faults["duration_suspicious"]
            == False
        ].copy()

    print(
        f"Valid faults: {len(faults):,}"
    )

    faults["fault_start"] = normalize_timestamp(
        faults["fault_start"]
    )

    faults = faults.dropna(
        subset=[
            "fault_start",
            "service",
        ]
    ).copy()

    faults["service"] = (
        faults["service"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    # ========================================================
    # 2. ALIGN ORIGINAL RUN TIMESTAMPS
    # ========================================================

    print(
        f"Working fault offset: "
        f"+{FAULT_OFFSET_HOURS}h"
    )

    faults["aligned_fault_start"] = (
        faults["fault_start"]
        + pd.Timedelta(
            hours=FAULT_OFFSET_HOURS
        )
    )

    # Project fault to 5-minute grid
    faults["fault_grid_time"] = (
        faults["aligned_fault_start"]
        .dt.floor(
            f"{GRID_MINUTES}min"
        )
    )

    # One fault window per service/grid timestamp
    faults = (
        faults
        .drop_duplicates(
            subset=[
                "service",
                "fault_grid_time",
            ]
        )
        .reset_index(drop=True)
    )

    print(
        "Unique service/fault windows: "
        f"{len(faults):,}"
    )

    # ========================================================
    # 3. LOAD FEATURES
    # ========================================================

    print(
        "\nLoading feature dataset..."
    )

    features = pd.read_csv(
        FEATURE_FILE
    )

    print(
        f"Feature dataset shape: "
        f"{features.shape}"
    )

    features["timestamp"] = normalize_timestamp(
        features["timestamp"]
    )

    features = features.dropna(
        subset=["timestamp"]
    ).copy()

    # ========================================================
    # 4. IDENTIFY SERVICE FEATURES
    # ========================================================

    candidate_features = []

    for column in features.columns:

        if (
            column.endswith("__last")
            or column.endswith("__delta")
            or column.endswith("__pct_change")
        ):

            candidate_features.append(
                column
            )

    print(
        f"Candidate signals: "
        f"{len(candidate_features):,}"
    )

    if not candidate_features:

        raise RuntimeError(
            "No candidate signal features found."
        )

    # ========================================================
    # 5. SERVICE TABLES
    # ========================================================

    print(
        "\nPreparing service-level data..."
    )

    service_names = sorted(
        faults["service"]
        .unique()
    )

    service_tables = {}

    for service in service_names:

        prefix = (
            service.lower()
            + "__"
        )

        service_columns = [
            column
            for column in candidate_features
            if column.lower().startswith(
                prefix
            )
        ]

        if not service_columns:
            continue

        table = features[
            ["timestamp"]
            + service_columns
        ].copy()

        table = (
            table
            .sort_values("timestamp")
            .drop_duplicates(
                subset=["timestamp"]
            )
            .reset_index(drop=True)
        )

        service_tables[
            service
        ] = table

    print(
        "Prepared service tables: "
        f"{len(service_tables)}"
    )

    print(
        "Services:"
    )

    for service in sorted(
        service_tables
    ):

        print(
            f"  {service}: "
            f"{len(service_tables[service]):,} rows"
        )

    # ========================================================
    # 6. CRITICAL TIMESTAMP DIAGNOSTIC
    # ========================================================

    print("\n" + "=" * 75)
    print(
        "TIMESTAMP ALIGNMENT CHECK"
    )
    print("=" * 75)

    print(
        "\nFeature time range:"
    )

    print(
        features["timestamp"].min(),
        "->",
        features["timestamp"].max(),
    )

    print(
        "\nAligned fault time range:"
    )

    print(
        faults["aligned_fault_start"].min(),
        "->",
        faults["aligned_fault_start"].max(),
    )

    print(
        "\nFault grid time range:"
    )

    print(
        faults["fault_grid_time"].min(),
        "->",
        faults["fault_grid_time"].max(),
    )

    # ========================================================
    # 7. QUICK MATCH TEST
    # ========================================================

    print("\n" + "=" * 75)
    print(
        "QUICK MATCH TEST"
    )
    print("=" * 75)

    quick_matches = 0

    for _, fault in faults.head(30).iterrows():

        service = fault["service"]

        if service not in service_tables:
            continue

        table = service_tables[
            service
        ]

        fault_time = fault[
            "fault_grid_time"
        ]

        target = (
            fault_time
            - pd.Timedelta(
                minutes=15
            )
        )

        distances = (
            table["timestamp"]
            - target
        ).abs()

        if (
            distances
            <= pd.Timedelta(
                minutes=TIME_TOLERANCE_MINUTES
            )
        ).any():

            quick_matches += 1

    print(
        f"Quick matches among first "
        f"30 faults: "
        f"{quick_matches}/30"
    )

    if quick_matches == 0:

        print(
            "\nNo quick matches detected."
        )

        print(
            "\nFirst feature timestamps:"
        )

        print(
            features[
                "timestamp"
            ]
            .head(10)
            .tolist()
        )

        print(
            "\nFirst fault grid timestamps:"
        )

        print(
            faults[
                "fault_grid_time"
            ]
            .head(10)
            .tolist()
        )

        print(
            "\nFirst fault services:"
        )

        print(
            faults[
                "service"
            ]
            .head(10)
            .tolist()
        )

        raise RuntimeError(
            "Timestamp/service alignment "
            "still produces zero matches."
        )

    # ========================================================
    # 8. BUILD CONTROL POOLS
    # ========================================================

    print("\n" + "=" * 75)
    print(
        "BUILDING SERVICE-MATCHED CONTROLS"
    )
    print("=" * 75)

    control_tables = {}

    for service, table in service_tables.items():

        service_faults = faults[
            faults["service"]
            == service
        ]

        fault_times = (
            service_faults[
                "fault_grid_time"
            ]
            .tolist()
        )

        control = table.copy()

        control["_is_control"] = True

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

            mask = (
                (control["timestamp"] >= left)
                & (
                    control["timestamp"]
                    <= right
                )
            )

            control.loc[
                mask,
                "_is_control",
            ] = False

        control_tables[
            service
        ] = control

    # ========================================================
    # 9. BUILD MATCHED SAMPLES
    # ========================================================

    print("\n" + "=" * 75)
    print(
        "BUILDING MATCHED CASE-CONTROL SAMPLES"
    )
    print("=" * 75)

    sample_rows = []

    total_faults = len(faults)

    for fault_index, fault in faults.iterrows():

        service = fault[
            "service"
        ]

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
        # Each horizon
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

            distances = (
                service_df["timestamp"]
                - target_time
            ).abs()

            case_candidates = service_df[
                distances
                <= pd.Timedelta(
                    minutes=TIME_TOLERANCE_MINUTES
                )
            ].copy()

            if case_candidates.empty:
                continue

            case_candidates[
                "_distance"
            ] = (
                case_candidates[
                    "timestamp"
                ]
                - target_time
            ).abs()

            case_row = (
                case_candidates
                .sort_values(
                    "_distance"
                )
                .iloc[0]
            )

            # ------------------------------------------------
            # CONTROLS
            # ------------------------------------------------

            controls = control_df[
                control_df["_is_control"]
            ].copy()

            if controls.empty:
                continue

            controls[
                "_distance"
            ] = (
                controls[
                    "timestamp"
                ]
                - target_time
            ).abs()

            # Prefer same hour
            same_hour = controls[
                controls[
                    "timestamp"
                ].dt.hour
                == target_time.hour
            ].copy()

            if len(same_hour) >= CONTROL_PER_CASE:

                selected_controls = (
                    same_hour
                    .sort_values(
                        "_distance"
                    )
                    .head(
                        CONTROL_PER_CASE
                    )
                )

            else:

                selected_controls = (
                    controls
                    .sort_values(
                        "_distance"
                    )
                    .head(
                        CONTROL_PER_CASE
                    )
                )

            if selected_controls.empty:
                continue

            # ------------------------------------------------
            # Store each signal
            # ------------------------------------------------

            for feature in candidate_features:

                case_value = case_row[
                    feature
                ]

                if pd.isna(case_value):
                    continue

                values = (
                    selected_controls[
                        feature
                    ]
                    .replace(
                        [np.inf, -np.inf],
                        np.nan,
                    )
                    .dropna()
                    .tolist()
                )

                if not values:
                    continue

                row = {
                    "fault_index": fault_index,
                    "service": service,
                    "fault_grid_time": fault_time,
                    "target_time": target_time,
                    "horizon_minutes": horizon,
                    "feature": feature,
                    "case_value": case_value,
                }

                for index, value in enumerate(
                    values[:CONTROL_PER_CASE],
                    start=1,
                ):

                    row[
                        f"control_{index}"
                    ] = value

                sample_rows.append(
                    row
                )

        if (
            (fault_index + 1) % 100
            == 0
        ):

            print(
                f"  processed faults "
                f"{fault_index + 1}/"
                f"{total_faults}"
            )

    samples = pd.DataFrame(
        sample_rows
    )

    print(
        f"\nMatched sample rows: "
        f"{len(samples):,}"
    )

    if samples.empty:

        raise RuntimeError(
            "No matched samples were created."
        )

    # ========================================================
    # 10. BASIC SAMPLE STATISTICS
    # ========================================================

    print(
        "\nCases by horizon:"
    )

    print(
        samples.groupby(
            "horizon_minutes"
        )[
            "fault_index"
        ]
        .nunique()
        .sort_index(
            ascending=False
        )
    )

    print(
        "\nServices represented:"
    )

    print(
        samples[
            "service"
        ]
        .nunique()
    )

    # ========================================================
    # 11. ROC-AUC
    # ========================================================

    print("\n" + "=" * 75)
    print(
        "RUNNING MATCHED CASE-CONTROL ROC-AUC"
    )
    print("=" * 75)

    results = []

    for (
        feature,
        horizon,
    ), group in samples.groupby(
        [
            "feature",
            "horizon_minutes",
        ]
    ):

        cases = (
            group[
                "case_value"
            ]
            .replace(
                [np.inf, -np.inf],
                np.nan,
            )
            .dropna()
        )

        controls = []

        for index in range(
            1,
            CONTROL_PER_CASE + 1,
        ):

            column = (
                f"control_{index}"
            )

            if column not in group.columns:
                continue

            controls.extend(
                group[
                    column
                ]
                .replace(
                    [np.inf, -np.inf],
                    np.nan,
                )
                .dropna()
                .tolist()
            )

        controls = pd.Series(
            controls
        )

        auc = discriminative_auc(
            cases,
            controls,
        )

        effect = robust_effect(
            cases,
            controls,
        )

        results.append(
            {
                "feature": feature,
                "horizon_minutes": horizon,
                "auc": auc,
                "effect": effect,
                "n_cases": len(cases),
                "n_controls": len(controls),
            }
        )

    results_df = pd.DataFrame(
        results
    )

    # ========================================================
    # 12. RANK FEATURES
    # ========================================================

    ranking_rows = []

    for feature, group in results_df.groupby(
        "feature"
    ):

        row = {
            "feature": feature
        }

        auc_values = {}

        effect_values = {}

        for horizon in HORIZONS:

            subset = group[
                group[
                    "horizon_minutes"
                ]
                == horizon
            ]

            if subset.empty:

                auc = np.nan
                effect = np.nan

            else:

                auc = subset.iloc[0][
                    "auc"
                ]

                effect = subset.iloc[0][
                    "effect"
                ]

            auc_values[
                horizon
            ] = auc

            effect_values[
                horizon
            ] = effect

            row[
                f"auc_tminus_{horizon}m"
            ] = auc

            row[
                f"effect_tminus_{horizon}m"
            ] = effect

        # ----------------------------------------------------
        # Weighted AUC
        # ----------------------------------------------------

        weighted_parts = []

        weights = {
            15: 0.20,
            10: 0.30,
            5: 0.50,
        }

        for horizon, weight in weights.items():

            auc = auc_values[
                horizon
            ]

            if pd.notna(auc):

                weighted_parts.append(
                    auc * weight
                )

        weighted_auc = (
            sum(weighted_parts)
            if weighted_parts
            else np.nan
        )

        row[
            "weighted_auc"
        ] = weighted_auc

        # ----------------------------------------------------
        # Persistence
        # ----------------------------------------------------

        persistence = sum(
            pd.notna(auc)
            and auc >= 0.60
            for auc in auc_values.values()
        )

        strong_horizons = sum(
            pd.notna(auc)
            and auc >= 0.70
            for auc in auc_values.values()
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

        effects = [
            value
            for value in effect_values.values()
            if pd.notna(value)
            and value != 0
        ]

        if len(effects) >= 2:

            positive = sum(
                value > 0
                for value in effects
            )

            negative = sum(
                value < 0
                for value in effects
            )

            consistency = (
                max(
                    positive,
                    negative,
                )
                / len(effects)
            )

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

            auc_component = max(
                0.0,
                (
                    weighted_auc
                    - 0.50
                ) * 2.0,
            )

        else:

            auc_component = 0.0

        persistence_component = (
            persistence / 3.0
        )

        score = (
            0.60 * auc_component
            + 0.25 * persistence_component
            + 0.15 * consistency
        )

        row[
            "signal_score"
        ] = score

        ranking_rows.append(
            row
        )

    ranking = pd.DataFrame(
        ranking_rows
    )

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
    # 13. SAVE
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
    # 14. FINAL REPORT
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
        f"Matched sample rows: "
        f"{len(samples):,}"
    )

    print(
        "\nTOP 30 MATCHED SIGNALS"
    )

    columns = [
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
            columns
        ]
        .head(30)
        .to_string(
            index=False
        )
    )

    # ========================================================
    # 15. INTERPRETATION
    # ========================================================

    print("\n" + "=" * 75)
    print(
        "INTERPRETATION"
    )
    print("=" * 75)

    if not ranking.empty:

        best_auc = ranking[
            "weighted_auc"
        ].max()

        count_60 = (
            ranking[
                "weighted_auc"
            ] >= 0.60
        ).sum()

        count_70 = (
            ranking[
                "weighted_auc"
            ] >= 0.70
        ).sum()

        print(
            f"Best weighted AUC: "
            f"{best_auc:.4f}"
        )

        print(
            f"Signals with weighted AUC >= 0.60: "
            f"{count_60}"
        )

        print(
            f"Signals with weighted AUC >= 0.70: "
            f"{count_70}"
        )

        if best_auc >= 0.70:

            print(
                "\nStrong pre-fault discriminative "
                "signals detected."
            )

        elif best_auc >= 0.60:

            print(
                "\nModerate pre-fault discriminative "
                "signals detected."
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


if __name__ == "__main__":
    main()