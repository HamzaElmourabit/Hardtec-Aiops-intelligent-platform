from pathlib import Path
import pandas as pd
import numpy as np


# ============================================================
# CONFIGURATION
# ============================================================

# build_micross_fault_windows.py
#      0 -> micross
#      1 -> aiops
#      2 -> src
#      3 -> hardtec-intelligent-ticketing
PROJECT_ROOT = Path(__file__).resolve().parents[3]

FAULT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "micross"
    / "micross_fault_events.csv"
)

METRIC_DIR = (
    PROJECT_ROOT
    / "MicroSS"
    / "metric_selected"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "micross"
)

OUTPUT_FILE = (
    OUTPUT_DIR
    / "micross_fault_windows.csv"
)

# Hypothèse actuelle issue de l'analyse d'alignement
CANDIDATE_OFFSET_HOURS = 1

# Fenêtres pré-faute
PRE_WINDOWS = {
    "pre_10m": 10,
    "pre_5m": 5,
    "pre_2m": 2,
}

# Fenêtre post-faute
POST_MINUTES = 5


# ============================================================
# UTILITAIRES
# ============================================================

def calculate_stats(series):
    """
    Calcule les statistiques principales d'une série.
    """

    if series is None or len(series) == 0:
        return {
            "count": 0,
            "mean": np.nan,
            "median": np.nan,
            "std": np.nan,
            "min": np.nan,
            "max": np.nan,
            "last": np.nan,
        }

    values = pd.to_numeric(
        series,
        errors="coerce"
    ).dropna()

    if len(values) == 0:
        return {
            "count": 0,
            "mean": np.nan,
            "median": np.nan,
            "std": np.nan,
            "min": np.nan,
            "max": np.nan,
            "last": np.nan,
        }

    return {
        "count": int(len(values)),
        "mean": float(values.mean()),
        "median": float(values.median()),
        "std": float(values.std()) if len(values) > 1 else 0.0,
        "min": float(values.min()),
        "max": float(values.max()),
        "last": float(values.iloc[-1]),
    }


def calculate_change(pre_value, during_value):
    """
    Variation relative entre PRE et DURING.
    """

    if pd.isna(pre_value) or pd.isna(during_value):
        return np.nan

    if abs(pre_value) < 1e-12:
        return np.nan

    return (
        (during_value - pre_value)
        / abs(pre_value)
    ) * 100.0


def robust_zscore(values):
    """
    Robust Z-score basé sur médiane + MAD.
    """

    values = pd.to_numeric(
        values,
        errors="coerce"
    )

    valid = values.dropna()

    if valid.empty:
        return pd.Series(
            index=values.index,
            dtype=float
        )

    median = valid.median()

    mad = np.median(
        np.abs(valid - median)
    )

    if mad == 0 or pd.isna(mad):
        return pd.Series(
            0.0,
            index=values.index
        )

    return (
        0.6745
        * (values - median)
        / mad
    )


# ============================================================
# 1. CHARGEMENT DES FAUTES
# ============================================================

print("=" * 70)
print("MICROSS - FAULT / METRIC TEMPORAL ANALYSIS")
print("=" * 70)

print("\nProject root:")
print(PROJECT_ROOT)

print("\nFault file:")
print(FAULT_FILE)

print("\nMetric directory:")
print(METRIC_DIR)


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


# ============================================================
# CONVERSION DES TIMESTAMPS
# ============================================================

faults["fault_start"] = pd.to_datetime(
    faults["fault_start"],
    errors="coerce"
)

faults["fault_end"] = pd.to_datetime(
    faults["fault_end"],
    errors="coerce"
)

faults = faults.dropna(
    subset=[
        "fault_start",
        "fault_end"
    ]
).copy()


# ============================================================
# SUPPRESSION DES DUREES SUSPECTES
# ============================================================

if "duration_suspicious" in faults.columns:

    suspicious_count = (
        faults["duration_suspicious"] == True
    ).sum()

    print(
        f"Suspicious duration rows excluded: "
        f"{suspicious_count}"
    )

    faults = faults[
        faults["duration_suspicious"] != True
    ].copy()


print(
    f"Valid faults used: {len(faults):,}"
)


# ============================================================
# ALIGNEMENT TEMPOREL HYPOTHETIQUE
# ============================================================

faults["aligned_fault_start"] = (
    faults["fault_start"]
    + pd.Timedelta(
        hours=CANDIDATE_OFFSET_HOURS
    )
)

faults["aligned_fault_end"] = (
    faults["fault_end"]
    + pd.Timedelta(
        hours=CANDIDATE_OFFSET_HOURS
    )
)

print(
    f"Candidate timestamp offset: "
    f"+{CANDIDATE_OFFSET_HOURS} hour"
)


# ============================================================
# 2. CHARGEMENT DES METRIQUES
# ============================================================

print("\n[2] Loading metric files...")

if not METRIC_DIR.exists():
    raise FileNotFoundError(
        f"Metric directory not found:\n{METRIC_DIR}"
    )

metric_files = sorted(
    METRIC_DIR.glob("*.csv")
)

print(
    f"Metric files found: {len(metric_files)}"
)

if len(metric_files) == 0:
    raise FileNotFoundError(
        f"No CSV files found in:\n{METRIC_DIR}"
    )


metrics = {}


for metric_file in metric_files:

    filename = metric_file.name.lower()

    # --------------------------------------------------------
    # Identification du type de métrique
    # --------------------------------------------------------

    if "docker_cpu_total_norm_pct" in filename:

        metric_type = "cpu"

    elif "docker_memory_usage_pct" in filename:

        metric_type = "memory"

    else:

        print(
            f"Skipping unrelated metric: "
            f"{metric_file.name}"
        )

        continue


    # --------------------------------------------------------
    # Identification du service
    # --------------------------------------------------------

    service = None

    for service_name in (
        faults["service"]
        .dropna()
        .astype(str)
        .unique()
    ):

        if service_name.lower() in filename:

            service = service_name
            break


    if service is None:

        print(
            f"WARNING: service not detected: "
            f"{metric_file.name}"
        )

        continue


    # --------------------------------------------------------
    # Lecture
    # --------------------------------------------------------

    df = pd.read_csv(
        metric_file
    )


    if (
        "timestamp" not in df.columns
        or
        "value" not in df.columns
    ):

        print(
            f"WARNING: invalid schema: "
            f"{metric_file.name}"
        )

        continue


    # --------------------------------------------------------
    # Timestamp Unix milliseconds -> UTC
    # --------------------------------------------------------

    df["timestamp"] = pd.to_datetime(
        pd.to_numeric(
            df["timestamp"],
            errors="coerce"
        ),
        unit="ms",
        utc=True,
        errors="coerce"
    )


    # --------------------------------------------------------
    # Valeurs numériques
    # --------------------------------------------------------

    df["value"] = pd.to_numeric(
        df["value"],
        errors="coerce"
    )


    df = df.dropna(
        subset=[
            "timestamp",
            "value"
        ]
    ).copy()


    df = df.sort_values(
        "timestamp"
    )


    # --------------------------------------------------------
    # Retrait timezone
    # --------------------------------------------------------

    df["timestamp"] = (
        df["timestamp"]
        .dt.tz_localize(None)
    )


    # --------------------------------------------------------
    # Robust Z-score
    # --------------------------------------------------------

    df["robust_z"] = robust_zscore(
        df["value"]
    )


    # --------------------------------------------------------
    # Stockage
    # --------------------------------------------------------

    key = (
        service,
        metric_type
    )

    metrics[key] = df


    print(
        f"  {service:15s} | "
        f"{metric_type:7s} | "
        f"{len(df):,} points | "
        f"{df['timestamp'].min()} -> "
        f"{df['timestamp'].max()}"
    )


print(
    f"\nLoaded metric series: "
    f"{len(metrics)}"
)


# ============================================================
# 3. CONSTRUCTION DES FENETRES
# ============================================================

print(
    "\n[3] Building temporal fault windows..."
)

results = []


for counter, (_, fault) in enumerate(
    faults.iterrows(),
    start=1
):

    service = str(
        fault["service"]
    )

    fault_type = str(
        fault.get(
            "fault_type",
            "unknown"
        )
    )


    fault_start = (
        fault["aligned_fault_start"]
    )

    fault_end = (
        fault["aligned_fault_end"]
    )


    # --------------------------------------------------------
    # Informations générales
    # --------------------------------------------------------

    row = {

        "fault_id": counter,

        "service": service,

        "fault_type": fault_type,

        # Timestamp RUN original
        "original_fault_start":
            fault["fault_start"],

        "original_fault_end":
            fault["fault_end"],

        # Timestamp aligné
        "aligned_fault_start":
            fault_start,

        "aligned_fault_end":
            fault_end,

        "candidate_offset_hours":
            CANDIDATE_OFFSET_HOURS,

        "fault_duration_seconds":
            (
                fault_end
                - fault_start
            ).total_seconds(),
    }


    # ========================================================
    # CPU ET MEMORY
    # ========================================================

    for metric_type in [
        "cpu",
        "memory"
    ]:

        key = (
            service,
            metric_type
        )


        # ----------------------------------------------------
        # Si métrique absente
        # ----------------------------------------------------

        if key not in metrics:

            print(
                f"WARNING: missing metric "
                f"{metric_type} for {service}"
            )

            for window_name in [
                "pre_10m",
                "pre_5m",
                "pre_2m",
                "during",
                "post_5m",
            ]:

                for stat in [
                    "count",
                    "mean",
                    "median",
                    "std",
                    "min",
                    "max",
                    "last",
                ]:

                    row[
                        f"{metric_type}_"
                        f"{window_name}_"
                        f"{stat}"
                    ] = np.nan


            row[
                f"{metric_type}_"
                "change_pre10m_to_during_pct"
            ] = np.nan

            row[
                f"{metric_type}_"
                "change_pre5m_to_during_pct"
            ] = np.nan

            row[
                f"{metric_type}_"
                "change_pre2m_to_during_pct"
            ] = np.nan

            continue


        df = metrics[key]


        # ====================================================
        # PRE-FAULT
        # ====================================================

        for window_name, minutes in (
            PRE_WINDOWS.items()
        ):

            start = (
                fault_start
                - pd.Timedelta(
                    minutes=minutes
                )
            )

            end = fault_start


            subset = df[
                (df["timestamp"] >= start)
                &
                (df["timestamp"] < end)
            ]


            stats = calculate_stats(
                subset["value"]
            )


            for stat_name, stat_value in (
                stats.items()
            ):

                row[
                    f"{metric_type}_"
                    f"{window_name}_"
                    f"{stat_name}"
                ] = stat_value


            # ------------------------------------------------
            # Anomalies pré-faute
            # ------------------------------------------------

            anomaly_count = int(
                (
                    subset["robust_z"].abs()
                    >= 3
                ).sum()
            )


            row[
                f"{metric_type}_"
                f"{window_name}_"
                "anomaly_count"
            ] = anomaly_count


            anomaly_rate = (
                anomaly_count / len(subset)
                if len(subset) > 0
                else 0
            )


            row[
                f"{metric_type}_"
                f"{window_name}_"
                "anomaly_rate"
            ] = anomaly_rate


            max_abs_z = (
                subset["robust_z"]
                .abs()
                .max()
                if len(subset) > 0
                else np.nan
            )


            row[
                f"{metric_type}_"
                f"{window_name}_"
                "max_abs_z"
            ] = max_abs_z


        # ====================================================
        # DURING FAULT
        # ====================================================

        during_subset = df[
            (df["timestamp"] >= fault_start)
            &
            (df["timestamp"] <= fault_end)
        ]


        during_stats = calculate_stats(
            during_subset["value"]
        )


        for stat_name, stat_value in (
            during_stats.items()
        ):

            row[
                f"{metric_type}_"
                f"during_"
                f"{stat_name}"
            ] = stat_value


        row[
            f"{metric_type}_"
            "during_anomaly_count"
        ] = int(
            (
                during_subset["robust_z"].abs()
                >= 3
            ).sum()
        )


        # ====================================================
        # POST-FAULT
        # ====================================================

        post_start = fault_end

        post_end = (
            fault_end
            + pd.Timedelta(
                minutes=POST_MINUTES
            )
        )


        post_subset = df[
            (df["timestamp"] > post_start)
            &
            (df["timestamp"] <= post_end)
        ]


        post_stats = calculate_stats(
            post_subset["value"]
        )


        for stat_name, stat_value in (
            post_stats.items()
        ):

            row[
                f"{metric_type}_"
                f"post_5m_"
                f"{stat_name}"
            ] = stat_value


        # ====================================================
        # VARIATIONS
        # ====================================================

        during_mean = row.get(
            f"{metric_type}_"
            "during_mean",
            np.nan
        )


        row[
            f"{metric_type}_"
            "change_pre10m_to_during_pct"
        ] = calculate_change(

            row.get(
                f"{metric_type}_"
                "pre_10m_mean",
                np.nan
            ),

            during_mean
        )


        row[
            f"{metric_type}_"
            "change_pre5m_to_during_pct"
        ] = calculate_change(

            row.get(
                f"{metric_type}_"
                "pre_5m_mean",
                np.nan
            ),

            during_mean
        )


        row[
            f"{metric_type}_"
            "change_pre2m_to_during_pct"
        ] = calculate_change(

            row.get(
                f"{metric_type}_"
                "pre_2m_mean",
                np.nan
            ),

            during_mean
        )


    results.append(row)


    # Progression
    if counter % 100 == 0:

        print(
            f"  Processed "
            f"{counter:,}/"
            f"{len(faults):,} faults"
        )


# ============================================================
# 4. DATAFRAME FINAL
# ============================================================

print(
    "\n[4] Creating final dataframe..."
)

result_df = pd.DataFrame(
    results
)


OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


result_df.to_csv(
    OUTPUT_FILE,
    index=False
)


print(
    f"\nOutput saved to:\n"
    f"{OUTPUT_FILE}"
)


print(
    f"\nShape: "
    f"{result_df.shape}"
)


# ============================================================
# 5. COUVERTURE DES DONNEES
# ============================================================

print(
    "\n" + "=" * 70
)

print(
    "DATA COVERAGE"
)

print(
    "=" * 70
)


cpu_during = (
    result_df["cpu_during_count"] > 0
)

memory_during = (
    result_df["memory_during_count"] > 0
)

both_during = (
    cpu_during
    &
    memory_during
)


print(
    f"Faults with CPU data during fault:"
    f" {cpu_during.sum():,}/"
    f"{len(result_df):,}"
)

print(
    f"Faults with Memory data during fault:"
    f" {memory_during.sum():,}/"
    f"{len(result_df):,}"
)

print(
    f"Faults with BOTH CPU + Memory:"
    f" {both_during.sum():,}/"
    f"{len(result_df):,}"
)


# ============================================================
# 6. SIGNAL QUALITY
# ============================================================

print(
    "\n" + "=" * 70
)

print(
    "SIGNAL QUALITY SUMMARY"
)

print(
    "=" * 70
)


for metric_type in [
    "cpu",
    "memory"
]:

    print(
        f"\n{metric_type.upper()}"
    )


    for window_name in [
        "pre_10m",
        "pre_5m",
        "pre_2m"
    ]:

        count_col = (
            f"{metric_type}_"
            f"{window_name}_count"
        )

        anomaly_col = (
            f"{metric_type}_"
            f"{window_name}_"
            "anomaly_count"
        )


        valid_count = (
            result_df[count_col] > 0
        ).sum()


        anomaly_faults = (
            result_df[anomaly_col] > 0
        ).sum()


        print(
            f"{window_name:10s} | "
            f"data: "
            f"{valid_count:4d}/"
            f"{len(result_df):4d} | "
            f"anomaly: "
            f"{anomaly_faults:4d}/"
            f"{len(result_df):4d}"
        )


# ============================================================
# 7. VARIATION MOYENNE
# ============================================================

print(
    "\n" + "=" * 70
)

print(
    "AVERAGE PRE-FAULT -> DURING-FAULT CHANGE"
)

print(
    "=" * 70
)


for metric_type in [
    "cpu",
    "memory"
]:

    print(
        f"\n{metric_type.upper()}"
    )


    for window_name in [
        "pre10m",
        "pre5m",
        "pre2m"
    ]:

        column = (
            f"{metric_type}_"
            f"change_{window_name}"
            "_to_during_pct"
        )


        if column not in result_df.columns:
            continue


        values = (
            result_df[column]
            .replace(
                [np.inf, -np.inf],
                np.nan
            )
            .dropna()
        )


        if values.empty:

            print(
                f"{window_name:10s} | "
                f"No valid data"
            )

        else:

            print(
                f"{window_name:10s} | "
                f"mean: "
                f"{values.mean():.2f}% | "
                f"median: "
                f"{values.median():.2f}%"
            )


# ============================================================
# 8. EXEMPLES EXTREMES CPU
# ============================================================

cpu_change_column = (
    "cpu_change_pre5m_to_during_pct"
)


if cpu_change_column in result_df.columns:

    print(
        "\n" + "=" * 70
    )

    print(
        "TOP CPU CHANGES"
    )

    print(
        "=" * 70
    )


    print(
        result_df[
            [
                "fault_id",
                "service",
                "fault_type",
                cpu_change_column,
            ]
        ]
        .replace(
            [np.inf, -np.inf],
            np.nan
        )
        .dropna(
            subset=[
                cpu_change_column
            ]
        )
        .sort_values(
            cpu_change_column,
            ascending=False
        )
        .head(10)
        .to_string(
            index=False
        )
    )


# ============================================================
# 9. EXEMPLES EXTREMES MEMORY
# ============================================================

memory_change_column = (
    "memory_change_pre5m_to_during_pct"
)


if memory_change_column in result_df.columns:

    print(
        "\n" + "=" * 70
    )

    print(
        "TOP MEMORY CHANGES"
    )

    print(
        "=" * 70
    )


    print(
        result_df[
            [
                "fault_id",
                "service",
                "fault_type",
                memory_change_column,
            ]
        ]
        .replace(
            [np.inf, -np.inf],
            np.nan
        )
        .dropna(
            subset=[
                memory_change_column
            ]
        )
        .sort_values(
            memory_change_column,
            ascending=False
        )
        .head(10)
        .to_string(
            index=False
        )
    )


# ============================================================
# FIN
# ============================================================

print(
    "\n" + "=" * 70
)

print(
    "DONE"
)

print(
    "=" * 70
)