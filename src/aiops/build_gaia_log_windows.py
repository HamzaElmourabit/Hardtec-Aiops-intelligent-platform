import os
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

ERROR_PATH = "./log/log semantics anomaly detection/error.csv"
INFO_PATH = "./log/log semantics anomaly detection/info.csv"

OUTPUT_DIR = "./data/processed"
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "gaia_log_windows.csv")

BLOCK_GAP_MINUTES = 30
MIN_BLOCK_LOGS = 50
WINDOW_MINUTES = 5


# ============================================================
# 1. CHARGEMENT DES LOGS
# ============================================================

print("=" * 70)
print("GAIA - CONSTRUCTION DES FENÊTRES TEMPORELLES")
print("=" * 70)

print("\n[1/6] Chargement des fichiers...")

error_df = pd.read_csv(ERROR_PATH)
info_df = pd.read_csv(INFO_PATH)

error_df["label"] = "error"
info_df["label"] = "info"

df = pd.concat(
    [error_df, info_df],
    ignore_index=True
)

print(f"ERROR logs : {len(error_df):,}")
print(f"INFO logs  : {len(info_df):,}")
print(f"TOTAL      : {len(df):,}")


# ============================================================
# 2. TIMESTAMP
# ============================================================

print("\n[2/6] Conversion des timestamps...")

df["datetime"] = pd.to_datetime(
    df["timestamp"],
    unit="ms",
    utc=True
)

df = df.sort_values("datetime").reset_index(drop=True)

print("Début :", df["datetime"].min())
print("Fin   :", df["datetime"].max())


# ============================================================
# 3. CONSTRUCTION DES BLOCS ACTIFS
# ============================================================

print("\n[3/6] Construction des blocs actifs...")

time_gap = df["datetime"].diff()

df["new_block"] = (
    time_gap > pd.Timedelta(minutes=BLOCK_GAP_MINUTES)
).astype(int)

df["block_id"] = df["new_block"].cumsum()

block_stats = (
    df.groupby("block_id")
    .agg(
        block_start=("datetime", "min"),
        block_end=("datetime", "max"),
        total_logs=("datetime", "size"),
    )
    .reset_index()
)

block_stats["duration"] = (
    block_stats["block_end"]
    - block_stats["block_start"]
)

usable_blocks = block_stats[
    block_stats["total_logs"] >= MIN_BLOCK_LOGS
].copy()

usable_block_ids = set(
    usable_blocks["block_id"]
)

df = df[
    df["block_id"].isin(usable_block_ids)
].copy()

print(f"Blocs totaux        : {len(block_stats)}")
print(f"Blocs utilisables    : {len(usable_blocks)}")
print(f"Logs conservés      : {len(df):,}")


# ============================================================
# 4. CONSTRUCTION DES FENÊTRES DE 5 MINUTES
# ============================================================

print("\n[4/6] Construction des fenêtres de 5 minutes...")

# Arrondi au début de la fenêtre de 5 minutes
df["window_start"] = (
    df["datetime"]
    .dt.floor(f"{WINDOW_MINUTES}min")
)

df["window_end"] = (
    df["window_start"]
    + pd.Timedelta(minutes=WINDOW_MINUTES)
)

# IMPORTANT :
# On garde le block_id afin de ne jamais mélanger
# deux périodes d'activité différentes.

window_rows = []

for (block_id, window_start), group in df.groupby(
    ["block_id", "window_start"]
):

    window_end = (
        window_start
        + pd.Timedelta(minutes=WINDOW_MINUTES)
    )

    total_logs = len(group)

    error_count = (
        group["label"]
        .eq("error")
        .sum()
    )

    info_count = (
        group["label"]
        .eq("info")
        .sum()
    )

    error_rate = (
        error_count / total_logs
        if total_logs > 0
        else 0
    )

    unique_sources = (
        group["source"]
        .nunique()
    )

    unique_hosts = (
        group["host"]
        .nunique()
    )

    unique_exception_types = (
        group["exception_type"]
        .nunique(dropna=True)
    )

    window_rows.append(
        {
            "block_id": block_id,
            "window_start": window_start,
            "window_end": window_end,

            "total_logs": total_logs,
            "error_count": error_count,
            "info_count": info_count,
            "error_rate": error_rate,

            "unique_sources": unique_sources,
            "unique_hosts": unique_hosts,
            "unique_exception_types": unique_exception_types,
        }
    )


windows = pd.DataFrame(window_rows)

windows = windows.sort_values(
    ["block_id", "window_start"]
).reset_index(drop=True)


# ============================================================
# 5. FEATURES TEMPORELLES
# ============================================================

print("\n[5/6] Ajout des variables temporelles...")

windows["hour"] = (
    windows["window_start"].dt.hour
)

windows["day_of_week"] = (
    windows["window_start"].dt.dayofweek
)

windows["is_weekend"] = (
    windows["day_of_week"] >= 5
).astype(int)


# ============================================================
# 6. VARIABLES DE CONTEXTE
# ============================================================

print("\n[6/6] Calcul des variables supplémentaires...")

# Taux d'erreur précédent
windows["previous_error_rate"] = (
    windows
    .groupby("block_id")["error_rate"]
    .shift(1)
)

# Variation du taux d'erreur
windows["error_rate_delta"] = (
    windows["error_rate"]
    - windows["previous_error_rate"]
)

# Volume précédent
windows["previous_total_logs"] = (
    windows
    .groupby("block_id")["total_logs"]
    .shift(1)
)

# Variation du volume
windows["volume_delta"] = (
    windows["total_logs"]
    - windows["previous_total_logs"]
)

# Moyenne mobile sur 3 fenêtres
windows["rolling_error_rate_3"] = (
    windows
    .groupby("block_id")["error_rate"]
    .transform(
        lambda x: x.rolling(
            3,
            min_periods=1
        ).mean()
    )
)

windows["rolling_volume_3"] = (
    windows
    .groupby("block_id")["total_logs"]
    .transform(
        lambda x: x.rolling(
            3,
            min_periods=1
        ).mean()
    )
)

# Remplacement des NaN créés par les premières fenêtres
windows = windows.fillna(0)


# ============================================================
# LABELS POUR VALIDATION
# ============================================================

# ATTENTION :
# Ces colonnes servent à l'évaluation et à l'analyse.
# Elles ne doivent PAS être utilisées comme features
# d'un modèle de détection.

windows["has_error"] = (
    windows["error_count"] > 0
).astype(int)


# ============================================================
# SAUVEGARDE
# ============================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

windows.to_csv(
    OUTPUT_PATH,
    index=False
)


# ============================================================
# RAPPORT
# ============================================================

print("\n" + "=" * 70)
print("RESULTAT")
print("=" * 70)

print(f"Fenêtres créées       : {len(windows):,}")
print(f"Blocs utilisés        : {windows['block_id'].nunique():,}")
print(f"Fenêtre temporelle    : {WINDOW_MINUTES} minutes")

print("\nDistribution des logs par fenêtre :")

print(
    windows["total_logs"]
    .describe()
    .to_string()
)

print("\nDistribution du taux d'erreur :")

print(
    windows["error_rate"]
    .describe()
    .to_string()
)

print("\nFenêtres avec au moins une erreur :")

print(
    windows["has_error"]
    .value_counts()
    .sort_index()
)

print("\nNombre de sources :")

print(
    windows["unique_sources"]
    .describe()
    .to_string()
)

print("\nNombre de hosts :")

print(
    windows["unique_hosts"]
    .describe()
    .to_string()
)

print("\nFichier sauvegardé :")
print(OUTPUT_PATH)

print("\nTerminé.")