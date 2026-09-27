from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(".")

PREDICTIONS_FILE = (
    BASE_DIR / "data" / "processed" / "gaia_log_semantic_predictions.csv"
)

WINDOWS_FILE = (
    BASE_DIR / "data" / "processed" / "gaia_log_windows.csv"
)

OUTPUT_FILE = (
    BASE_DIR / "data" / "processed" / "gaia_temporal_aiops_features.csv"
)


# ============================================================
# HELPERS
# ============================================================


def normalize_timestamp(series):
    """
    Convertit les timestamps GAIA en datetime UTC.

    GAIA utilise généralement des timestamps Unix en millisecondes.
    """
    numeric = pd.to_numeric(series, errors="coerce")

    return pd.to_datetime(
        numeric,
        unit="ms",
        errors="coerce",
        utc=True,
    )


def build_window_key(df):
    """
    Crée une clé temporelle permettant d'associer les logs
    aux fenêtres de 5 minutes.

    IMPORTANT :
    On utilise directement block_id + window_start
    provenant de gaia_log_windows.csv.
    """

    result = df.copy()

    result["window_start"] = pd.to_datetime(
        result["window_start"],
        utc=True,
        errors="coerce",
    )

    result["window_key"] = (
        result["block_id"].astype(str)
        + "_"
        + result["window_start"].astype(str)
    )

    return result


# ============================================================
# MAIN
# ============================================================


def main():

    print("=" * 70)
    print("GAIA - SEMANTIC ANOMALIES -> TEMPORAL AIOPS FEATURES")
    print("=" * 70)

    # ========================================================
    # [1/6] CHARGEMENT
    # ========================================================

    print("\n[1/6] Chargement des fichiers...")

    if not PREDICTIONS_FILE.exists():
        raise FileNotFoundError(
            f"Fichier introuvable : {PREDICTIONS_FILE}"
        )

    if not WINDOWS_FILE.exists():
        raise FileNotFoundError(
            f"Fichier introuvable : {WINDOWS_FILE}"
        )

    logs = pd.read_csv(PREDICTIONS_FILE)
    windows = pd.read_csv(WINDOWS_FILE)

    print(f"Logs prédits : {len(logs):,}")
    print(f"Fenêtres     : {len(windows):,}")

    # ========================================================
    # VERIFICATION COLONNES
    # ========================================================

    required_log_columns = [
        "timestamp",
        "source",
        "host",
        "is_semantic_anomaly",
        "decision_score",
        "anomaly_confidence",
    ]

    missing_logs = [
        c for c in required_log_columns
        if c not in logs.columns
    ]

    if missing_logs:
        raise ValueError(
            "Colonnes manquantes dans le fichier des prédictions : "
            + ", ".join(missing_logs)
        )

    required_window_columns = [
        "block_id",
        "window_start",
        "window_end",
    ]

    missing_windows = [
        c for c in required_window_columns
        if c not in windows.columns
    ]

    if missing_windows:
        raise ValueError(
            "Colonnes manquantes dans le fichier des fenêtres : "
            + ", ".join(missing_windows)
        )

    # ========================================================
    # [2/6] TIMESTAMPS
    # ========================================================

    print("\n[2/6] Préparation des timestamps...")

    logs["timestamp_dt"] = normalize_timestamp(logs["timestamp"])

    windows["window_start"] = pd.to_datetime(
        windows["window_start"],
        utc=True,
        errors="coerce",
    )

    windows["window_end"] = pd.to_datetime(
        windows["window_end"],
        utc=True,
        errors="coerce",
    )

    invalid_logs = logs["timestamp_dt"].isna().sum()

    if invalid_logs > 0:
        print(
            f"ATTENTION : {invalid_logs:,} logs ont un timestamp invalide."
        )

        logs = logs.dropna(subset=["timestamp_dt"]).copy()

    print(
        "Période logs : "
        f"{logs['timestamp_dt'].min()} -> "
        f"{logs['timestamp_dt'].max()}"
    )

    # ========================================================
    # [3/6] RECONSTRUCTION DES BLOCS
    # ========================================================

    print("\n[3/6] Reconstruction des blocs temporels...")

    # Trier exactement comme lors de la construction des fenêtres
    logs = logs.sort_values("timestamp_dt").reset_index(drop=True)

    gap = logs["timestamp_dt"].diff()

    # Un nouveau bloc commence lorsqu'il y a plus de 30 minutes
    logs["new_block"] = (
        gap.isna()
        | (gap > pd.Timedelta(minutes=30))
    )

    logs["reconstructed_block_id"] = (
        logs["new_block"].cumsum() - 1
    )

    print(
        "Blocs reconstruits : "
        f"{logs['reconstructed_block_id'].nunique():,}"
    )

    # ========================================================
    # IMPORTANT :
    # vérifier la correspondance des block_id
    # ========================================================

    window_block_ids = set(
        pd.to_numeric(
            windows["block_id"],
            errors="coerce",
        )
        .dropna()
        .astype(int)
        .unique()
    )

    log_block_ids = set(
        logs["reconstructed_block_id"]
        .astype(int)
        .unique()
    )

    common_blocks = window_block_ids.intersection(
        log_block_ids
    )

    print(
        f"Blocs dans fenêtres : {len(window_block_ids):,}"
    )

    print(
        f"Blocs dans logs     : {len(log_block_ids):,}"
    )

    print(
        f"Blocs communs       : {len(common_blocks):,}"
    )

    if len(common_blocks) == 0:
        raise RuntimeError(
            "\nERREUR : aucun block_id commun.\n"
            "Les fenêtres et les logs ne sont pas construits "
            "avec la même logique temporelle."
        )

    # ========================================================
    # [4/6] ATTRIBUTION DIRECTE AUX FENETRES
    # ========================================================

    print("\n[4/6] Attribution des logs aux fenêtres...")

    # --------------------------------------------------------
    # On prépare les fenêtres originales.
    # --------------------------------------------------------

    windows["block_id"] = pd.to_numeric(
        windows["block_id"],
        errors="coerce",
    ).astype("Int64")

    windows["window_start"] = pd.to_datetime(
        windows["window_start"],
        utc=True,
        errors="coerce",
    )

    windows["window_end"] = pd.to_datetime(
        windows["window_end"],
        utc=True,
        errors="coerce",
    )

    # --------------------------------------------------------
    # Attribution par :
    #
    #   reconstructed_block_id
    #   +
    #   position de 5 minutes depuis le début du bloc
    #
    # MAIS nous prenons comme référence les fenêtres
    # originales afin d'éviter le décalage constaté.
    # --------------------------------------------------------

    window_starts = (
        windows[
            [
                "block_id",
                "window_start",
                "window_end",
            ]
        ]
        .dropna(subset=["block_id", "window_start"])
        .copy()
    )

    # Pour chaque bloc, récupérer le début de la première fenêtre
    block_first_window = (
        window_starts
        .groupby("block_id")["window_start"]
        .min()
        .to_dict()
    )

    # --------------------------------------------------------
    # Construire la clé de fenêtre pour chaque log.
    # --------------------------------------------------------

    logs["block_id"] = (
        logs["reconstructed_block_id"]
        .astype(int)
    )

    logs["block_start"] = logs["block_id"].map(
        block_first_window
    )

    logs = logs.dropna(
        subset=["block_start"]
    ).copy()

    elapsed = (
        logs["timestamp_dt"]
        - logs["block_start"]
    )

    # Position de la fenêtre de 5 minutes
    logs["window_offset"] = (
        elapsed.dt.total_seconds() // 300
    ).astype(int)

    logs["window_start"] = (
        logs["block_start"]
        + pd.to_timedelta(
            logs["window_offset"] * 5,
            unit="min",
        )
    )

    # --------------------------------------------------------
    # Clé finale
    # --------------------------------------------------------

    logs["window_key"] = (
        logs["block_id"].astype(str)
        + "_"
        + logs["window_start"].astype(str)
    )

    windows["window_key"] = (
        windows["block_id"].astype(str)
        + "_"
        + windows["window_start"].astype(str)
    )

    valid_window_keys = set(
        windows["window_key"]
    )

    logs["valid_window"] = logs["window_key"].isin(
        valid_window_keys
    )

    matched_logs = logs[
        logs["valid_window"]
    ].copy()

    print(
        f"Logs attribués à une fenêtre : "
        f"{len(matched_logs):,}"
    )

    print(
        f"Logs non attribués : "
        f"{len(logs) - len(matched_logs):,}"
    )

    # ========================================================
    # PROTECTION CRITIQUE
    # ========================================================

    if len(matched_logs) == 0:

        print("\nQuelques exemples de clés calculées :")
        print(
            logs[
                [
                    "timestamp_dt",
                    "block_id",
                    "window_start",
                    "window_key",
                ]
            ].head(10).to_string(index=False)
        )

        print("\nQuelques clés de fenêtres :")
        print(
            windows[
                [
                    "block_id",
                    "window_start",
                    "window_key",
                ]
            ].head(10).to_string(index=False)
        )

        raise RuntimeError(
            "\nERREUR CRITIQUE : "
            "aucun log n'a pu être attribué aux fenêtres.\n"
            "Le fichier final n'a PAS été généré."
        )

    # ========================================================
    # [5/6] AGREGATION SEMANTIQUE
    # ========================================================

    print("\n[5/6] Agrégation des anomalies sémantiques...")

    matched_logs["is_semantic_anomaly"] = (
        matched_logs["is_semantic_anomaly"]
        .astype(bool)
    )

    aggregation = (
        matched_logs
        .groupby("window_key")
        .agg(
            semantic_total_logs=(
                "timestamp_dt",
                "count",
            ),

            semantic_anomaly_count=(
                "is_semantic_anomaly",
                "sum",
            ),

            semantic_mean_score=(
                "decision_score",
                "mean",
            ),

            semantic_max_score=(
                "decision_score",
                "max",
            ),

            semantic_mean_confidence=(
                "anomaly_confidence",
                "mean",
            ),

            semantic_max_confidence=(
                "anomaly_confidence",
                "max",
            ),

            semantic_unique_sources=(
                "source",
                "nunique",
            ),

            semantic_unique_hosts=(
                "host",
                "nunique",
            ),
        )
        .reset_index()
    )

    aggregation[
        "semantic_anomaly_rate"
    ] = (
        aggregation["semantic_anomaly_count"]
        / aggregation["semantic_total_logs"]
    )

    # ========================================================
    # [6/6] FUSION
    # ========================================================

    print("\n[6/6] Fusion avec les fenêtres temporelles...")

    final = windows.merge(
        aggregation,
        on="window_key",
        how="left",
    )

    # Fenêtres sans logs → valeurs 0
    numeric_columns = [
        "semantic_total_logs",
        "semantic_anomaly_count",
        "semantic_mean_score",
        "semantic_max_score",
        "semantic_mean_confidence",
        "semantic_max_confidence",
        "semantic_unique_sources",
        "semantic_unique_hosts",
        "semantic_anomaly_rate",
    ]

    for column in numeric_columns:
        final[column] = final[column].fillna(0)

    # ========================================================
    # VALIDATION
    # ========================================================

    total_logs = final[
        "semantic_total_logs"
    ].sum()

    total_anomalies = final[
        "semantic_anomaly_count"
    ].sum()

    global_anomaly_rate = (
        total_anomalies / total_logs
        if total_logs > 0
        else 0
    )

    windows_with_anomalies = (
        final["semantic_anomaly_count"] > 0
    ).sum()

    windows_without_anomalies = (
        final["semantic_anomaly_count"] == 0
    ).sum()

    # ========================================================
    # RESULTATS
    # ========================================================

    print("\n" + "=" * 70)
    print("RESULTATS")
    print("=" * 70)

    print(
        f"\nFenêtres finales : "
        f"{len(final):,}"
    )

    print(
        f"Logs couverts par agrégation : "
        f"{int(total_logs):,}"
    )

    print(
        f"Anomalies sémantiques détectées : "
        f"{int(total_anomalies):,}"
    )

    print(
        f"Taux global d'anomalies : "
        f"{global_anomaly_rate:.4%}"
    )

    print(
        f"\nNombre de fenêtres avec anomalies : "
        f"{windows_with_anomalies:,}"
    )

    print(
        f"Nombre de fenêtres sans anomalies : "
        f"{windows_without_anomalies:,}"
    )

    print("\nStatistiques anomaly rate :")

    print(
        final[
            "semantic_anomaly_rate"
        ].describe()
    )

    print(
        "\nTop 10 fenêtres par densité d'anomalies :"
    )

    top10 = (
        final[
            [
                "block_id",
                "window_start",
                "semantic_total_logs",
                "semantic_anomaly_count",
                "semantic_anomaly_rate",
                "semantic_mean_confidence",
            ]
        ]
        .sort_values(
            [
                "semantic_anomaly_rate",
                "semantic_anomaly_count",
            ],
            ascending=False,
        )
        .head(10)
    )

    print(
        top10.to_string(index=False)
    )

    # ========================================================
    # VALIDATION FINALE
    # ========================================================

    if total_logs == 0:
        raise RuntimeError(
            "\nERREUR : total_logs = 0. "
            "Le fichier final ne sera pas sauvegardé."
        )

    # ========================================================
    # NETTOYAGE
    # ========================================================

    final = final.drop(
        columns=["window_key"],
        errors="ignore",
    )

    # ========================================================
    # SAUVEGARDE
    # ========================================================

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    final.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print("\n" + "=" * 70)
    print("FICHIER CREE")
    print("=" * 70)

    print(
        f"\n{OUTPUT_FILE}"
    )

    print("\nTerminé.")


if __name__ == "__main__":
    main()