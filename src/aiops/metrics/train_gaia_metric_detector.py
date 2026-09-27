"""
HARDTEC AIOps
GAIA Metrics - Metric Anomaly Detection

Objectif
--------
Détecter automatiquement les anomalies dans les séries temporelles
métriques GAIA, puis comparer les prédictions aux labels fournis
par GAIA.

IMPORTANT
---------
- label n'est JAMAIS utilisé comme feature du modèle.
- label sert uniquement de ground truth pour l'évaluation.
- Chaque fichier CSV représente une série temporelle indépendante.
- Les séries sont normalisées individuellement.

Familles utilisées pour cette première version :
    - changepoint_data
    - periodic_data
    - low_signal-to-noise_ratio_data

Méthode :
    Robust temporal features
        ↓
    Isolation Forest
        ↓
    comparaison avec label
        ↓
    Precision / Recall / F1
"""

from pathlib import Path
import glob
import warnings

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)

warnings.filterwarnings("ignore")


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]

METRIC_ROOT = PROJECT_ROOT / "metric_detection"

OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "gaia_metrics"
MODEL_DIR = PROJECT_ROOT / "models"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)


# Première version :
SELECTED_FAMILIES = [
    "changepoint_data",
    "periodic_data",
    "low_signal-to-noise_ratio_data",
]

RANDOM_STATE = 42

# Isolation Forest
N_ESTIMATORS = 300
CONTAMINATION = 0.01


# ============================================================
# CHARGEMENT DES SERIES
# ============================================================


def load_metric_series():
    """
    Charge les séries métriques des familles sélectionnées.

    Chaque fichier est considéré comme une série temporelle
    indépendante.
    """

    all_series = []

    print("=" * 70)
    print("HARDTEC AIOps - GAIA METRIC ANOMALY DETECTION")
    print("=" * 70)

    for family in SELECTED_FAMILIES:

        family_dir = METRIC_ROOT / family

        if not family_dir.exists():
            print(f"[WARNING] Dossier introuvable : {family_dir}")
            continue

        files = sorted(family_dir.glob("*.csv"))

        print()
        print(f"Famille : {family}")
        print(f"Fichiers : {len(files)}")

        for file_path in files:

            try:

                df = pd.read_csv(file_path)

                required_columns = {
                    "timestamp",
                    "value",
                    "label",
                }

                if not required_columns.issubset(df.columns):
                    print(
                        f"[SKIP] Colonnes manquantes : "
                        f"{file_path.name}"
                    )
                    continue

                # ------------------------------------------------
                # Nettoyage
                # ------------------------------------------------

                df = df[
                    [
                        "timestamp",
                        "value",
                        "label",
                    ]
                ].copy()

                df["timestamp"] = pd.to_datetime(
                    df["timestamp"],
                    unit="ms",
                    utc=True,
                    errors="coerce",
                )

                df["value"] = pd.to_numeric(
                    df["value"],
                    errors="coerce",
                )

                df["label"] = pd.to_numeric(
                    df["label"],
                    errors="coerce",
                )

                df = df.dropna(
                    subset=[
                        "timestamp",
                        "value",
                        "label",
                    ]
                )

                if len(df) < 20:
                    continue

                df["label"] = (
                    df["label"]
                    .astype(float)
                    .astype(int)
                )

                # ------------------------------------------------
                # Identité de la série
                # ------------------------------------------------

                df["series_id"] = file_path.stem
                df["family"] = family

                # ------------------------------------------------
                # Trier temporellement
                # ------------------------------------------------

                df = df.sort_values("timestamp")

                df = df.reset_index(drop=True)

                all_series.append(df)

            except Exception as exc:

                print(
                    f"[ERROR] {file_path.name}: {exc}"
                )

    if not all_series:
        raise RuntimeError(
            "Aucune série métrique valide trouvée."
        )

    result = pd.concat(
        all_series,
        ignore_index=True,
    )

    print()
    print("-" * 70)
    print("CHARGEMENT TERMINE")
    print("-" * 70)

    print(
        f"Series chargees : "
        f"{result['series_id'].nunique()}"
    )

    print(
        f"Points charges : "
        f"{len(result):,}"
    )

    print(
        f"Anomalies ground truth : "
        f"{int(result['label'].sum()):,}"
    )

    print(
        f"Taux anomalie : "
        f"{result['label'].mean() * 100:.4f}%"
    )

    return result


# ============================================================
# FEATURE ENGINEERING
# ============================================================


def create_features(df):
    """
    Construit des caractéristiques temporelles.

    label est volontairement exclu.

    Les features sont calculées indépendamment pour chaque série.
    """

    print()
    print("=" * 70)
    print("FEATURE ENGINEERING")
    print("=" * 70)

    data = df.copy()

    # ------------------------------------------------------------
    # Valeur brute
    # ------------------------------------------------------------

    data["value_feature"] = data["value"]

    # ------------------------------------------------------------
    # Variations
    # ------------------------------------------------------------

    data["diff_1"] = (
        data.groupby("series_id")["value"]
        .diff(1)
    )

    data["diff_2"] = (
        data.groupby("series_id")["value"]
        .diff(2)
    )

    # Variation relative
    previous_value = (
        data.groupby("series_id")["value"]
        .shift(1)
    )

    denominator = previous_value.abs().replace(
        0,
        np.nan,
    )

    data["relative_change"] = (
        data["diff_1"] / denominator
    )

    # ------------------------------------------------------------
    # Rolling statistics
    # ------------------------------------------------------------

    grouped = data.groupby("series_id")["value"]

    data["rolling_mean_5"] = (
        grouped.transform(
            lambda x: x.rolling(
                window=5,
                min_periods=2,
            ).mean()
        )
    )

    data["rolling_std_5"] = (
        grouped.transform(
            lambda x: x.rolling(
                window=5,
                min_periods=2,
            ).std()
        )
    )

    data["rolling_mean_15"] = (
        grouped.transform(
            lambda x: x.rolling(
                window=15,
                min_periods=3,
            ).mean()
        )
    )

    data["rolling_std_15"] = (
        grouped.transform(
            lambda x: x.rolling(
                window=15,
                min_periods=3,
            ).std()
        )
    )

    # ------------------------------------------------------------
    # Distance from rolling mean
    # ------------------------------------------------------------

    data["deviation_from_mean"] = (
        data["value"]
        - data["rolling_mean_15"]
    )

    # ------------------------------------------------------------
    # Rolling z-score
    # ------------------------------------------------------------

    std = data["rolling_std_15"].replace(
        0,
        np.nan,
    )

    data["rolling_zscore"] = (
        data["deviation_from_mean"] / std
    )

    # ------------------------------------------------------------
    # Trend
    # ------------------------------------------------------------

    data["trend_5"] = (
        data.groupby("series_id")["value"]
        .transform(
            lambda x: x.diff(5)
        )
    )

    data["trend_15"] = (
        data.groupby("series_id")["value"]
        .transform(
            lambda x: x.diff(15)
        )
    )

    # ------------------------------------------------------------
    # Valeur absolue de certaines variations
    # ------------------------------------------------------------

    data["abs_diff_1"] = data["diff_1"].abs()

    data["abs_relative_change"] = (
        data["relative_change"].abs()
    )

    data["abs_zscore"] = (
        data["rolling_zscore"].abs()
    )

    # ------------------------------------------------------------
    # Nettoyage inf / NaN
    # ------------------------------------------------------------

    data = data.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    feature_columns = [
        "value_feature",
        "diff_1",
        "diff_2",
        "relative_change",
        "rolling_mean_5",
        "rolling_std_5",
        "rolling_mean_15",
        "rolling_std_15",
        "deviation_from_mean",
        "rolling_zscore",
        "trend_5",
        "trend_15",
        "abs_diff_1",
        "abs_relative_change",
        "abs_zscore",
    ]

    # Remplacement des valeurs manquantes
    for column in feature_columns:

        data[column] = (
            data.groupby("series_id")[column]
            .transform(
                lambda x: x.fillna(
                    x.median()
                )
            )
        )

        data[column] = data[column].fillna(0)

    print(
        f"Features creees : "
        f"{len(feature_columns)}"
    )

    print(
        f"Points disponibles : "
        f"{len(data):,}"
    )

    return data, feature_columns


# ============================================================
# NORMALISATION PAR SERIE
# ============================================================


def normalize_features(df, feature_columns):
    """
    Normalisation robuste par série.

    Ceci évite qu'une série avec des valeurs énormes
    domine toutes les autres séries.
    """

    print()
    print("=" * 70)
    print("NORMALISATION PAR SERIE")
    print("=" * 70)

    data = df.copy()

    normalized_columns = []

    for column in feature_columns:

        new_column = f"{column}_norm"

        data[new_column] = (
            data.groupby("series_id")[column]
            .transform(
                lambda x: (
                    x - x.median()
                )
                / (
                    (x.quantile(0.75)
                     - x.quantile(0.25))
                    + 1e-9
                )
            )
        )

        data[new_column] = (
            data[new_column]
            .replace(
                [np.inf, -np.inf],
                np.nan,
            )
            .fillna(0)
            .clip(-20, 20)
        )

        normalized_columns.append(
            new_column
        )

    print(
        f"Features normalisees : "
        f"{len(normalized_columns)}"
    )

    return data, normalized_columns


# ============================================================
# DETECTION ISOLATION FOREST
# ============================================================


def detect_anomalies(
    df,
    normalized_columns,
):
    """
    Entraîne Isolation Forest.

    label n'est pas utilisé.
    """

    print()
    print("=" * 70)
    print("ISOLATION FOREST")
    print("=" * 70)

    data = df.copy()

    X = data[
        normalized_columns
    ].values

    print(
        f"Observations : "
        f"{X.shape[0]:,}"
    )

    print(
        f"Features : "
        f"{X.shape[1]}"
    )

    print(
        f"Contamination : "
        f"{CONTAMINATION}"
    )

    model = IsolationForest(
        n_estimators=N_ESTIMATORS,
        contamination=CONTAMINATION,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    model.fit(X)

    predictions = model.predict(X)

    # Isolation Forest :
    # -1 = anomalie
    # +1 = normal

    data["predicted_anomaly"] = (
        predictions == -1
    ).astype(int)

    # Score brut
    data["anomaly_score_raw"] = (
        -model.decision_function(X)
    )

    # Score normalisé 0-1
    score_min = data[
        "anomaly_score_raw"
    ].min()

    score_max = data[
        "anomaly_score_raw"
    ].max()

    if score_max > score_min:

        data["anomaly_score"] = (
            data["anomaly_score_raw"]
            - score_min
        ) / (
            score_max - score_min
        )

    else:

        data["anomaly_score"] = 0.0

    print()
    print(
        "Anomalies detectees : "
        f"{data['predicted_anomaly'].sum():,}"
    )

    print(
        "Taux detecte : "
        f"{data['predicted_anomaly'].mean() * 100:.4f}%"
    )

    return data, model


# ============================================================
# EVALUATION
# ============================================================


def evaluate_model(df):
    """
    Compare predicted_anomaly avec le label GAIA.
    """

    print()
    print("=" * 70)
    print("EVALUATION")
    print("=" * 70)

    y_true = df["label"].astype(int)

    y_pred = df[
        "predicted_anomaly"
    ].astype(int)

    accuracy = accuracy_score(
        y_true,
        y_pred,
    )

    precision = precision_score(
        y_true,
        y_pred,
        zero_division=0,
    )

    recall = recall_score(
        y_true,
        y_pred,
        zero_division=0,
    )

    f1 = f1_score(
        y_true,
        y_pred,
        zero_division=0,
    )

    print()
    print(
        f"Accuracy  : {accuracy:.4f}"
    )

    print(
        f"Precision : {precision:.4f}"
    )

    print(
        f"Recall    : {recall:.4f}"
    )

    print(
        f"F1-score  : {f1:.4f}"
    )

    print()
    print("Classification Report")
    print("-" * 70)

    print(
        classification_report(
            y_true,
            y_pred,
            target_names=[
                "Normal",
                "Anomaly",
            ],
            zero_division=0,
        )
    )

    print("Matrice de confusion")
    print("-" * 70)

    cm = confusion_matrix(
        y_true,
        y_pred,
    )

    print(cm)

    # --------------------------------------------------------
    # Evaluation par famille
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("EVALUATION PAR FAMILLE")
    print("=" * 70)

    family_results = []

    for family, group in df.groupby(
        "family"
    ):

        yt = group["label"].astype(int)
        yp = group[
            "predicted_anomaly"
        ].astype(int)

        family_results.append(
            {
                "family": family,
                "points": len(group),
                "ground_truth_anomalies": int(
                    yt.sum()
                ),
                "predicted_anomalies": int(
                    yp.sum()
                ),
                "precision": precision_score(
                    yt,
                    yp,
                    zero_division=0,
                ),
                "recall": recall_score(
                    yt,
                    yp,
                    zero_division=0,
                ),
                "f1": f1_score(
                    yt,
                    yp,
                    zero_division=0,
                ),
            }
        )

    family_df = pd.DataFrame(
        family_results
    )

    print(
        family_df.to_string(
            index=False
        )
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": cm.tolist(),
        "family_results": family_df,
    }


# ============================================================
# SAUVEGARDE
# ============================================================


def save_results(
    df,
    model,
    metrics,
    normalized_columns,
):
    """
    Sauvegarde :
        - predictions
        - modèle
        - métriques
    """

    print()
    print("=" * 70)
    print("SAUVEGARDE")
    print("=" * 70)

    # --------------------------------------------------------
    # Résultats
    # --------------------------------------------------------

    output_columns = [
        "timestamp",
        "series_id",
        "family",
        "value",
        "label",
        "predicted_anomaly",
        "anomaly_score",
        "anomaly_score_raw",
        "rolling_zscore",
        "relative_change",
        "trend_5",
        "trend_15",
    ]

    output_columns = [
        col
        for col in output_columns
        if col in df.columns
    ]

    output_path = (
        OUTPUT_DIR
        / "gaia_metric_anomaly_predictions.csv"
    )

    df[
        output_columns
    ].to_csv(
        output_path,
        index=False,
    )

    # --------------------------------------------------------
    # Modèle
    # --------------------------------------------------------

    model_path = (
        MODEL_DIR
        / "gaia_metric_isolation_forest.pkl"
    )

    joblib.dump(
        {
            "model": model,
            "features": normalized_columns,
            "families": SELECTED_FAMILIES,
            "contamination": CONTAMINATION,
            "random_state": RANDOM_STATE,
        },
        model_path,
    )

    # --------------------------------------------------------
    # Métriques globales
    # --------------------------------------------------------

    metrics_path = (
        OUTPUT_DIR
        / "gaia_metric_evaluation.csv"
    )

    metrics_df = pd.DataFrame(
        [
            {
                "accuracy": metrics["accuracy"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "ground_truth_anomalies": int(
                    df["label"].sum()
                ),
                "predicted_anomalies": int(
                    df["predicted_anomaly"].sum()
                ),
                "total_points": len(df),
                "ground_truth_anomaly_rate": (
                    df["label"].mean()
                ),
                "predicted_anomaly_rate": (
                    df["predicted_anomaly"].mean()
                ),
            }
        ]
    )

    metrics_df.to_csv(
        metrics_path,
        index=False,
    )

    # --------------------------------------------------------
    # Evaluation par famille
    # --------------------------------------------------------

    family_path = (
        OUTPUT_DIR
        / "gaia_metric_evaluation_by_family.csv"
    )

    metrics[
        "family_results"
    ].to_csv(
        family_path,
        index=False,
    )

    print()
    print(
        f"Predictions : {output_path}"
    )

    print(
        f"Modele      : {model_path}"
    )

    print(
        f"Evaluation  : {metrics_path}"
    )

    print(
        f"Par famille : {family_path}"
    )


# ============================================================
# MAIN
# ============================================================


def main():

    # --------------------------------------------------------
    # 1. Chargement
    # --------------------------------------------------------

    df = load_metric_series()

    # --------------------------------------------------------
    # 2. Features
    # --------------------------------------------------------

    df, feature_columns = create_features(
        df
    )

    # --------------------------------------------------------
    # 3. Normalisation
    # --------------------------------------------------------

    df, normalized_columns = (
        normalize_features(
            df,
            feature_columns,
        )
    )

    # --------------------------------------------------------
    # 4. Détection
    # --------------------------------------------------------

    df, model = detect_anomalies(
        df,
        normalized_columns,
    )

    # --------------------------------------------------------
    # 5. Evaluation
    # --------------------------------------------------------

    metrics = evaluate_model(
        df
    )

    # --------------------------------------------------------
    # 6. Sauvegarde
    # --------------------------------------------------------

    save_results(
        df,
        model,
        metrics,
        normalized_columns,
    )

    print()
    print("=" * 70)
    print("PIPELINE METRICS TERMINE")
    print("=" * 70)


if __name__ == "__main__":
    main()

