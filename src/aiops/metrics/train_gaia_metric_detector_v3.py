import os
import glob
import warnings

import numpy as np
import pandas as pd

from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

warnings.filterwarnings("ignore")


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..")
)

METRIC_DIR = os.path.join(BASE_DIR, "metric_detection")

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "gaia_metrics",
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "models",
)

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)


# Pour commencer, nous conservons les mêmes familles que V1/V2.
FAMILIES = [
    "changepoint_data",
    "periodic_data",
    "low_signal-to-noise_ratio_data",
]

TRAIN_RATIO = 0.80

# Paramètres des détecteurs
ROLLING_WINDOW = 30
MAD_WINDOW = 30
EWMA_SPAN = 20

# Seuils
ROBUST_Z_THRESHOLD = 3.5
MAD_THRESHOLD = 3.5
EWMA_THRESHOLD = 3.0

# Isolation Forest baseline
IF_CONTAMINATION = 0.01
IF_ESTIMATORS = 300
RANDOM_STATE = 42

MIN_SERIES_LENGTH = 50


# ============================================================
# UTILITAIRES
# ============================================================

def print_separator(title=None):
    print("\n" + "-" * 70)

    if title:
        print(title)
        print("-" * 70)


def safe_divide(a, b):
    return np.divide(
        a,
        b,
        out=np.zeros_like(np.asarray(a, dtype=float)),
        where=np.asarray(b) != 0,
    )


# ============================================================
# CHARGEMENT DES SERIES
# ============================================================

def load_series():

    print("=" * 70)
    print("HARDTEC AioPS - GAIA METRIC ANOMALY DETECTION V3")
    print("=" * 70)

    all_series = []

    total_points = 0
    total_anomalies = 0

    for family in FAMILIES:

        family_dir = os.path.join(METRIC_DIR, family)

        files = glob.glob(
            os.path.join(family_dir, "*.csv")
        )

        print(f"\nFamille : {family}")
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
                        f"{os.path.basename(file_path)}"
                    )
                    continue

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

                if len(df) < MIN_SERIES_LENGTH:
                    continue

                df = df.sort_values(
                    "timestamp"
                ).reset_index(drop=True)

                df["family"] = family

                # Identifiant unique de série
                df["series_id"] = (
                    family
                    + "__"
                    + os.path.splitext(
                        os.path.basename(file_path)
                    )[0]
                )

                all_series.append(df)

                total_points += len(df)
                total_anomalies += int(
                    df["label"].sum()
                )

            except Exception as e:

                print(
                    f"[ERREUR] {file_path}: {e}"
                )

    if not all_series:
        raise RuntimeError(
            "Aucune série valide trouvée."
        )

    data = pd.concat(
        all_series,
        ignore_index=True,
    )

    print_separator(
        "CHARGEMENT TERMINE"
    )

    print(
        f"Series chargees : "
        f"{data['series_id'].nunique()}"
    )

    print(
        f"Points charges : "
        f"{len(data):,}"
    )

    print(
        f"Anomalies ground truth : "
        f"{int(data['label'].sum()):,}"
    )

    print(
        f"Taux anomalie : "
        f"{data['label'].mean():.4%}"
    )

    return data


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def create_features(data):

    print_separator(
        "FEATURE ENGINEERING"
    )

    result = []

    grouped = data.groupby(
        "series_id",
        sort=False,
    )

    total_series = len(grouped)

    for counter, (series_id, group) in enumerate(
        grouped,
        start=1,
    ):

        group = group.copy()

        values = group["value"].astype(float)

        # ----------------------------------------------------
        # Différences
        # ----------------------------------------------------

        group["diff_1"] = values.diff()

        group["abs_diff_1"] = (
            group["diff_1"].abs()
        )

        # ----------------------------------------------------
        # Rolling median
        # ----------------------------------------------------

        rolling_median = (
            values
            .rolling(
                window=ROLLING_WINDOW,
                min_periods=5,
            )
            .median()
        )

        group["rolling_median"] = (
            rolling_median
        )

        # ----------------------------------------------------
        # Rolling MAD
        # ----------------------------------------------------

        rolling_mad = (
            values
            .rolling(
                window=MAD_WINDOW,
                min_periods=5,
            )
            .apply(
                lambda x: np.median(
                    np.abs(
                        x - np.median(x)
                    )
                ),
                raw=True,
            )
        )

        group["rolling_mad"] = (
            rolling_mad
        )

        # ----------------------------------------------------
        # Robust Z-Score
        #
        # z = 0.6745 * (x - median) / MAD
        # ----------------------------------------------------

        safe_mad = rolling_mad.replace(
            0,
            np.nan,
        )

        group["robust_z"] = (
            0.6745
            * (
                values
                - rolling_median
            )
            / safe_mad
        )

        group["robust_z"] = (
            group["robust_z"]
            .replace(
                [np.inf, -np.inf],
                np.nan,
            )
            .fillna(0)
        )

        # ----------------------------------------------------
        # EWMA
        # ----------------------------------------------------

        ewma = (
            values
            .ewm(
                span=EWMA_SPAN,
                adjust=False,
            )
            .mean()
        )

        group["ewma"] = ewma

        # ----------------------------------------------------
        # EWMA residual
        # ----------------------------------------------------

        group["ewma_residual"] = (
            values - ewma
        )

        # ----------------------------------------------------
        # EWMA rolling residual std
        # ----------------------------------------------------

        residual_std = (
            group["ewma_residual"]
            .rolling(
                window=ROLLING_WINDOW,
                min_periods=5,
            )
            .std()
        )

        group["ewma_std"] = (
            residual_std
            .replace(
                0,
                np.nan,
            )
            .fillna(
                residual_std.median()
            )
        )

        group["ewma_z"] = (
            group["ewma_residual"]
            / group["ewma_std"]
        )

        group["ewma_z"] = (
            group["ewma_z"]
            .replace(
                [np.inf, -np.inf],
                np.nan,
            )
            .fillna(0)
        )

        # ----------------------------------------------------
        # Variation relative
        # ----------------------------------------------------

        previous = values.shift(1)

        group["relative_change"] = (
            safe_divide(
                values - previous,
                previous.abs(),
            )
        )

        group["relative_change"] = (
            pd.Series(
                group["relative_change"],
                index=group.index,
            )
            .replace(
                [np.inf, -np.inf],
                np.nan,
            )
            .fillna(0)
        )

        # ----------------------------------------------------
        # Rolling standard deviation
        # ----------------------------------------------------

        group["rolling_std"] = (
            values
            .rolling(
                window=ROLLING_WINDOW,
                min_periods=5,
            )
            .std()
            .fillna(0)
        )

        # ----------------------------------------------------
        # Déviation par rapport à la médiane
        # ----------------------------------------------------

        group["median_deviation"] = (
            values
            - rolling_median
        )

        # ----------------------------------------------------
        # Quantile-based local range
        # ----------------------------------------------------

        rolling_q25 = (
            values
            .rolling(
                window=ROLLING_WINDOW,
                min_periods=5,
            )
            .quantile(0.25)
        )

        rolling_q75 = (
            values
            .rolling(
                window=ROLLING_WINDOW,
                min_periods=5,
            )
            .quantile(0.75)
        )

        group["iqr"] = (
            rolling_q75 - rolling_q25
        )

        # ----------------------------------------------------
        # Nettoyage
        # ----------------------------------------------------

        numeric_columns = [
            "diff_1",
            "abs_diff_1",
            "rolling_median",
            "rolling_mad",
            "robust_z",
            "ewma",
            "ewma_residual",
            "ewma_std",
            "ewma_z",
            "relative_change",
            "rolling_std",
            "median_deviation",
            "iqr",
        ]

        for column in numeric_columns:

            group[column] = (
                pd.to_numeric(
                    group[column],
                    errors="coerce",
                )
                .replace(
                    [np.inf, -np.inf],
                    np.nan,
                )
                .fillna(0)
            )

        result.append(group)

        if counter % 25 == 0:
            print(
                f"Séries traitées : "
                f"{counter}/{total_series}"
            )

    data = pd.concat(
        result,
        ignore_index=True,
    )

    print(
        f"\nFeatures créées : "
        f"{len(data.columns)} colonnes"
    )

    return data


# ============================================================
# TEMPORAL SPLIT
# ============================================================

def temporal_split(data):

    print_separator(
        "SPLIT TEMPOREL"
    )

    data = data.copy()

    data["split"] = "train"

    for series_id, group in data.groupby(
        "series_id",
        sort=False,
    ):

        indices = group.index.tolist()

        split_index = int(
            len(indices) * TRAIN_RATIO
        )

        # Protection
        split_index = max(
            1,
            min(
                split_index,
                len(indices) - 1,
            ),
        )

        test_indices = indices[
            split_index:
        ]

        data.loc[
            test_indices,
            "split",
        ] = "test"

    train = data[
        data["split"] == "train"
    ]

    test = data[
        data["split"] == "test"
    ]

    print(
        f"Train : {len(train):,} points"
    )

    print(
        f"Test  : {len(test):,} points"
    )

    print(
        f"Train anomalies : "
        f"{int(train['label'].sum()):,}"
    )

    print(
        f"Test anomalies : "
        f"{int(test['label'].sum()):,}"
    )

    print(
        f"Train anomaly rate : "
        f"{train['label'].mean():.4%}"
    )

    print(
        f"Test anomaly rate : "
        f"{test['label'].mean():.4%}"
    )

    return data


# ============================================================
# ROBUST Z-SCORE DETECTOR
# ============================================================

def robust_zscore_detector(data):

    score = (
        data["robust_z"]
        .abs()
    )

    prediction = (
        score >= ROBUST_Z_THRESHOLD
    ).astype(int)

    return prediction, score


# ============================================================
# ROLLING MAD DETECTOR
# ============================================================

def rolling_mad_detector(data):

    deviation = (
        data["median_deviation"]
        .abs()
    )

    mad = (
        data["rolling_mad"]
        .replace(
            0,
            np.nan,
        )
    )

    score = (
        deviation
        / (mad + 1e-9)
    )

    score = (
        score
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
        .fillna(0)
    )

    prediction = (
        score >= MAD_THRESHOLD
    ).astype(int)

    return prediction, score


# ============================================================
# EWMA DETECTOR
# ============================================================

def ewma_detector(data):

    score = (
        data["ewma_z"]
        .abs()
    )

    prediction = (
        score >= EWMA_THRESHOLD
    ).astype(int)

    return prediction, score


# ============================================================
# ISOLATION FOREST
# ============================================================

def isolation_forest_detector(
    train,
    test,
    feature_columns,
):

    print_separator(
        "ISOLATION FOREST BASELINE"
    )

    X_train = (
        train[
            feature_columns
        ]
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
        .fillna(0)
    )

    X_test = (
        test[
            feature_columns
        ]
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
        .fillna(0)
    )

    model = IsolationForest(
        n_estimators=IF_ESTIMATORS,
        contamination=IF_CONTAMINATION,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    print(
        f"Train observations : "
        f"{len(X_train):,}"
    )

    print(
        f"Test observations  : "
        f"{len(X_test):,}"
    )

    print(
        f"Features : "
        f"{len(feature_columns)}"
    )

    print(
        "\nEntraînement..."
    )

    model.fit(X_train)

    print(
        "Entraînement terminé."
    )

    train_raw = (
        model.decision_function(
            X_train
        )
    )

    test_raw = (
        model.decision_function(
            X_test
        )
    )

    # Plus grand = plus anomal
    train_score = -train_raw
    test_score = -test_raw

    # Seuil basé sur le quantile du train
    threshold = np.quantile(
        train_score,
        1 - IF_CONTAMINATION,
    )

    train_prediction = (
        train_score >= threshold
    ).astype(int)

    test_prediction = (
        test_score >= threshold
    ).astype(int)

    return (
        model,
        train_prediction,
        test_prediction,
        train_score,
        test_score,
    )


# ============================================================
# EVALUATION
# ============================================================

def evaluate(
    y_true,
    y_pred,
    detector_name,
    split_name,
):

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

    print(
        f"\n{detector_name} - {split_name}"
    )

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

    return {
        "detector": detector_name,
        "split": split_name,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_anomalies": int(
            np.sum(y_true)
        ),
        "predicted_anomalies": int(
            np.sum(y_pred)
        ),
    }


# ============================================================
# EVALUATION PAR FAMILLE
# ============================================================

def evaluate_by_family(
    data,
    prediction_column,
    detector_name,
):

    rows = []

    test = data[
        data["split"] == "test"
    ]

    for family, group in test.groupby(
        "family"
    ):

        y_true = group["label"].astype(int)

        y_pred = group[
            prediction_column
        ].astype(int)

        rows.append(
            {
                "detector": detector_name,
                "family": family,
                "points": len(group),
                "ground_truth_anomalies": int(
                    y_true.sum()
                ),
                "predicted_anomalies": int(
                    y_pred.sum()
                ),
                "precision": precision_score(
                    y_true,
                    y_pred,
                    zero_division=0,
                ),
                "recall": recall_score(
                    y_true,
                    y_pred,
                    zero_division=0,
                ),
                "f1": f1_score(
                    y_true,
                    y_pred,
                    zero_division=0,
                ),
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # 1. LOAD
    # --------------------------------------------------------

    data = load_series()

    # --------------------------------------------------------
    # 2. FEATURES
    # --------------------------------------------------------

    data = create_features(
        data
    )

    # --------------------------------------------------------
    # 3. TEMPORAL SPLIT
    # --------------------------------------------------------

    data = temporal_split(
        data
    )

    train = data[
        data["split"] == "train"
    ].copy()

    test = data[
        data["split"] == "test"
    ].copy()

    # --------------------------------------------------------
    # 4. DETECTEURS STATISTIQUES
    # --------------------------------------------------------

    print_separator(
        "DETECTEURS TIME-SERIES"
    )

    # Robust Z
    train_pred_z, train_score_z = (
        robust_zscore_detector(
            train
        )
    )

    test_pred_z, test_score_z = (
        robust_zscore_detector(
            test
        )
    )

    # Rolling MAD
    train_pred_mad, train_score_mad = (
        rolling_mad_detector(
            train
        )
    )

    test_pred_mad, test_score_mad = (
        rolling_mad_detector(
            test
        )
    )

    # EWMA
    train_pred_ewma, train_score_ewma = (
        ewma_detector(
            train
        )
    )

    test_pred_ewma, test_score_ewma = (
        ewma_detector(
            test
        )
    )

    # --------------------------------------------------------
    # 5. ISOLATION FOREST
    # --------------------------------------------------------

    feature_columns = [
        "value",
        "diff_1",
        "abs_diff_1",
        "rolling_median",
        "rolling_mad",
        "robust_z",
        "ewma",
        "ewma_residual",
        "ewma_std",
        "ewma_z",
        "relative_change",
        "rolling_std",
        "median_deviation",
        "iqr",
    ]

    (
        isolation_model,
        train_pred_if,
        test_pred_if,
        train_score_if,
        test_score_if,
    ) = isolation_forest_detector(
        train,
        test,
        feature_columns,
    )

    # --------------------------------------------------------
    # 6. AJOUT DES PREDICTIONS
    # --------------------------------------------------------

    train = train.copy()
    test = test.copy()

    train["robust_z_prediction"] = (
        train_pred_z
    )

    test["robust_z_prediction"] = (
        test_pred_z
    )

    train["rolling_mad_prediction"] = (
        train_pred_mad
    )

    test["rolling_mad_prediction"] = (
        test_pred_mad
    )

    train["ewma_prediction"] = (
        train_pred_ewma
    )

    test["ewma_prediction"] = (
        test_pred_ewma
    )

    train["isolation_forest_prediction"] = (
        train_pred_if
    )

    test["isolation_forest_prediction"] = (
        test_pred_if
    )

    train["robust_z_score"] = (
        train_score_z
    )

    test["robust_z_score"] = (
        test_score_z
    )

    train["rolling_mad_score"] = (
        train_score_mad
    )

    test["rolling_mad_score"] = (
        test_score_mad
    )

    train["ewma_score"] = (
        train_score_ewma
    )

    test["ewma_score"] = (
        test_score_ewma
    )

    train["isolation_forest_score"] = (
        train_score_if
    )

    test["isolation_forest_score"] = (
        test_score_if
    )

    combined = pd.concat(
        [
            train,
            test,
        ],
        ignore_index=True,
    )

    # --------------------------------------------------------
    # 7. EVALUATION GLOBALE
    # --------------------------------------------------------

    print_separator(
        "EVALUATION GLOBALE"
    )

    detectors = [
        (
            "Robust_Z",
            "robust_z_prediction",
        ),
        (
            "Rolling_MAD",
            "rolling_mad_prediction",
        ),
        (
            "EWMA",
            "ewma_prediction",
        ),
        (
            "Isolation_Forest_V2",
            "isolation_forest_prediction",
        ),
    ]

    evaluation_rows = []

    for detector_name, prediction_column in detectors:

        y_true = test["label"].astype(int)

        y_pred = test[
            prediction_column
        ].astype(int)

        metrics = evaluate(
            y_true,
            y_pred,
            detector_name,
            "TEST",
        )

        evaluation_rows.append(
            metrics
        )

    evaluation = pd.DataFrame(
        evaluation_rows
    )

    # --------------------------------------------------------
    # 8. EVALUATION PAR FAMILLE
    # --------------------------------------------------------

    print_separator(
        "EVALUATION PAR FAMILLE"
    )

    family_results = []

    for detector_name, prediction_column in detectors:

        family_result = evaluate_by_family(
            combined,
            prediction_column,
            detector_name,
        )

        family_results.append(
            family_result
        )

    evaluation_by_family = pd.concat(
        family_results,
        ignore_index=True,
    )

    print(
        evaluation_by_family.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # 9. MEILLEUR MODELE
    # --------------------------------------------------------

    print_separator(
        "SELECTION DU MEILLEUR DETECTEUR"
    )

    ranking = (
        evaluation
        .sort_values(
            "f1",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    print(
        ranking[
            [
                "detector",
                "precision",
                "recall",
                "f1",
            ]
        ].to_string(
            index=False
        )
    )

    best_detector = ranking.iloc[0]

    print(
        "\nMEILLEUR DETECTEUR : "
        f"{best_detector['detector']}"
    )

    print(
        f"Precision : "
        f"{best_detector['precision']:.4f}"
    )

    print(
        f"Recall : "
        f"{best_detector['recall']:.4f}"
    )

    print(
        f"F1 : "
        f"{best_detector['f1']:.4f}"
    )

    # --------------------------------------------------------
    # 10. RAPPORT DETAILLE DU MEILLEUR DETECTEUR
    # --------------------------------------------------------

    best_name = best_detector[
        "detector"
    ]

    best_column = dict(
        detectors
    )[best_name]

    y_true = test["label"].astype(int)

    y_pred = test[
        best_column
    ].astype(int)

    print_separator(
        f"RAPPORT : {best_name}"
    )

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

    print(
        "Matrice de confusion"
    )

    print(
        confusion_matrix(
            y_true,
            y_pred,
        )
    )

    # --------------------------------------------------------
    # 11. SAUVEGARDE
    # --------------------------------------------------------

    prediction_path = os.path.join(
        OUTPUT_DIR,
        "gaia_metric_anomaly_predictions_v3.csv",
    )

    evaluation_path = os.path.join(
        OUTPUT_DIR,
        "gaia_metric_evaluation_v3.csv",
    )

    family_path = os.path.join(
        OUTPUT_DIR,
        "gaia_metric_evaluation_by_family_v3.csv",
    )

    ranking_path = os.path.join(
        OUTPUT_DIR,
        "gaia_metric_detector_comparison_v3.csv",
    )

    combined.to_csv(
        prediction_path,
        index=False,
    )

    evaluation.to_csv(
        evaluation_path,
        index=False,
    )

    evaluation_by_family.to_csv(
        family_path,
        index=False,
    )

    ranking.to_csv(
        ranking_path,
        index=False,
    )

    # --------------------------------------------------------
    # 12. SAUVEGARDE ISOLATION FOREST
    # --------------------------------------------------------

    import joblib

    model_path = os.path.join(
        MODEL_DIR,
        "gaia_metric_isolation_forest_v3.pkl",
    )

    joblib.dump(
        {
            "model": isolation_model,
            "features": feature_columns,
            "contamination": IF_CONTAMINATION,
            "random_state": RANDOM_STATE,
        },
        model_path,
    )

    print_separator(
        "SAUVEGARDE"
    )

    print(
        f"Predictions : "
        f"{prediction_path}"
    )

    print(
        f"Evaluation : "
        f"{evaluation_path}"
    )

    print(
        f"Par famille : "
        f"{family_path}"
    )

    print(
        f"Comparaison : "
        f"{ranking_path}"
    )

    print(
        f"Modele IF : "
        f"{model_path}"
    )

    print("\n" + "=" * 70)
    print(
        "METRICS V3 TERMINE"
    )
    print("=" * 70)

    print(
        "\nIMPORTANT :"
    )

    print(
        "- Le label GAIA n'est PAS utilise comme feature."
    )

    print(
        "- Le test est temporel : les 20% les plus récents "
        "de chaque série sont réservés au test."
    )

    print(
        "- Le meilleur détecteur est sélectionné "
        "sur le F1-score du TEST."
    )


if __name__ == "__main__":
    main()

