# ============================================================
# HARDTEC AioPS - GAIA Metrics Anomaly Detection V2
# ============================================================
#
# Objectifs :
#   1. Traiter chaque série temporelle indépendamment
#   2. Construire une baseline temporelle robuste
#   3. Détecter les écarts anormaux sans utiliser "label"
#   4. Évaluer uniquement avec le label GAIA
#   5. Utiliser un split temporel Train/Test
#
# Dataset :
#   GAIA metric_detection
#
# Important :
#   label = ground truth uniquement
#   label N'EST JAMAIS utilisé comme feature
# ============================================================

from pathlib import Path
import warnings

import joblib
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

PROJECT_ROOT = Path(__file__).resolve().parents[3]

METRIC_ROOT = PROJECT_ROOT / "metric_detection"

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "gaia_metrics"
)

MODEL_DIR = PROJECT_ROOT / "models"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)


# Familles utilisées dans V2
FAMILIES = [
    "changepoint_data",
    "periodic_data",
    "low_signal-to-noise_ratio_data",
]

# Pour éviter les séries trop courtes
MIN_SERIES_LENGTH = 50

# 80% passé / 20% futur
TRAIN_RATIO = 0.80

# Isolation Forest
N_ESTIMATORS = 300
CONTAMINATION = 0.01
RANDOM_STATE = 42


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def create_features(series_df: pd.DataFrame) -> pd.DataFrame:
    """
    Crée des features temporelles robustes pour UNE série.
    """

    df = series_df.copy()

    df = df.sort_values("timestamp").reset_index(drop=True)

    value = df["value"].astype(float)

    # --------------------------------------------------------
    # Valeur brute
    # --------------------------------------------------------

    df["feature_value"] = value

    # --------------------------------------------------------
    # Différences
    # --------------------------------------------------------

    df["diff_1"] = value.diff()

    df["diff_2"] = value.diff().diff()

    # --------------------------------------------------------
    # Variation relative
    # --------------------------------------------------------

    previous = value.shift(1)

    denominator = previous.abs().replace(0, np.nan)

    df["relative_change"] = (
        (value - previous) / denominator
    )

    # --------------------------------------------------------
    # Rolling median
    # --------------------------------------------------------

    rolling_median_5 = (
        value
        .rolling(window=5, min_periods=3)
        .median()
    )

    rolling_median_15 = (
        value
        .rolling(window=15, min_periods=5)
        .median()
    )

    rolling_median_30 = (
        value
        .rolling(window=30, min_periods=10)
        .median()
    )

    df["rolling_median_5"] = rolling_median_5
    df["rolling_median_15"] = rolling_median_15
    df["rolling_median_30"] = rolling_median_30

    # --------------------------------------------------------
    # Rolling standard deviation
    # --------------------------------------------------------

    df["rolling_std_5"] = (
        value
        .rolling(window=5, min_periods=3)
        .std()
    )

    df["rolling_std_15"] = (
        value
        .rolling(window=15, min_periods=5)
        .std()
    )

    df["rolling_std_30"] = (
        value
        .rolling(window=30, min_periods=10)
        .std()
    )

    # --------------------------------------------------------
    # Déviation par rapport à la médiane
    # --------------------------------------------------------

    df["median_deviation_5"] = (
        value - rolling_median_5
    )

    df["median_deviation_15"] = (
        value - rolling_median_15
    )

    df["median_deviation_30"] = (
        value - rolling_median_30
    )

    # --------------------------------------------------------
    # MAD robuste
    # --------------------------------------------------------

    rolling_mad_15 = (
        value
        .rolling(window=15, min_periods=5)
        .apply(
            lambda x: np.median(
                np.abs(x - np.median(x))
            ),
            raw=True,
        )
    )

    df["rolling_mad_15"] = rolling_mad_15

    # --------------------------------------------------------
    # Robust Z-score
    # --------------------------------------------------------

    df["robust_zscore"] = (
        (value - rolling_median_15)
        / (
            1.4826 * rolling_mad_15
            + 1e-9
        )
    )

    # --------------------------------------------------------
    # Tendance locale
    # --------------------------------------------------------

    df["trend_5"] = (
        value - value.shift(5)
    )

    df["trend_15"] = (
        value - value.shift(15)
    )

    # --------------------------------------------------------
    # Magnitude du changement
    # --------------------------------------------------------

    df["abs_diff_1"] = df["diff_1"].abs()

    df["abs_relative_change"] = (
        df["relative_change"].abs()
    )

    return df


# ============================================================
# LOAD DATA
# ============================================================

def load_series():

    all_series = []

    print("=" * 70)
    print("HARDTEC AioPS - GAIA METRIC ANOMALY DETECTION V2")
    print("=" * 70)

    for family in FAMILIES:

        family_path = METRIC_ROOT / family

        if not family_path.exists():
            print(
                f"\n[WARNING] Famille absente : {family}"
            )
            continue

        csv_files = sorted(
            family_path.glob("*.csv")
        )

        print(
            f"\nFamille : {family}"
        )
        print(
            f"Fichiers : {len(csv_files)}"
        )

        for csv_file in csv_files:

            try:

                df = pd.read_csv(csv_file)

                required_columns = {
                    "timestamp",
                    "value",
                    "label",
                }

                if not required_columns.issubset(
                    df.columns
                ):
                    print(
                        f"[SKIP] Colonnes manquantes : "
                        f"{csv_file.name}"
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

                # Identifiant unique de série
                df["series_id"] = (
                    family
                    + "__"
                    + csv_file.stem
                )

                df["family"] = family

                all_series.append(df)

            except Exception as e:

                print(
                    f"[ERROR] {csv_file.name} : {e}"
                )

    if not all_series:
        raise RuntimeError(
            "Aucune série valide trouvée."
        )

    combined = pd.concat(
        all_series,
        ignore_index=True,
    )

    print("\n" + "-" * 70)
    print("CHARGEMENT TERMINE")
    print("-" * 70)

    print(
        f"Series chargees : "
        f"{combined['series_id'].nunique()}"
    )

    print(
        f"Points charges : "
        f"{len(combined):,}"
    )

    print(
        f"Anomalies ground truth : "
        f"{int(combined['label'].sum()):,}"
    )

    print(
        f"Taux anomalie : "
        f"{combined['label'].mean():.4%}"
    )

    return combined


# ============================================================
# BUILD FEATURES
# ============================================================

def build_all_features(df):

    print("\n" + "=" * 70)
    print("FEATURE ENGINEERING")
    print("=" * 70)

    feature_frames = []

    series_count = df["series_id"].nunique()

    for index, (series_id, group) in enumerate(
        df.groupby("series_id"),
        start=1,
    ):

        features = create_features(group)

        feature_frames.append(features)

        if index % 25 == 0:
            print(
                f"Séries traitées : "
                f"{index}/{series_count}"
            )

    result = pd.concat(
        feature_frames,
        ignore_index=True,
    )

    result = result.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    print(
        f"\nFeatures créées : "
        f"{len(FEATURE_COLUMNS)}"
    )

    print(
        f"Points disponibles : "
        f"{len(result):,}"
    )

    return result


# ============================================================
# FEATURE LIST
# ============================================================

FEATURE_COLUMNS = [
    "feature_value",
    "diff_1",
    "diff_2",
    "relative_change",
    "rolling_median_5",
    "rolling_median_15",
    "rolling_median_30",
    "rolling_std_5",
    "rolling_std_15",
    "rolling_std_30",
    "median_deviation_5",
    "median_deviation_15",
    "median_deviation_30",
    "rolling_mad_15",
    "robust_zscore",
    "trend_5",
    "trend_15",
    "abs_diff_1",
    "abs_relative_change",
]


# ============================================================
# ROBUST NORMALIZATION PER SERIES
# ============================================================

def robust_normalize(
    train_df,
    test_df,
):

    print("\n" + "=" * 70)
    print("NORMALISATION ROBUSTE PAR SERIE")
    print("=" * 70)

    train_out = train_df.copy()
    test_out = test_df.copy()

    parameters = {}

    for series_id in train_df["series_id"].unique():

        train_mask = (
            train_df["series_id"]
            == series_id
        )

        test_mask = (
            test_df["series_id"]
            == series_id
        )

        parameters[series_id] = {}

        for feature in FEATURE_COLUMNS:

            values = pd.to_numeric(
                train_df.loc[
                    train_mask,
                    feature,
                ],
                errors="coerce",
            )

            median = values.median()

            q1 = values.quantile(0.25)
            q3 = values.quantile(0.75)

            iqr = q3 - q1

            if (
                pd.isna(iqr)
                or iqr == 0
            ):
                iqr = 1.0

            if pd.isna(median):
                median = 0.0

            parameters[series_id][
                feature
            ] = {
                "median": median,
                "iqr": iqr,
            }

            train_values = pd.to_numeric(
                train_df.loc[
                    train_mask,
                    feature,
                ],
                errors="coerce",
            )

            train_out.loc[
                train_mask,
                feature,
            ] = (
                train_values - median
            ) / iqr

            if test_mask.any():

                test_values = pd.to_numeric(
                    test_df.loc[
                        test_mask,
                        feature,
                    ],
                    errors="coerce",
                )

                test_out.loc[
                    test_mask,
                    feature,
                ] = (
                    test_values - median
                ) / iqr

    print(
        f"Features normalisées : "
        f"{len(FEATURE_COLUMNS)}"
    )

    return (
        train_out,
        test_out,
        parameters,
    )


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

def temporal_split(df):

    print("\n" + "=" * 70)
    print("SPLIT TEMPOREL")
    print("=" * 70)

    train_parts = []
    test_parts = []

    for series_id, group in df.groupby(
        "series_id"
    ):

        group = group.sort_values(
            "timestamp"
        ).reset_index(drop=True)

        split_index = int(
            len(group) * TRAIN_RATIO
        )

        if split_index < 10:
            continue

        train_part = group.iloc[
            :split_index
        ].copy()

        test_part = group.iloc[
            split_index:
        ].copy()

        train_parts.append(train_part)
        test_parts.append(test_part)

    train = pd.concat(
        train_parts,
        ignore_index=True,
    )

    test = pd.concat(
        test_parts,
        ignore_index=True,
    )

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

    return train, test


# ============================================================
# TRAIN ISOLATION FOREST
# ============================================================

def train_model(
    train_df,
    test_df,
):

    print("\n" + "=" * 70)
    print("ISOLATION FOREST V2")
    print("=" * 70)

    X_train = train_df[
        FEATURE_COLUMNS
    ].copy()

    X_test = test_df[
        FEATURE_COLUMNS
    ].copy()

    # Médiane du TRAIN uniquement
    train_medians = X_train.median()

    X_train = X_train.fillna(
        train_medians
    )

    X_test = X_test.fillna(
        train_medians
    )

    model = IsolationForest(
        n_estimators=N_ESTIMATORS,
        contamination=CONTAMINATION,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    print(
        f"Train observations : "
        f"{len(X_train):,}"
    )

    print(
        f"Test observations : "
        f"{len(X_test):,}"
    )

    print(
        f"Features : "
        f"{len(FEATURE_COLUMNS)}"
    )

    print(
        f"Contamination : "
        f"{CONTAMINATION}"
    )

    print("\nEntraînement...")

    model.fit(X_train)

    print("Entraînement terminé.")

    # --------------------------------------------------------
    # Scores
    # --------------------------------------------------------

    train_raw_score = (
        model.decision_function(
            X_train
        )
    )

    test_raw_score = (
        model.decision_function(
            X_test
        )
    )

    # Isolation Forest :
    # score faible = plus anormal
    # inversion pour avoir :
    # score élevé = plus anormal

    train_anomaly_score = (
        -train_raw_score
    )

    test_anomaly_score = (
        -test_raw_score
    )

    # Normalisation basée uniquement
    # sur le TRAIN
    score_min = np.min(
        train_anomaly_score
    )

    score_max = np.max(
        train_anomaly_score
    )

    denominator = (
        score_max - score_min
    )

    if denominator == 0:
        denominator = 1.0

    train_score_normalized = (
        train_anomaly_score
        - score_min
    ) / denominator

    test_score_normalized = (
        test_anomaly_score
        - score_min
    ) / denominator

    # --------------------------------------------------------
    # Predictions
    # --------------------------------------------------------

    train_prediction = (
        model.predict(X_train)
        == -1
    )

    test_prediction = (
        model.predict(X_test)
        == -1
    )

    train_result = train_df.copy()
    test_result = test_df.copy()

    train_result[
        "anomaly_score"
    ] = train_score_normalized

    test_result[
        "anomaly_score"
    ] = test_score_normalized

    train_result[
        "predicted_anomaly"
    ] = train_prediction.astype(int)

    test_result[
        "predicted_anomaly"
    ] = test_prediction.astype(int)

    return (
        model,
        train_result,
        test_result,
        train_medians,
        {
            "score_min": score_min,
            "score_max": score_max,
        },
    )


# ============================================================
# EVALUATION
# ============================================================

def evaluate(
    result_df,
    dataset_name,
):

    y_true = result_df[
        "label"
    ].astype(int)

    y_pred = result_df[
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

    print("\n" + "-" * 70)

    print(
        f"EVALUATION : {dataset_name}"
    )

    print("-" * 70)

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

    print("\nClassification Report")

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

    return {
        "dataset": dataset_name,
        "points": len(result_df),
        "ground_truth_anomalies": int(
            y_true.sum()
        ),
        "predicted_anomalies": int(
            y_pred.sum()
        ),
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


# ============================================================
# EVALUATION BY FAMILY
# ============================================================

def evaluate_by_family(
    test_result,
):

    print("\n" + "=" * 70)
    print("EVALUATION PAR FAMILLE")
    print("=" * 70)

    rows = []

    for family, group in test_result.groupby(
        "family"
    ):

        y_true = group[
            "label"
        ].astype(int)

        y_pred = group[
            "predicted_anomaly"
        ].astype(int)

        rows.append(
            {
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

    evaluation = pd.DataFrame(
        rows
    )

    print(
        evaluation.to_string(
            index=False
        )
    )

    return evaluation


# ============================================================
# SAVE RESULTS
# ============================================================

def save_results(
    model,
    train_result,
    test_result,
    evaluation,
    evaluation_by_family,
    normalization_parameters,
    train_medians,
    score_parameters,
):

    print("\n" + "=" * 70)
    print("SAUVEGARDE")
    print("=" * 70)

    # --------------------------------------------------------
    # Combine predictions
    # --------------------------------------------------------

    train_result[
        "split"
    ] = "train"

    test_result[
        "split"
    ] = "test"

    all_predictions = pd.concat(
        [
            train_result,
            test_result,
        ],
        ignore_index=True,
    )

    prediction_path = (
        OUTPUT_DIR
        / "gaia_metric_anomaly_predictions_v2.csv"
    )

    all_predictions.to_csv(
        prediction_path,
        index=False,
    )

    # --------------------------------------------------------
    # Evaluation
    # --------------------------------------------------------

    evaluation_path = (
        OUTPUT_DIR
        / "gaia_metric_evaluation_v2.csv"
    )

    pd.DataFrame(
        [evaluation]
    ).to_csv(
        evaluation_path,
        index=False,
    )

    # --------------------------------------------------------
    # Evaluation by family
    # --------------------------------------------------------

    family_path = (
        OUTPUT_DIR
        / "gaia_metric_evaluation_by_family_v2.csv"
    )

    evaluation_by_family.to_csv(
        family_path,
        index=False,
    )

    # --------------------------------------------------------
    # Model package
    # --------------------------------------------------------

    model_package = {
        "model": model,
        "features": FEATURE_COLUMNS,
        "train_medians": train_medians,
        "normalization_parameters": normalization_parameters,
        "score_parameters": score_parameters,
        "train_ratio": TRAIN_RATIO,
        "contamination": CONTAMINATION,
    }

    model_path = (
        MODEL_DIR
        / "gaia_metric_isolation_forest_v2.pkl"
    )

    joblib.dump(
        model_package,
        model_path,
    )

    print(
        f"\nPredictions : "
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
        f"Modele : "
        f"{model_path}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # 1. Load
    # --------------------------------------------------------

    data = load_series()

    # --------------------------------------------------------
    # 2. Feature engineering
    # --------------------------------------------------------

    data = build_all_features(
        data
    )

    # --------------------------------------------------------
    # 3. Temporal split
    # --------------------------------------------------------

    train_df, test_df = temporal_split(
        data
    )

    # --------------------------------------------------------
    # 4. Normalization
    # --------------------------------------------------------

    (
        train_normalized,
        test_normalized,
        normalization_parameters,
    ) = robust_normalize(
        train_df,
        test_df,
    )

    # --------------------------------------------------------
    # 5. Train
    # --------------------------------------------------------

    (
        model,
        train_result,
        test_result,
        train_medians,
        score_parameters,
    ) = train_model(
        train_normalized,
        test_normalized,
    )

    # --------------------------------------------------------
    # 6. Evaluation
    # --------------------------------------------------------

    train_evaluation = evaluate(
        train_result,
        "TRAIN",
    )

    test_evaluation = evaluate(
        test_result,
        "TEST",
    )

    # --------------------------------------------------------
    # 7. Family evaluation
    # --------------------------------------------------------

    evaluation_by_family = (
        evaluate_by_family(
            test_result
        )
    )

    # --------------------------------------------------------
    # 8. Save
    # --------------------------------------------------------

    save_results(
        model,
        train_result,
        test_result,
        test_evaluation,
        evaluation_by_family,
        normalization_parameters,
        train_medians,
        score_parameters,
    )

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("METRICS V2 TERMINE")
    print("=" * 70)

    print(
        "\nRésultat TEST :"
    )

    print(
        f"Precision : "
        f"{test_evaluation['precision']:.4f}"
    )

    print(
        f"Recall    : "
        f"{test_evaluation['recall']:.4f}"
    )

    print(
        f"F1-score  : "
        f"{test_evaluation['f1']:.4f}"
    )

    print(
        "\nIMPORTANT : "
        "le label GAIA n'a pas été utilisé "
        "comme feature."
    )


if __name__ == "__main__":
    main()