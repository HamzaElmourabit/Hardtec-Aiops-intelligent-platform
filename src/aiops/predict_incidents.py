
"""
HARDTEC - AIOps Temporal Incident Prediction
--------------------------------------------

Objectif :
    Prédire le risque qu'une anomalie apparaisse dans la prochaine
    fenêtre temporelle à partir de l'historique des fenêtres précédentes.

Approche :
    t-3, t-2, t-1, t
              ↓
        modèle temporel
              ↓
        risque à t+1

IMPORTANT :
    - Les informations de t+1 ne sont JAMAIS utilisées comme features.
    - La cible est basée sur l'anomalie détectée par Isolation Forest
      dans la fenêtre future.
    - Ce n'est donc PAS une prédiction d'incident réel labellisé par HARDTEC.
    - Il s'agit d'un Proof of Concept de "future anomaly / incident risk prediction".
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = PROJECT_ROOT / "data" / "processed" / "aiops_anomalies.csv"

OUTPUT_FILE = PROJECT_ROOT / "data" / "processed" / "aiops_incident_predictions.csv"

MODEL_DIR = PROJECT_ROOT / "models"

MODEL_FILE = MODEL_DIR / "aiops_incident_predictor.pkl"
SCALER_FILE = MODEL_DIR / "aiops_incident_predictor_scaler.pkl"


# Nombre de fenêtres historiques utilisées
LAGS = 3


# ============================================================
# FEATURES TEMPORELLES
# ============================================================

BASE_FEATURES = [
    "event_count",
    "raw_row_count",
    "avg_duration_ms",
    "max_duration_ms",
    "p95_duration_ms",
    "high_severity_count",
    "high_severity_rate",
    "error_density",
    "unique_paths",
    "unique_event_types",
    "unique_categories",
    "unique_severities",
]


# ============================================================
# CHARGEMENT
# ============================================================


def load_data():
    """Charge les fenêtres AIOps avec les anomalies détectées."""

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Fichier introuvable : {INPUT_FILE}\n"
            "Exécute d'abord detect_anomalies.py."
        )

    df = pd.read_csv(INPUT_FILE)

    if "window_start" not in df.columns:
        raise ValueError("La colonne 'window_start' est absente.")

    if "is_anomaly" not in df.columns:
        raise ValueError(
            "La colonne 'is_anomaly' est absente. "
            "Exécute d'abord detect_anomalies.py."
        )

    df["window_start"] = pd.to_datetime(
        df["window_start"],
        utc=True,
    )

    df = df.sort_values("window_start").reset_index(drop=True)

    return df


# ============================================================
# CREATION DES FEATURES TEMPORELLES
# ============================================================


def create_temporal_features(df):
    """
    Crée les variables historiques :

        t-1
        t-2
        t-3

    puis définit la cible :

        future_anomaly = anomalie à t+1

    Aucune information de t+1 n'est utilisée dans X.
    """

    data = df.copy()

    # --------------------------------------------------------
    # TARGET FUTURE
    # --------------------------------------------------------

    # La cible correspond à l'anomalie de la fenêtre suivante.
    data["future_anomaly"] = (
        data["is_anomaly"]
        .shift(-1)
        .astype("float")
    )

    # --------------------------------------------------------
    # HISTORICAL FEATURES
    # --------------------------------------------------------

    temporal_columns = []

    for lag in range(1, LAGS + 1):

        for feature in BASE_FEATURES:

            column_name = f"{feature}_lag_{lag}"

            data[column_name] = data[feature].shift(lag)

            temporal_columns.append(column_name)

    # --------------------------------------------------------
    # HISTORICAL ANOMALY SIGNAL
    # --------------------------------------------------------

    for lag in range(1, LAGS + 1):

        column_name = f"is_anomaly_lag_{lag}"

        data[column_name] = data["is_anomaly"].shift(lag)

        temporal_columns.append(column_name)

    return data, temporal_columns


# ============================================================
# PREPARATION DATASET
# ============================================================


def prepare_dataset(df):

    data, temporal_features = create_temporal_features(df)

    required_columns = temporal_features + ["future_anomaly"]

    data = data.dropna(
        subset=required_columns
    ).reset_index(drop=True)

    X = data[temporal_features].copy()

    y = data["future_anomaly"].astype(int)

    return data, X, y, temporal_features


# ============================================================
# TEMPORAL TRAIN / TEST SPLIT
# ============================================================


def temporal_split(X, y, data):

    """
    Split temporel :

        anciennes observations → TRAIN
        observations récentes → TEST

    Pas de random split afin d'éviter la fuite temporelle.
    """

    n = len(X)

    if n < 10:
        raise ValueError(
            f"Pas assez d'observations après création des lags : {n}"
        )

    train_size = int(n * 0.70)

    X_train = X.iloc[:train_size]
    X_test = X.iloc[train_size:]

    y_train = y.iloc[:train_size]
    y_test = y.iloc[train_size:]

    data_train = data.iloc[:train_size]
    data_test = data.iloc[train_size:]

    return (
        X_train,
        X_test,
        y_train,
        y_test,
        data_train,
        data_test,
    )


# ============================================================
# ENTRAINEMENT
# ============================================================


def train_model(X_train, y_train):

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)

    model = LogisticRegression(
        class_weight="balanced",
        max_iter=2000,
        random_state=42,
    )

    model.fit(
        X_train_scaled,
        y_train,
    )

    return model, scaler


# ============================================================
# EVALUATION
# ============================================================


def evaluate_model(
    model,
    scaler,
    X_test,
    y_test,
):

    X_test_scaled = scaler.transform(X_test)

    predictions = model.predict(X_test_scaled)

    probabilities = model.predict_proba(
        X_test_scaled
    )[:, 1]

    print()
    print("=" * 70)
    print("TEMPORAL INCIDENT PREDICTION - EVALUATION")
    print("=" * 70)

    print(f"Test observations : {len(y_test)}")
    print(f"Positive cases    : {int(y_test.sum())}")
    print(f"Negative cases    : {int((y_test == 0).sum())}")

    print()
    print("Metrics")
    print("-" * 70)

    print(
        f"Accuracy  : {accuracy_score(y_test, predictions):.4f}"
    )

    print(
        f"Precision : {precision_score(y_test, predictions, zero_division=0):.4f}"
    )

    print(
        f"Recall    : {recall_score(y_test, predictions, zero_division=0):.4f}"
    )

    print(
        f"F1-score  : {f1_score(y_test, predictions, zero_division=0):.4f}"
    )

    # ROC-AUC uniquement si les deux classes sont présentes
    if len(np.unique(y_test)) == 2:

        auc = roc_auc_score(
            y_test,
            probabilities,
        )

        print(
            f"ROC-AUC   : {auc:.4f}"
        )

    print()
    print("Confusion Matrix")
    print("-" * 70)

    print(
        confusion_matrix(
            y_test,
            predictions,
        )
    )

    print()
    print("Classification Report")
    print("-" * 70)

    print(
        classification_report(
            y_test,
            predictions,
            zero_division=0,
        )
    )

    return predictions, probabilities


# ============================================================
# PREDICTION SUR TOUTES LES FENETRES
# ============================================================


def generate_predictions(
    data,
    model,
    scaler,
    temporal_features,
):

    X = data[temporal_features]

    X_scaled = scaler.transform(X)

    probabilities = model.predict_proba(
        X_scaled
    )[:, 1]

    predictions = (
        probabilities >= 0.50
    ).astype(int)

    result = data[
        [
            "window_start",
            "is_anomaly",
            "anomaly_level",
            "anomaly_score",
            "event_count",
            "avg_duration_ms",
            "p95_duration_ms",
            "failure_rate",
        ]
    ].copy()

    result["future_anomaly_actual"] = (
        data["future_anomaly"].astype(int)
    )

    result["incident_risk"] = probabilities

    result["incident_prediction"] = predictions

    # --------------------------------------------------------
    # RISK LEVEL
    # --------------------------------------------------------

    def risk_level(probability):

        if probability >= 0.75:
            return "CRITICAL"

        if probability >= 0.50:
            return "HIGH"

        if probability >= 0.25:
            return "MEDIUM"

        return "LOW"

    result["incident_risk_level"] = (
        result["incident_risk"]
        .apply(risk_level)
    )

    return result


# ============================================================
# SAVE
# ============================================================


def save_artifacts(
    model,
    scaler,
    predictions,
):

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        model,
        MODEL_FILE,
    )

    joblib.dump(
        scaler,
        SCALER_FILE,
    )

    predictions.to_csv(
        OUTPUT_FILE,
        index=False,
    )


# ============================================================
# MAIN
# ============================================================


def main():

    print()
    print("=" * 70)
    print("HARDTEC - AIOps Temporal Incident Prediction")
    print("=" * 70)

    # --------------------------------------------------------
    # LOAD
    # --------------------------------------------------------

    df = load_data()

    print()
    print(f"Input file       : {INPUT_FILE}")
    print(f"Total windows    : {len(df)}")
    print(
        f"Anomalous windows: {int(df['is_anomaly'].sum())}"
    )

    # --------------------------------------------------------
    # CREATE TEMPORAL DATASET
    # --------------------------------------------------------

    data, X, y, temporal_features = prepare_dataset(df)

    print()
    print("Temporal configuration")
    print("-" * 70)
    print(f"Lags used        : {LAGS}")
    print("Prediction target: anomaly at t+1")
    print(f"Usable samples   : {len(data)}")
    print(f"Positive targets : {int(y.sum())}")

    # --------------------------------------------------------
    # CHECK TARGET
    # --------------------------------------------------------

    if y.nunique() < 2:

        raise ValueError(
            "La cible ne contient qu'une seule classe. "
            "Un modèle supervisé ne peut pas être entraîné."
        )

    # --------------------------------------------------------
    # TEMPORAL SPLIT
    # --------------------------------------------------------

    (
        X_train,
        X_test,
        y_train,
        y_test,
        data_train,
        data_test,
    ) = temporal_split(
        X,
        y,
        data,
    )

    print()
    print("Temporal split")
    print("-" * 70)
    print(f"Train samples    : {len(X_train)}")
    print(f"Test samples     : {len(X_test)}")
    print(f"Train positives  : {int(y_train.sum())}")
    print(f"Test positives   : {int(y_test.sum())}")

    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    model, scaler = train_model(
        X_train,
        y_train,
    )

    # --------------------------------------------------------
    # EVALUATE
    # --------------------------------------------------------

    evaluate_model(
        model,
        scaler,
        X_test,
        y_test,
    )

    # --------------------------------------------------------
    # PREDICT ALL USABLE WINDOWS
    # --------------------------------------------------------

    predictions = generate_predictions(
        data,
        model,
        scaler,
        temporal_features,
    )

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    save_artifacts(
        model,
        scaler,
        predictions,
    )

    # --------------------------------------------------------
    # DISPLAY RESULTS
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("TOP INCIDENT-RISK WINDOWS")
    print("=" * 70)

    display_columns = [
        "window_start",
        "incident_risk",
        "incident_risk_level",
        "is_anomaly",
        "anomaly_level",
        "event_count",
        "failure_rate",
    ]

    print(
        predictions
        .sort_values(
            "incident_risk",
            ascending=False,
        )
        [display_columns]
        .head(10)
        .to_string(index=False)
    )

    print()
    print("=" * 70)
    print("FILES SAVED")
    print("=" * 70)

    print(f"Predictions : {OUTPUT_FILE}")
    print(f"Model       : {MODEL_FILE}")
    print(f"Scaler      : {SCALER_FILE}")

    print()
    print("IMPORTANT:")
    print(
        "This model predicts future anomaly risk, "
        "not ground-truth production incidents."
    )

    print(
        "Because the dataset contains only 41 temporal windows, "
        "results must be considered Proof of Concept."
    )

    print()


if __name__ == "__main__":
    main()

