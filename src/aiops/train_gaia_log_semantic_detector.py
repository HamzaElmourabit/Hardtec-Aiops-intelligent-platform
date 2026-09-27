# ============================================================
# GAIA - LOG SEMANTIC ANOMALY DETECTOR
# ============================================================
# Objectif :
# Détecter les logs ERROR vs INFO à partir du contenu du log
# sans utiliser "level" ni "exception_type" comme features.
#
# Pipeline :
#   error.csv + info.csv
#          ↓
#      nettoyage
#          ↓
#   texte + source + host
#          ↓
#        TF-IDF
#          ↓
#      LinearSVC
#          ↓
#    INFO / ERROR
#
# IMPORTANT :
# "level" et "exception_type" sont exclus des features
# pour éviter la fuite de données.
# ============================================================

from pathlib import Path
import pickle

import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC


# ============================================================
# CONFIGURATION
# ============================================================

ERROR_PATH = Path("./log/log semantics anomaly detection/error.csv")
INFO_PATH = Path("./log/log semantics anomaly detection/info.csv")

MODEL_DIR = Path("./models")
OUTPUT_DIR = Path("./data/processed")

MODEL_PATH = MODEL_DIR / "gaia_log_semantic_detector.pkl"
PREDICTIONS_PATH = OUTPUT_DIR / "gaia_log_semantic_predictions.csv"

RANDOM_STATE = 42


# ============================================================
# 1. CHARGEMENT
# ============================================================

print("=" * 70)
print("GAIA - LOG SEMANTIC ANOMALY DETECTOR")
print("=" * 70)

print("\n[1/7] Chargement des données...")

if not ERROR_PATH.exists():
    raise FileNotFoundError(f"Fichier introuvable : {ERROR_PATH}")

if not INFO_PATH.exists():
    raise FileNotFoundError(f"Fichier introuvable : {INFO_PATH}")

error_df = pd.read_csv(ERROR_PATH)
info_df = pd.read_csv(INFO_PATH)

error_df["label"] = "ERROR"
info_df["label"] = "INFO"

df = pd.concat(
    [error_df, info_df],
    ignore_index=True,
)

print(f"ERROR logs : {len(error_df):,}")
print(f"INFO logs  : {len(info_df):,}")
print(f"TOTAL      : {len(df):,}")


# ============================================================
# 2. PREPARATION DU TEXTE
# ============================================================

print("\n[2/7] Préparation des données...")

# Colonnes texte potentielles
TEXT_COLUMNS = [
    "description",
    "raw_data",
    "source",
    "host",
]


def combine_text(row):
    """
    Construit une représentation textuelle du log.

    IMPORTANT :
    level et exception_type ne sont PAS utilisés.
    """

    parts = []

    for column in TEXT_COLUMNS:
        if column in row.index:
            value = row[column]

            if pd.notna(value):
                value = str(value).strip()

                if value:
                    parts.append(value)

    return " ".join(parts)


df["text"] = df.apply(combine_text, axis=1)

# Supprimer les logs sans contenu exploitable
df["text"] = df["text"].fillna("").astype(str)

before = len(df)

df = df[df["text"].str.strip() != ""].copy()

removed = before - len(df)

print(f"Logs supprimés sans texte : {removed}")
print(f"Logs utilisables           : {len(df):,}")

print("\nDistribution des labels :")
print(df["label"].value_counts())


# ============================================================
# 3. TRAIN / TEST SPLIT
# ============================================================

print("\n[3/7] Séparation Train/Test...")

X = df["text"]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=y,
)

print(f"Train : {len(X_train):,}")
print(f"Test  : {len(X_test):,}")


# ============================================================
# 4. MODELE TF-IDF + LINEARSVC
# ============================================================

print("\n[4/7] Entraînement TF-IDF + LinearSVC...")

model = Pipeline(
    [
        (
            "tfidf",
            TfidfVectorizer(
                lowercase=True,
                strip_accents="unicode",
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.98,
                sublinear_tf=True,
                max_features=100_000,
            ),
        ),
        (
            "classifier",
            LinearSVC(
                C=1.0,
                class_weight="balanced",
                random_state=RANDOM_STATE,
            ),
        ),
    ]
)

model.fit(X_train, y_train)

print("Entraînement terminé.")


# ============================================================
# 5. EVALUATION
# ============================================================

print("\n[5/7] Évaluation du modèle...")

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

f1_macro = f1_score(
    y_test,
    y_pred,
    average="macro",
)

f1_weighted = f1_score(
    y_test,
    y_pred,
    average="weighted",
)

print("\n" + "=" * 70)
print("RESULTATS")
print("=" * 70)

print(f"\nAccuracy     : {accuracy:.4f}")
print(f"F1 Macro     : {f1_macro:.4f}")
print(f"F1 Weighted  : {f1_weighted:.4f}")

print("\nClassification Report :")
print(
    classification_report(
        y_test,
        y_pred,
        digits=4,
    )
)

print("\nMatrice de confusion :")

labels = ["INFO", "ERROR"]

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=labels,
)

cm_df = pd.DataFrame(
    cm,
    index=[f"Actual_{x}" for x in labels],
    columns=[f"Predicted_{x}" for x in labels],
)

print(cm_df)


# ============================================================
# 6. PREDICTIONS SUR TOUT LE DATASET
# ============================================================

print("\n[6/7] Génération des prédictions...")

df["predicted_label"] = model.predict(df["text"])

# Score de décision
decision_scores = model.decision_function(df["text"])

df["decision_score"] = decision_scores

# Anomalie = prédiction ERROR
df["is_semantic_anomaly"] = (
    df["predicted_label"] == "ERROR"
).astype(int)

# Confiance approximative basée sur la distance à l'hyperplan
df["anomaly_confidence"] = (
    abs(df["decision_score"])
)

# Normalisation 0-1
max_score = df["anomaly_confidence"].max()

if max_score > 0:
    df["anomaly_confidence"] = (
        df["anomaly_confidence"] / max_score
    )
else:
    df["anomaly_confidence"] = 0.0


# ============================================================
# 7. SAUVEGARDE
# ============================================================

print("\n[7/7] Sauvegarde...")

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

# Sauvegarder le modèle
with open(MODEL_PATH, "wb") as f:
    pickle.dump(model, f)

# Colonnes utiles pour la suite AIOps
prediction_columns = [
    "file_id",
    "line_id",
    "timestamp",
    "host",
    "source",
    "level",
    "description",
    "raw_data",
    "label",
    "predicted_label",
    "decision_score",
    "is_semantic_anomaly",
    "anomaly_confidence",
]

prediction_columns = [
    col
    for col in prediction_columns
    if col in df.columns
]

df[prediction_columns].to_csv(
    PREDICTIONS_PATH,
    index=False,
)

print("\n" + "=" * 70)
print("FICHIERS CREES")
print("=" * 70)

print(f"\nModèle :")
print(MODEL_PATH)

print("\nPrédictions :")
print(PREDICTIONS_PATH)

print("\nDistribution des prédictions :")
print(
    df["predicted_label"].value_counts()
)

print("\nLogs détectés comme anomalies sémantiques :")
print(
    df["is_semantic_anomaly"].value_counts()
)

print("\nTerminé.")