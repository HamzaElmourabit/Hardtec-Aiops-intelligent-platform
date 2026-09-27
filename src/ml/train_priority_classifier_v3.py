import os
import pandas as pd
import joblib
import mlflow

try:
    from src.ml.tracking import (
        end_training_run,
        log_classification_result,
        start_training_run,
    )
except ModuleNotFoundError:
    from tracking import end_training_run, log_classification_result, start_training_run

from scipy.sparse import hstack
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score
)


# ============================================================
# CONFIGURATION
# ============================================================

DATA_PATH = "data/processed/tickets_clean.csv"
MODEL_PATH = "models/ticket_priority_model_v3.pkl"

RANDOM_STATE = 42


# ============================================================
# CHARGEMENT
# ============================================================

print("=" * 60)
print("PRIORITY CLASSIFIER V3")
print("=" * 60)

print("\nChargement du dataset...")

start_training_run("ticket_priority_classifier_v3", DATA_PATH)

mlflow.log_params(
    {
        "test_size": 0.20,
        "random_state": RANDOM_STATE,
        "tfidf_max_features": 30000,
        "tfidf_ngram_range": "(1, 2)",
        "classifier": "LinearSVC",
        "classifier_C": 1.0,
        "class_weight": "balanced",
    }
)

df = pd.read_csv(DATA_PATH)

print(f"Nombre total de lignes : {len(df)}")


# ============================================================
# VERIFICATION
# ============================================================

required_columns = [
    "ticket_text",
    "type",
    "queue",
    "priority"
]

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Colonnes manquantes : {missing_columns}"
    )


# ============================================================
# NETTOYAGE
# ============================================================

df = df.dropna(
    subset=required_columns
)

df["ticket_text"] = (
    df["ticket_text"]
    .astype(str)
    .str.strip()
)

df["type"] = (
    df["type"]
    .astype(str)
    .str.strip()
)

df["queue"] = (
    df["queue"]
    .astype(str)
    .str.strip()
)

df["priority"] = (
    df["priority"]
    .astype(str)
    .str.lower()
    .str.strip()
)


df = df[
    df["ticket_text"].str.len() > 0
]


print(
    f"\nNombre de tickets après nettoyage : "
    f"{len(df)}"
)


# ============================================================
# DISTRIBUTION
# ============================================================

print("\n" + "=" * 60)
print("DISTRIBUTION DES PRIORITES")
print("=" * 60)

print(
    df["priority"].value_counts()
)


# ============================================================
# FEATURES
# ============================================================

X_text = df["ticket_text"]

# Le modèle apprend :
#
# ticket_text + type + queue
#            ↓
#         priority
#
X_meta = df[
    ["type", "queue"]
]

y = df["priority"]


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

(
    X_text_train,
    X_text_test,
    X_meta_train,
    X_meta_test,
    y_train,
    y_test
) = train_test_split(
    X_text,
    X_meta,
    y,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=y
)


print("\n" + "=" * 60)
print("TRAIN / TEST")
print("=" * 60)

print(
    f"Train : {len(X_text_train)}"
)

print(
    f"Test  : {len(X_text_test)}"
)


# ============================================================
# TF-IDF
# ============================================================

print("\n" + "=" * 60)
print("VECTORISATION TF-IDF")
print("=" * 60)

tfidf = TfidfVectorizer(
    lowercase=True,
    max_features=30000,
    ngram_range=(1, 2),
    min_df=2,
    sublinear_tf=True
)


X_train_text = tfidf.fit_transform(
    X_text_train
)

X_test_text = tfidf.transform(
    X_text_test
)


print(
    f"Nombre de features texte : "
    f"{X_train_text.shape[1]}"
)


# ============================================================
# ENCODAGE TYPE + QUEUE
# ============================================================

print("\nEncodage des métadonnées...")

encoder = OneHotEncoder(
    handle_unknown="ignore"
)


X_train_meta = encoder.fit_transform(
    X_meta_train
)

X_test_meta = encoder.transform(
    X_meta_test
)


# ============================================================
# COMBINAISON
# ============================================================

X_train = hstack([
    X_train_text,
    X_train_meta
]).tocsr()


X_test = hstack([
    X_test_text,
    X_test_meta
]).tocsr()


print("\nDimensions finales :")

print(
    f"Train : {X_train.shape}"
)

print(
    f"Test  : {X_test.shape}"
)


# ============================================================
# MODELE
# ============================================================

print("\n" + "=" * 60)
print("ENTRAINEMENT DU MODELE")
print("=" * 60)

classifier = LinearSVC(
    C=1.0,
    class_weight="balanced"
)


classifier.fit(
    X_train,
    y_train
)


print("\n✅ Entraînement terminé.")


# ============================================================
# PREDICTION
# ============================================================

y_pred = classifier.predict(
    X_test
)


# ============================================================
# EVALUATION
# ============================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

macro_f1 = f1_score(
    y_test,
    y_pred,
    average="macro"
)

weighted_f1 = f1_score(
    y_test,
    y_pred,
    average="weighted"
)


print("\n" + "=" * 60)
print("RESULTATS PRIORITY CLASSIFIER V3")
print("=" * 60)

print(
    f"\nAccuracy   : {accuracy:.4f}"
)

print(
    f"Macro F1   : {macro_f1:.4f}"
)

print(
    f"Weighted F1: {weighted_f1:.4f}"
)


print("\nClassification Report :")

print(
    classification_report(
        y_test,
        y_pred,
        zero_division=0
    )
)


print("\nConfusion Matrix :")

print(
    confusion_matrix(
        y_test,
        y_pred
    )
)


# ============================================================
# SAUVEGARDE
# ============================================================

model_data = {
    "tfidf": tfidf,
    "encoder": encoder,
    "classifier": classifier
}


os.makedirs(
    os.path.dirname(MODEL_PATH),
    exist_ok=True
)


joblib.dump(
    model_data,
    MODEL_PATH
)

log_classification_result(
    y_test,
    y_pred,
    model_path=MODEL_PATH,
)
end_training_run()


print("\n" + "=" * 60)
print("SAUVEGARDE")
print("=" * 60)

print(
    f"\n✅ Modèle sauvegardé : {MODEL_PATH}"
)

print("\nTerminé.")