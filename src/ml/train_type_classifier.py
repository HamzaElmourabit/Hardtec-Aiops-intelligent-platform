import pandas as pd
import joblib
import mlflow

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

try:
    from src.ml.tracking import (
        end_training_run,
        log_classification_result,
        start_training_run,
    )
except ModuleNotFoundError:
    from tracking import end_training_run, log_classification_result, start_training_run


# ==========================================
# 1. Configuration
# ==========================================

DATA_PATH = "data/processed/tickets_clean.csv"
MODEL_PATH = "models/ticket_type_model.pkl"


# ==========================================
# 2. Chargement des données
# ==========================================

print("Chargement du dataset...")

start_training_run("ticket_type_classifier", DATA_PATH)

mlflow.log_params(
    {
        "test_size": 0.20,
        "random_state": 42,
        "tfidf_max_features": 30000,
        "tfidf_ngram_range": "(1, 2)",
        "classifier": "LinearSVC",
        "classifier_C": 1.0,
    }
)

df = pd.read_csv(DATA_PATH)

print(f"Nombre de tickets : {len(df)}")
print(f"Colonnes : {df.columns.tolist()}")


# ==========================================
# 3. Nettoyage minimal
# ==========================================

df = df.dropna(subset=["ticket_text", "type"])

df["ticket_text"] = df["ticket_text"].astype(str)
df["type"] = df["type"].astype(str).str.strip()


print("\nDistribution des classes :")
print(df["type"].value_counts())


# ==========================================
# 4. Features / Target
# ==========================================

X = df["ticket_text"]
y = df["type"]


# ==========================================
# 5. Train / Test Split
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTrain :", len(X_train))
print("Test  :", len(X_test))


# ==========================================
# 6. Pipeline NLP + Machine Learning
# ==========================================

model = Pipeline([
    (
        "tfidf",
        TfidfVectorizer(
            lowercase=True,
            max_features=30000,
            ngram_range=(1, 2),
            min_df=2,
            sublinear_tf=True
        )
    ),
    (
        "classifier",
        LinearSVC(
            C=1.0
        )
    )
])


# ==========================================
# 7. Entraînement
# ==========================================

print("\nEntraînement du modèle...")

model.fit(X_train, y_train)

print("✅ Entraînement terminé.")


# ==========================================
# 8. Prédiction
# ==========================================

y_pred = model.predict(X_test)


# ==========================================
# 9. Évaluation
# ==========================================

accuracy = accuracy_score(y_test, y_pred)

print("\n================================")
print("RÉSULTATS DU MODÈLE")
print("================================")

print(f"\nAccuracy : {accuracy:.4f}")

print("\nClassification Report :")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix :")
print(confusion_matrix(y_test, y_pred))


# ==========================================
# 10. Sauvegarde
# ==========================================

print("\nSauvegarde du modèle...")

joblib.dump(model, MODEL_PATH)

log_classification_result(
    y_test,
    y_pred,
    model_path=MODEL_PATH,
)
end_training_run()

print(f"✅ Modèle sauvegardé : {MODEL_PATH}")