import sys
import joblib
import pandas as pd
from scipy.sparse import hstack


MODEL_PATH = "models/ticket_priority_model_v2.pkl"


# ==========================================
# 1. Vérification des arguments
# ==========================================

if len(sys.argv) < 4:
    print("\nUtilisation :")
    print(
        'python src\\ml\\predict_priority.py '
        '"ticket text" "type" "queue"'
    )
    sys.exit(1)


ticket_text = sys.argv[1]
ticket_type = sys.argv[2]
ticket_queue = sys.argv[3]


# ==========================================
# 2. Chargement du modèle
# ==========================================

print("\nChargement du modèle...")

model_data = joblib.load(MODEL_PATH)

tfidf = model_data["tfidf"]
encoder = model_data["encoder"]
classifier = model_data["classifier"]

print("✅ Modèle chargé")


# ==========================================
# 3. Transformation du texte
# ==========================================

text_vector = tfidf.transform([ticket_text])


# ==========================================
# 4. Transformation des métadonnées
# ==========================================

metadata = pd.DataFrame([
    {
        "type": ticket_type,
        "queue": ticket_queue
    }
])

metadata_vector = encoder.transform(metadata)


# ==========================================
# 5. Fusion des features
# ==========================================

final_vector = hstack([
    text_vector,
    metadata_vector
]).tocsr()


# ==========================================
# 6. Prédiction
# ==========================================

prediction = classifier.predict(final_vector)[0]


# ==========================================
# 7. Résultat
# ==========================================

print("\n==============================")
print("AI PRIORITY CLASSIFICATION")
print("==============================")

print(f"\nTicket : {ticket_text}")
print(f"Type   : {ticket_type}")
print(f"Queue  : {ticket_queue}")

print(f"\nPriorité prédite : {prediction.upper()}")