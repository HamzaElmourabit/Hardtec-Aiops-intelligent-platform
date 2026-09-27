import sys
import os
import joblib
import pandas as pd
from scipy.sparse import hstack, csr_matrix


# ============================================================
# CONFIGURATION
# ============================================================

TYPE_MODEL_PATH = "models/ticket_type_model.pkl"
PRIORITY_MODEL_PATH = "models/ticket_priority_model_v2.pkl"
QUEUE_MODEL_PATH = "models/ticket_queue_model_v3.pkl"


# ============================================================
# VERIFICATION DES ARGUMENTS
# ============================================================

if len(sys.argv) < 2:
    print('Usage:')
    print('python src\\pipeline\\ticket_classifier.py "Your ticket text"')
    sys.exit(1)

ticket_text = sys.argv[1]


# ============================================================
# CHARGEMENT DES MODELES
# ============================================================

print("\nChargement des modèles...")

try:
    type_model = joblib.load(TYPE_MODEL_PATH)
    priority_data = joblib.load(PRIORITY_MODEL_PATH)
    queue_data = joblib.load(QUEUE_MODEL_PATH)

    print("✅ Modèles chargés")

except Exception as e:
    print(f"❌ Erreur lors du chargement des modèles : {e}")
    sys.exit(1)


# ============================================================
# EXTRACTION DES COMPOSANTS
# ============================================================

priority_tfidf = priority_data["tfidf"]
priority_encoder = priority_data["encoder"]
priority_classifier = priority_data["classifier"]

queue_tfidf = queue_data["tfidf"]
queue_encoder = queue_data["encoder"]
queue_classifier = queue_data["classifier"]


# ============================================================
# 1. PREDICTION DU TYPE
# ============================================================

predicted_type = type_model.predict([ticket_text])[0]


# ============================================================
# 2. PREDICTION DE LA QUEUE
# ============================================================
#
# IMPORTANT :
# Le modèle Queue V3 a été entraîné avec :
#
# ["type", "priority"]
#
# Nous devons donc lui fournir une priority.
#
# Comme la priorité dépend actuellement de la queue,
# il existe une dépendance circulaire.
#
# Pour éviter de bloquer le pipeline, on utilise ici
# une priorité initiale basée sur le texte uniquement.
#
# ============================================================

# On utilise une estimation initiale de priorité
# basée sur le modèle Priority uniquement avec une queue
# par défaut.
#
# Queue par défaut choisie selon la distribution du dataset.
default_queue = "Technical Support"


priority_input = pd.DataFrame([
    {
        "type": predicted_type,
        "queue": default_queue
    }
])

priority_meta = priority_encoder.transform(priority_input)

priority_text = priority_tfidf.transform([ticket_text])

priority_features = hstack([
    priority_text,
    priority_meta
]).tocsr()

predicted_priority = priority_classifier.predict(
    priority_features
)[0]


# ============================================================
# 3. PREDICTION DE LA QUEUE
# ============================================================

queue_input = pd.DataFrame([
    {
        "type": predicted_type,
        "priority": predicted_priority
    }
])

queue_meta = queue_encoder.transform(queue_input)

queue_text = queue_tfidf.transform([ticket_text])

queue_features = hstack([
    queue_text,
    queue_meta
]).tocsr()

predicted_queue = queue_classifier.predict(
    queue_features
)[0]


# ============================================================
# 4. AFFICHAGE
# ============================================================

print("\n")
print("=" * 55)
print("          AI TICKET CLASSIFICATION")
print("=" * 55)

print(f"\nTicket : {ticket_text}")

print("\n----------------------------------------")
print(f"Type      : {predicted_type}")
print(f"Priority  : {predicted_priority.upper()}")
print(f"Queue     : {predicted_queue}")
print("----------------------------------------")

print("\n✅ Classification terminée.")