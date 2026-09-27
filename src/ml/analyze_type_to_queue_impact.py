import os
import joblib
import pandas as pd

from scipy.sparse import hstack
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)


# ============================================================
# CONFIGURATION
# ============================================================

DATA_PATH = "data/processed/tickets_clean.csv"
QUEUE_MODEL_PATH = "models/ticket_queue_model_v4.pkl"

REPORT_DIR = "reports/type_to_queue_impact"

RANDOM_STATE = 42


# ============================================================
# CHARGEMENT
# ============================================================

print("=" * 70)
print("TYPE → QUEUE IMPACT ANALYSIS")
print("=" * 70)

print("\nChargement du dataset...")

df = pd.read_csv(DATA_PATH)

required_columns = [
    "ticket_text",
    "type",
    "queue",
]

missing = [
    col for col in required_columns
    if col not in df.columns
]

if missing:
    raise ValueError(
        f"Colonnes manquantes : {missing}"
    )

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

df = df[
    df["ticket_text"].str.len() > 0
].copy()

print(f"Tickets disponibles : {len(df)}")


# ============================================================
# CHARGEMENT DU MODELE QUEUE V4
# ============================================================

print("\nChargement du Queue Model V4...")

if not os.path.exists(QUEUE_MODEL_PATH):
    raise FileNotFoundError(
        f"Modèle introuvable : {QUEUE_MODEL_PATH}"
    )

queue_data = joblib.load(
    QUEUE_MODEL_PATH
)

queue_tfidf = queue_data["tfidf"]
queue_encoder = queue_data["encoder"]
queue_classifier = queue_data["classifier"]

print("✅ Queue Model V4 chargé.")


# ============================================================
# SPLIT
# ============================================================

X_text = df["ticket_text"]
X_type = df[["type"]]
y_queue = df["queue"]

(
    X_text_train,
    X_text_test,
    X_type_train,
    X_type_test,
    y_queue_train,
    y_queue_test,
) = train_test_split(
    X_text,
    X_type,
    y_queue,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=y_queue,
)


print("\n" + "=" * 70)
print("TRAIN / TEST SPLIT")
print("=" * 70)

print(f"Train : {len(X_text_train)}")
print(f"Test  : {len(X_text_test)}")


# ============================================================
# TYPE MODEL
# ============================================================

print("\nChargement du Type Model...")

TYPE_MODEL_PATH = "models/ticket_type_model.pkl"

if not os.path.exists(TYPE_MODEL_PATH):
    raise FileNotFoundError(
        f"Type Model introuvable : {TYPE_MODEL_PATH}"
    )

type_model = joblib.load(
    TYPE_MODEL_PATH
)

print("✅ Type Model chargé.")


# ============================================================
# PREDICTION TYPE
# ============================================================

print("\nPrédiction des types...")

predicted_types = type_model.predict(
    X_text_test.tolist()
)

predicted_types = pd.Series(
    predicted_types,
    index=X_text_test.index,
    name="predicted_type",
)

true_types = X_type_test["type"].copy()

print("✅ Types prédits.")


# ============================================================
# CONSTRUCTION DES FEATURES QUEUE
# ============================================================

print("\nVectorisation du texte...")

X_test_text = queue_tfidf.transform(
    X_text_test
)


# ============================================================
# TEST A : TRUE TYPE
# ============================================================

print("\n" + "=" * 70)
print("TEST A — TRUE TYPE → QUEUE")
print("=" * 70)

true_type_df = pd.DataFrame(
    {
        "type": true_types.values
    },
    index=X_text_test.index,
)

true_type_encoded = queue_encoder.transform(
    true_type_df
)

X_true_type = hstack(
    [
        X_test_text,
        true_type_encoded,
    ]
).tocsr()

queue_pred_true_type = queue_classifier.predict(
    X_true_type
)


# ============================================================
# TEST B : PREDICTED TYPE
# ============================================================

print("\n" + "=" * 70)
print("TEST B — PREDICTED TYPE → QUEUE")
print("=" * 70)

predicted_type_df = pd.DataFrame(
    {
        "type": predicted_types.values
    },
    index=X_text_test.index,
)

predicted_type_encoded = queue_encoder.transform(
    predicted_type_df
)

X_predicted_type = hstack(
    [
        X_test_text,
        predicted_type_encoded,
    ]
).tocsr()

queue_pred_predicted_type = queue_classifier.predict(
    X_predicted_type
)


# ============================================================
# FONCTION METRIQUES
# ============================================================

def calculate_metrics(
    y_true,
    y_pred,
):
    return {
        "accuracy": accuracy_score(
            y_true,
            y_pred,
        ),
        "precision_macro": precision_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0,
        ),
        "recall_macro": recall_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0,
        ),
        "macro_f1": f1_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0,
        ),
        "weighted_f1": f1_score(
            y_true,
            y_pred,
            average="weighted",
            zero_division=0,
        ),
    }


# ============================================================
# METRIQUES
# ============================================================

metrics_true_type = calculate_metrics(
    y_queue_test,
    queue_pred_true_type,
)

metrics_predicted_type = calculate_metrics(
    y_queue_test,
    queue_pred_predicted_type,
)


# ============================================================
# RESULTATS
# ============================================================

print("\n" + "=" * 70)
print("COMPARAISON")
print("=" * 70)

comparison = pd.DataFrame(
    [
        {
            "scenario": "TRUE TYPE → QUEUE",
            **metrics_true_type,
        },
        {
            "scenario": "PREDICTED TYPE → QUEUE",
            **metrics_predicted_type,
        },
    ]
)

print(
    comparison.to_string(
        index=False
    )
)


# ============================================================
# IMPACT DU TYPE MODEL
# ============================================================

impact = {
    "accuracy_drop": (
        metrics_true_type["accuracy"]
        - metrics_predicted_type["accuracy"]
    ),
    "precision_macro_drop": (
        metrics_true_type["precision_macro"]
        - metrics_predicted_type["precision_macro"]
    ),
    "recall_macro_drop": (
        metrics_true_type["recall_macro"]
        - metrics_predicted_type["recall_macro"]
    ),
    "macro_f1_drop": (
        metrics_true_type["macro_f1"]
        - metrics_predicted_type["macro_f1"]
    ),
    "weighted_f1_drop": (
        metrics_true_type["weighted_f1"]
        - metrics_predicted_type["weighted_f1"]
    ),
}

print("\n" + "=" * 70)
print("IMPACT DU TYPE MODEL")
print("=" * 70)

for metric, value in impact.items():
    print(
        f"{metric:<25}: {value:+.4f}"
    )


# ============================================================
# COMPARAISON TICKET PAR TICKET
# ============================================================

analysis_df = pd.DataFrame(
    {
        "ticket_text": X_text_test.values,
        "true_type": true_types.values,
        "predicted_type": predicted_types.values,
        "true_queue": y_queue_test.values,
        "queue_with_true_type": queue_pred_true_type,
        "queue_with_predicted_type": queue_pred_predicted_type,
    }
)


analysis_df["type_correct"] = (
    analysis_df["true_type"]
    == analysis_df["predicted_type"]
)

analysis_df["queue_correct_with_true_type"] = (
    analysis_df["true_queue"]
    == analysis_df["queue_with_true_type"]
)

analysis_df["queue_correct_with_predicted_type"] = (
    analysis_df["true_queue"]
    == analysis_df["queue_with_predicted_type"]
)


analysis_df["queue_changed"] = (
    analysis_df["queue_with_true_type"]
    != analysis_df["queue_with_predicted_type"]
)


analysis_df["type_error_caused_queue_error"] = (
    (~analysis_df["type_correct"])
    &
    (
        analysis_df["queue_correct_with_true_type"]
        &
        ~analysis_df[
            "queue_correct_with_predicted_type"
        ]
    )
)


# ============================================================
# STATISTIQUES D'IMPACT
# ============================================================

total = len(analysis_df)

type_errors = (
    ~analysis_df["type_correct"]
).sum()

queue_errors_true_type = (
    ~analysis_df[
        "queue_correct_with_true_type"
    ]
).sum()

queue_errors_predicted_type = (
    ~analysis_df[
        "queue_correct_with_predicted_type"
    ]
).sum()

queue_changed = (
    analysis_df["queue_changed"]
).sum()

type_caused_queue_errors = (
    analysis_df[
        "type_error_caused_queue_error"
    ]
).sum()


print("\n" + "=" * 70)
print("STATISTIQUES D'IMPACT")
print("=" * 70)

print(
    f"\nTickets testés                    : {total}"
)

print(
    f"Erreurs Type Model                : "
    f"{type_errors} "
    f"({type_errors / total:.2%})"
)

print(
    f"Erreurs Queue avec TRUE type      : "
    f"{queue_errors_true_type} "
    f"({queue_errors_true_type / total:.2%})"
)

print(
    f"Erreurs Queue avec PREDICTED type : "
    f"{queue_errors_predicted_type} "
    f"({queue_errors_predicted_type / total:.2%})"
)

print(
    f"Queues modifiées par type erroné  : "
    f"{queue_changed} "
    f"({queue_changed / total:.2%})"
)

print(
    f"Erreurs Queue directement causées "
    f"par Type                       : "
    f"{type_caused_queue_errors} "
    f"({type_caused_queue_errors / total:.2%})"
)


# ============================================================
# RAPPORT TYPE
# ============================================================

print("\n" + "=" * 70)
print("TYPE MODEL — CLASSIFICATION REPORT")
print("=" * 70)

print(
    classification_report(
        true_types,
        predicted_types,
        zero_division=0,
    )
)


# ============================================================
# EXEMPLES D'IMPACT
# ============================================================

impact_examples = analysis_df[
    analysis_df[
        "type_error_caused_queue_error"
    ]
].copy()

impact_examples = impact_examples[
    [
        "ticket_text",
        "true_type",
        "predicted_type",
        "true_queue",
        "queue_with_true_type",
        "queue_with_predicted_type",
    ]
]

impact_examples = impact_examples.head(
    100
)


# ============================================================
# CREATION DOSSIER RAPPORTS
# ============================================================

os.makedirs(
    REPORT_DIR,
    exist_ok=True
)


# ============================================================
# SAUVEGARDE COMPARAISON
# ============================================================

comparison.to_csv(
    os.path.join(
        REPORT_DIR,
        "type_queue_comparison.csv",
    ),
    index=False,
)


# ============================================================
# SAUVEGARDE IMPACT
# ============================================================

pd.DataFrame(
    [
        {
            "metric": key,
            "value": value,
        }
        for key, value in impact.items()
    ]
).to_csv(
    os.path.join(
        REPORT_DIR,
        "type_model_impact.csv",
    ),
    index=False,
)


# ============================================================
# SAUVEGARDE ANALYSE COMPLETE
# ============================================================

analysis_df.to_csv(
    os.path.join(
        REPORT_DIR,
        "type_to_queue_ticket_analysis.csv",
    ),
    index=False,
)


# ============================================================
# SAUVEGARDE EXEMPLES
# ============================================================

impact_examples.to_csv(
    os.path.join(
        REPORT_DIR,
        "type_caused_queue_errors.csv",
    ),
    index=False,
)


# ============================================================
# RAPPORT TYPE
# ============================================================

type_report = pd.DataFrame(
    classification_report(
        true_types,
        predicted_types,
        output_dict=True,
        zero_division=0,
    )
).transpose()

type_report.to_csv(
    os.path.join(
        REPORT_DIR,
        "type_classification_report.csv",
    )
)


# ============================================================
# FIN
# ============================================================

print("\n" + "=" * 70)
print("RAPPORTS SAUVEGARDES")
print("=" * 70)

print(
    f"\nDossier : {REPORT_DIR}"
)

print(
    "\nFichiers créés :"
)

print(
    " - type_queue_comparison.csv"
)

print(
    " - type_model_impact.csv"
)

print(
    " - type_to_queue_ticket_analysis.csv"
)

print(
    " - type_caused_queue_errors.csv"
)

print(
    " - type_classification_report.csv"
)

print("\n✅ Analyse terminée.")

