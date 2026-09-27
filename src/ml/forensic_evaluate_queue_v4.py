import os
import pandas as pd
import joblib

from scipy.sparse import hstack
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)


# ============================================================
# CONFIGURATION
# ============================================================

DATA_PATH = "data/processed/tickets_clean.csv"
MODEL_PATH = "models/ticket_queue_model_v4.pkl"

RANDOM_STATE = 42
TEST_SIZE = 0.20


# ============================================================
# UTILITAIRES
# ============================================================

def print_separator(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def calculate_metrics(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
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


def print_metrics(metrics):
    print(f"Accuracy        : {metrics['accuracy']:.4f}")
    print(f"Precision Macro : {metrics['precision_macro']:.4f}")
    print(f"Recall Macro    : {metrics['recall_macro']:.4f}")
    print(f"Macro F1        : {metrics['macro_f1']:.4f}")
    print(f"Weighted F1     : {metrics['weighted_f1']:.4f}")


# ============================================================
# CHARGEMENT DATASET
# ============================================================

print_separator("FORENSIC EVALUATION - QUEUE CLASSIFIER V4")

print("\nChargement du dataset...")

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(
        f"Dataset introuvable : {DATA_PATH}"
    )

df = pd.read_csv(DATA_PATH)

print(f"Nombre total de lignes : {len(df)}")
print(f"Colonnes : {df.columns.tolist()}")


# ============================================================
# VALIDATION DES COLONNES
# ============================================================

required_columns = [
    "ticket_text",
    "type",
    "queue",
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
).copy()

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

df = df.reset_index(drop=True)

print(
    f"\nNombre de tickets après nettoyage : {len(df)}"
)


# ============================================================
# FEATURES
# ============================================================

X_text = df["ticket_text"]

X_meta = df[["type"]]

y = df["queue"]


# ============================================================
# SPLIT EXACTEMENT COMME V4
# ============================================================

print_separator("CONSTRUCTION DU SPLIT V4")

(
    X_text_train,
    X_text_test,
    X_meta_train,
    X_meta_test,
    y_train,
    y_test,
) = train_test_split(
    X_text,
    X_meta,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y,
)

print(f"Train : {len(X_text_train)}")
print(f"Test  : {len(X_text_test)}")

print("\nStratification utilisée : queue")
print(f"Random state : {RANDOM_STATE}")
print(f"Test size    : {TEST_SIZE}")


# ============================================================
# TF-IDF EXACTEMENT COMME V4
# ============================================================

print_separator("TF-IDF V4")

tfidf_retrained = TfidfVectorizer(
    lowercase=True,
    max_features=40000,
    ngram_range=(1, 2),
    min_df=2,
    sublinear_tf=True,
)

X_train_text = tfidf_retrained.fit_transform(
    X_text_train
)

X_test_text = tfidf_retrained.transform(
    X_text_test
)

print(
    f"Features TF-IDF : {X_train_text.shape[1]}"
)

print(
    f"Train text shape : {X_train_text.shape}"
)

print(
    f"Test text shape  : {X_test_text.shape}"
)


# ============================================================
# ENCODER EXACTEMENT COMME V4
# ============================================================

print_separator("ONE-HOT ENCODER V4")

encoder_retrained = OneHotEncoder(
    handle_unknown="ignore"
)

X_train_meta = encoder_retrained.fit_transform(
    X_meta_train
)

X_test_meta = encoder_retrained.transform(
    X_meta_test
)

print(
    f"Metadata features : {X_train_meta.shape[1]}"
)

print(
    f"Train metadata shape : {X_train_meta.shape}"
)

print(
    f"Test metadata shape  : {X_test_meta.shape}"
)

print("\nCatégories du type :")

for category in encoder_retrained.categories_[0]:
    print(f"  - {category}")


# ============================================================
# COMBINAISON
# ============================================================

X_train = hstack(
    [
        X_train_text,
        X_train_meta,
    ]
).tocsr()

X_test = hstack(
    [
        X_test_text,
        X_test_meta,
    ]
).tocsr()

print_separator("FEATURES FINALES")

print(
    f"Train final : {X_train.shape}"
)

print(
    f"Test final  : {X_test.shape}"
)


# ============================================================
# RETRAINING EXACT V4
# ============================================================

print_separator("RETRAINING EXACT V4")

classifier_retrained = LinearSVC(
    C=1.5,
    class_weight="balanced",
)

classifier_retrained.fit(
    X_train,
    y_train,
)

print("\n✅ Modèle V4 temporaire entraîné.")


# ============================================================
# PREDICTION RETRAINED MODEL
# ============================================================

y_pred_retrained = classifier_retrained.predict(
    X_test
)


# ============================================================
# EVALUATION RETRAINED MODEL
# ============================================================

metrics_retrained = calculate_metrics(
    y_test,
    y_pred_retrained,
)

print_separator("RESULTATS - RETRAINING V4")

print_metrics(metrics_retrained)


print("\nClassification Report :")

print(
    classification_report(
        y_test,
        y_pred_retrained,
        zero_division=0,
    )
)


print("\nConfusion Matrix :")

labels = sorted(
    y.unique()
)

cm_retrained = confusion_matrix(
    y_test,
    y_pred_retrained,
    labels=labels,
)

cm_retrained_df = pd.DataFrame(
    cm_retrained,
    index=[f"TRUE: {label}" for label in labels],
    columns=[f"PRED: {label}" for label in labels],
)

print(cm_retrained_df.to_string())


# ============================================================
# CHARGEMENT DU VRAI MODELE V4
# ============================================================

print_separator("CHARGEMENT DU MODELE V4 SAUVEGARDE")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Modèle introuvable : {MODEL_PATH}"
    )

model_data = joblib.load(
    MODEL_PATH
)

print(
    f"Type de l'objet chargé : {type(model_data)}"
)

if not isinstance(model_data, dict):
    raise TypeError(
        "Le modèle V4 attendu doit être un dictionnaire."
    )

required_model_keys = [
    "tfidf",
    "encoder",
    "classifier",
]

missing_model_keys = [
    key
    for key in required_model_keys
    if key not in model_data
]

if missing_model_keys:
    raise KeyError(
        f"Clés manquantes dans le modèle : "
        f"{missing_model_keys}"
    )


tfidf_saved = model_data["tfidf"]
encoder_saved = model_data["encoder"]
classifier_saved = model_data["classifier"]


print("\nStructure du modèle :")

print(
    f"TF-IDF    : {type(tfidf_saved)}"
)

print(
    f"Encoder   : {type(encoder_saved)}"
)

print(
    f"Classifier: {type(classifier_saved)}"
)


# ============================================================
# INSPECTION TF-IDF SAUVEGARDE
# ============================================================

print_separator("INSPECTION DU TF-IDF SAUVEGARDE")

print(
    f"max_features : {tfidf_saved.max_features}"
)

print(
    f"ngram_range  : {tfidf_saved.ngram_range}"
)

print(
    f"min_df       : {tfidf_saved.min_df}"
)

print(
    f"sublinear_tf : {tfidf_saved.sublinear_tf}"
)

print(
    f"lowercase    : {tfidf_saved.lowercase}"
)

print(
    f"Nombre de features apprises : "
    f"{len(tfidf_saved.vocabulary_)}"
)


# ============================================================
# INSPECTION ENCODER SAUVEGARDE
# ============================================================

print_separator("INSPECTION DE L'ENCODER SAUVEGARDE")

print(
    f"Nombre de catégories : "
    f"{len(encoder_saved.categories_[0])}"
)

print("\nCatégories :")

for category in encoder_saved.categories_[0]:
    print(f"  - {category}")


# ============================================================
# INSPECTION CLASSIFIER SAUVEGARDE
# ============================================================

print_separator("INSPECTION DU CLASSIFIER SAUVEGARDE")

print(
    f"Type : {type(classifier_saved)}"
)

if hasattr(classifier_saved, "C"):
    print(
        f"C : {classifier_saved.C}"
    )

if hasattr(classifier_saved, "class_weight"):
    print(
        f"class_weight : "
        f"{classifier_saved.class_weight}"
    )

if hasattr(classifier_saved, "classes_"):
    print("\nClasses du classifier :")

    for class_name in classifier_saved.classes_:
        print(f"  - {class_name}")


# ============================================================
# TRANSFORMATION DU TEST AVEC LE MODELE SAUVEGARDE
# ============================================================

print_separator(
    "TRANSFORMATION DU TEST AVEC LE MODELE SAUVEGARDE"
)

X_saved_test_text = tfidf_saved.transform(
    X_text_test
)

X_saved_test_meta = encoder_saved.transform(
    X_meta_test
)

X_saved_test = hstack(
    [
        X_saved_test_text,
        X_saved_test_meta,
    ]
).tocsr()

print(
    f"Test text shape : "
    f"{X_saved_test_text.shape}"
)

print(
    f"Test meta shape : "
    f"{X_saved_test_meta.shape}"
)

print(
    f"Test final shape : "
    f"{X_saved_test.shape}"
)


# ============================================================
# COMPARAISON DES DIMENSIONS
# ============================================================

print_separator("COMPARAISON DES DIMENSIONS")

print(
    "TF-IDF retrained : "
    f"{X_train_text.shape[1]}"
)

print(
    "TF-IDF saved     : "
    f"{X_saved_test_text.shape[1]}"
)

print(
    "Meta retrained   : "
    f"{X_train_meta.shape[1]}"
)

print(
    "Meta saved       : "
    f"{X_saved_test_meta.shape[1]}"
)

print(
    "Final retrained  : "
    f"{X_test.shape[1]}"
)

print(
    "Final saved      : "
    f"{X_saved_test.shape[1]}"
)


if (
    X_train_text.shape[1]
    == X_saved_test_text.shape[1]
):
    print("\n✅ Nombre de features TF-IDF compatible.")
else:
    print(
        "\n⚠️ DIFFERENCE dans le nombre de features TF-IDF."
    )


if (
    X_train_meta.shape[1]
    == X_saved_test_meta.shape[1]
):
    print("✅ Nombre de features metadata compatible.")
else:
    print(
        "⚠️ DIFFERENCE dans le nombre de features metadata."
    )


if (
    X_test.shape[1]
    == X_saved_test.shape[1]
):
    print("✅ Dimension finale compatible.")
else:
    print(
        "⚠️ DIFFERENCE dans la dimension finale."
    )


# ============================================================
# PREDICTION MODELE SAUVEGARDE
# ============================================================

print_separator("PREDICTION - MODELE V4 SAUVEGARDE")

y_pred_saved = classifier_saved.predict(
    X_saved_test
)

print(
    f"Nombre de prédictions : "
    f"{len(y_pred_saved)}"
)


# ============================================================
# EVALUATION MODELE SAUVEGARDE
# ============================================================

metrics_saved = calculate_metrics(
    y_test,
    y_pred_saved,
)

print_separator("RESULTATS - MODELE V4 SAUVEGARDE")

print_metrics(metrics_saved)


print("\nClassification Report :")

print(
    classification_report(
        y_test,
        y_pred_saved,
        zero_division=0,
    )
)


print("\nConfusion Matrix :")

cm_saved = confusion_matrix(
    y_test,
    y_pred_saved,
    labels=labels,
)

cm_saved_df = pd.DataFrame(
    cm_saved,
    index=[f"TRUE: {label}" for label in labels],
    columns=[f"PRED: {label}" for label in labels],
)

print(cm_saved_df.to_string())


# ============================================================
# COMPARAISON RETRAINED VS SAVED
# ============================================================

print_separator(
    "COMPARAISON RETRAINED V4 VS MODELE SAUVEGARDE"
)

comparison = pd.DataFrame(
    [
        {
            "model": "Retrained V4",
            **metrics_retrained,
        },
        {
            "model": "Saved V4",
            **metrics_saved,
        },
    ]
)

print(
    comparison.to_string(
        index=False
    )
)


# ============================================================
# DIFFERENCE DES METRIQUES
# ============================================================

print_separator("DIFFERENCES DE METRIQUES")

for metric_name in [
    "accuracy",
    "precision_macro",
    "recall_macro",
    "macro_f1",
    "weighted_f1",
]:

    difference = (
        metrics_saved[metric_name]
        - metrics_retrained[metric_name]
    )

    print(
        f"{metric_name:20s}: "
        f"{difference:+.6f}"
    )


# ============================================================
# COMPARAISON DES PREDICTIONS
# ============================================================

print_separator(
    "COMPARAISON DES PREDICTIONS"
)

same_predictions = (
    y_pred_retrained
    == y_pred_saved
)

number_same = int(
    same_predictions.sum()
)

number_different = int(
    (~same_predictions).sum()
)

total_predictions = len(
    y_pred_saved
)

print(
    f"Prédictions identiques : "
    f"{number_same}/{total_predictions}"
)

print(
    f"Prédictions différentes : "
    f"{number_different}/{total_predictions}"
)

print(
    f"Taux de concordance : "
    f"{number_same / total_predictions:.4%}"
)


# ============================================================
# ANALYSE DES DIFFERENCES
# ============================================================

if number_different > 0:

    print_separator(
        "EXEMPLES DE PREDICTIONS DIFFERENTES"
    )

    difference_indices = [
        i
        for i, same in enumerate(
            same_predictions
        )
        if not same
    ]

    max_examples = min(
        20,
        len(difference_indices)
    )

    for position in difference_indices[
        :max_examples
    ]:

        original_index = X_text_test.index[
            position
        ]

        ticket_text = X_text_test.iloc[
            position
        ]

        true_queue = y_test.iloc[
            position
        ]

        retrained_prediction = (
            y_pred_retrained[position]
        )

        saved_prediction = (
            y_pred_saved[position]
        )

        print("\n" + "-" * 70)

        print(
            f"Ticket #{position + 1}"
        )

        print(
            f"TRUE QUEUE       : {true_queue}"
        )

        print(
            f"RETRAINED V4     : "
            f"{retrained_prediction}"
        )

        print(
            f"SAVED V4         : "
            f"{saved_prediction}"
        )

        print(
            f"DATASET INDEX    : "
            f"{original_index}"
        )

        print(
            f"TYPE             : "
            f"{X_meta_test.iloc[position]['type']}"
        )

        print(
            f"TICKET TEXT      : "
            f"{ticket_text}"
        )

else:

    print(
        "\n✅ Les deux modèles produisent "
        "exactement les mêmes prédictions."
    )


# ============================================================
# TEST FINAL DE REPRODUCTIBILITE
# ============================================================

print_separator(
    "TEST FINAL DE REPRODUCTIBILITE"
)

reproducible = True

if X_train_text.shape[1] != X_saved_test_text.shape[1]:
    reproducible = False

if X_train_meta.shape[1] != X_saved_test_meta.shape[1]:
    reproducible = False

if X_test.shape[1] != X_saved_test.shape[1]:
    reproducible = False

if number_different > 0:
    reproducible = False


if reproducible:

    print(
        "\n✅ REPRODUCTIBILITE CONFIRMEE"
    )

    print(
        "\nLe modèle sauvegardé V4 correspond "
        "au modèle reconstruit avec la même "
        "configuration et produit les mêmes "
        "prédictions sur le même test set."
    )

else:

    print(
        "\n⚠️ DIFFERENCE DETECTEE"
    )

    print(
        "\nLe modèle sauvegardé V4 ne correspond "
        "pas parfaitement au modèle reconstruit "
        "avec cette configuration."
    )


# ============================================================
# DIAGNOSTIC FINAL
# ============================================================

print_separator("DIAGNOSTIC FINAL")

print(
    "\nRESULTAT RETRAINED V4 :"
)

print_metrics(
    metrics_retrained
)

print(
    "\nRESULTAT SAVED V4 :"
)

print_metrics(
    metrics_saved
)

print("\nInterprétation :")

if (
    metrics_saved["accuracy"] < 0.60
    and metrics_saved["macro_f1"] < 0.60
):

    print(
        "\n⚠️ Le modèle V4 présente une performance "
        "faible sur ce split."
    )

    print(
        "Cela confirme que le problème Queue "
        "mérite une investigation supplémentaire."
    )

elif (
    metrics_saved["accuracy"] >= 0.80
):

    print(
        "\n✅ Le modèle V4 présente une bonne "
        "performance globale."
    )

else:

    print(
        "\n🟡 Le modèle V4 présente une performance "
        "intermédiaire."
    )


print(
    "\nIMPORTANT :"
)

print(
    "Cette analyse ne modifie ni le dataset, "
    "ni le modèle V4, ni predictor.py, "
    "ni app.py."
)

print("\nTerminé.")

