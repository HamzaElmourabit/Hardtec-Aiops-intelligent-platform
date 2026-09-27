import os
from typing import Dict

import joblib
import pandas as pd

from scipy.sparse import hstack


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        ".."
    )
)

MODELS_DIR = os.path.join(
    BASE_DIR,
    "models"
)


TYPE_MODEL_PATH = os.path.join(
    MODELS_DIR,
    "ticket_type_model.pkl"
)

QUEUE_MODEL_PATH = os.path.join(
    MODELS_DIR,
    "ticket_queue_model_v4.pkl"
)

PRIORITY_MODEL_PATH = os.path.join(
    MODELS_DIR,
    "ticket_priority_model_v3.pkl"
)


# ============================================================
# CHARGEMENT DES MODELES
# ============================================================

def _load_models() -> Dict[str, object]:
    """
    Charge les trois modèles ML.
    """

    if not os.path.exists(TYPE_MODEL_PATH):
        raise FileNotFoundError(
            f"Modèle TYPE introuvable : "
            f"{TYPE_MODEL_PATH}"
        )

    if not os.path.exists(QUEUE_MODEL_PATH):
        raise FileNotFoundError(
            f"Modèle QUEUE introuvable : "
            f"{QUEUE_MODEL_PATH}"
        )

    if not os.path.exists(PRIORITY_MODEL_PATH):
        raise FileNotFoundError(
            f"Modèle PRIORITY introuvable : "
            f"{PRIORITY_MODEL_PATH}"
        )


    type_model = joblib.load(
        TYPE_MODEL_PATH
    )

    queue_data = joblib.load(
        QUEUE_MODEL_PATH
    )

    priority_data = joblib.load(
        PRIORITY_MODEL_PATH
    )


    return {
        "type_model": type_model,
        "queue_data": queue_data,
        "priority_data": priority_data
    }


# ============================================================
# PREDICTION
# ============================================================

def predict_ticket(ticket_text: str) -> dict:
    """
    Predict ticket type, queue and priority.

    Pipeline:

        Ticket text
             ↓
           Type
             ↓
           Queue
             ↓
         Priority

    Parameters
    ----------
    ticket_text : str
        Description du ticket.

    Returns
    -------
    dict
        {
            "ticket_type": ...,
            "priority": ...,
            "queue": ...
        }
    """


    # ========================================================
    # VALIDATION
    # ========================================================

    if not isinstance(
        ticket_text,
        str
    ):
        raise ValueError(
            "ticket_text doit être une chaîne de caractères."
        )


    text = ticket_text.strip()


    if not text:
        raise ValueError(
            "La description du ticket ne peut pas être vide."
        )


    # ========================================================
    # CHARGEMENT
    # ========================================================

    models = _load_models()


    type_model = models[
        "type_model"
    ]

    queue_data = models[
        "queue_data"
    ]

    priority_data = models[
        "priority_data"
    ]


    # ========================================================
    # 1. PREDICTION DU TYPE
    # ========================================================

    predicted_type = str(
        type_model.predict(
            [text]
        )[0]
    )


    # ========================================================
    # 2. PREDICTION DE LA QUEUE
    # ========================================================
    #
    # Queue model :
    #
    # ticket_text + type
    #        ↓
    #      queue
    #
    # ========================================================

    queue_tfidf = queue_data[
        "tfidf"
    ]

    queue_encoder = queue_data[
        "encoder"
    ]

    queue_classifier = queue_data[
        "classifier"
    ]


    queue_text_vector = queue_tfidf.transform(
        [text]
    )


    queue_meta = pd.DataFrame(
        [
            {
                "type": predicted_type
            }
        ]
    )


    queue_meta_vector = queue_encoder.transform(
        queue_meta
    )


    queue_final_vector = hstack(
        [
            queue_text_vector,
            queue_meta_vector
        ]
    ).tocsr()


    predicted_queue = str(
        queue_classifier.predict(
            queue_final_vector
        )[0]
    )


    # ========================================================
    # 3. PREDICTION DE LA PRIORITE
    # ========================================================
    #
    # Priority model :
    #
    # ticket_text + type + queue
    #              ↓
    #           priority
    #
    # ========================================================

    priority_tfidf = priority_data[
        "tfidf"
    ]

    priority_encoder = priority_data[
        "encoder"
    ]

    priority_classifier = priority_data[
        "classifier"
    ]


    priority_text_vector = priority_tfidf.transform(
        [text]
    )


    priority_meta = pd.DataFrame(
        [
            {
                "type": predicted_type,
                "queue": predicted_queue
            }
        ]
    )


    priority_meta_vector = priority_encoder.transform(
        priority_meta
    )


    priority_final_vector = hstack(
        [
            priority_text_vector,
            priority_meta_vector
        ]
    ).tocsr()


    predicted_priority = str(
        priority_classifier.predict(
            priority_final_vector
        )[0]
    ).upper()


    # ========================================================
    # RESULTAT
    # ========================================================

    return {
        "ticket_type": predicted_type,
        "priority": predicted_priority,
        "queue": predicted_queue
    }