from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

import joblib
import os
import pandas as pd

from scipy.sparse import hstack
import snowflake.connector

from src.config import get_snowflake_config
from src.data.ticket_store import TicketStore


# ============================================================
# CONFIGURATION DES CHEMINS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

MODELS_DIR = os.path.join(BASE_DIR, "models")


TYPE_MODEL_PATH = os.path.join(
    MODELS_DIR,
    "ticket_type_model.pkl"
)

PRIORITY_MODEL_PATH = os.path.join(
    MODELS_DIR,
    "ticket_priority_model_v2.pkl"
)

QUEUE_MODEL_PATH = os.path.join(
    MODELS_DIR,
    "ticket_queue_model_v3.pkl"
)


# ============================================================
# FASTAPI
# ============================================================

app = FastAPI(
    title="Hardtec Intelligent Ticketing API",
    description="AI API for automatic ticket classification",
    version="1.0.0"
)

# Initialize the persistent ticket store
try:
    ticket_store = TicketStore()
    print("✅ Ticket store initialisé (SQLite local)")
except Exception as e:
    print(f"❌ Erreur lors de l'initialisation du ticket store : {e}")
    ticket_store = None


# ============================================================
# SNOWFLAKE CONNECTION
# ============================================================

def get_snowflake_connection():
    try:
        config = get_snowflake_config()
    except RuntimeError:
        raise RuntimeError(
            "Snowflake configuration is missing. Provide SNOWFLAKE_ACCOUNT, "
            "SNOWFLAKE_USER and SNOWFLAKE_PASSWORD in the environment or .env file."
        )

    return snowflake.connector.connect(
        account=config["account"],
        user=config["user"],
        password=config["password"],
        warehouse=config["warehouse"],
        database=config["database"],
        schema=config["schema"],
        role=config["role"],
    )


# ============================================================
# SAVE PREDICTION INTO SNOWFLAKE
# ============================================================

def save_prediction_to_snowflake(
    ticket_text,
    predicted_type,
    predicted_priority,
    predicted_queue
):

    conn = None
    cursor = None

    try:
        conn = get_snowflake_connection()

    except RuntimeError as exc:
        print(f"⚠️  Snowflake non configuré, persistance locale ignorée : {exc}")
        return False

    try:

        cursor = conn.cursor()

        sql = """
        INSERT INTO TICKET_PREDICTIONS
        (
            TICKET_TEXT,
            TYPE,
            PRIORITY,
            QUEUE
        )
        VALUES (%s, %s, %s, %s)
        """

        cursor.execute(
            sql,
            (
                ticket_text,
                predicted_type,
                predicted_priority,
                predicted_queue
            )
        )

        conn.commit()

        print(
            "✅ Prédiction enregistrée dans Snowflake"
        )
        return True

    except Exception as e:

        print(
            f"❌ Erreur Snowflake : {e}"
        )
        return False

    finally:

        if cursor is not None:
            cursor.close()

        if conn is not None:
            conn.close()


# ============================================================
# CHARGEMENT DES MODÈLES
# ============================================================

print("Chargement des modèles...")


try:

    # --------------------------------------------------------
    # TYPE MODEL
    # --------------------------------------------------------

    type_model = joblib.load(
        TYPE_MODEL_PATH
    )


    # --------------------------------------------------------
    # PRIORITY MODEL
    # --------------------------------------------------------

    priority_data = joblib.load(
        PRIORITY_MODEL_PATH
    )


    # --------------------------------------------------------
    # QUEUE MODEL
    # --------------------------------------------------------

    queue_data = joblib.load(
        QUEUE_MODEL_PATH
    )


    print("✅ Modèles chargés")


except Exception as e:

    print(
        f"❌ Erreur lors du chargement des modèles : {e}"
    )

    type_model = None
    priority_data = None
    queue_data = None


# ============================================================
# SCHEMAS
# ============================================================

class TicketRequest(BaseModel):

    ticket_text: str


class TicketResponse(BaseModel):

    ticket_text: str
    type: str
    priority: str
    queue: str
    confidence: float
    recommendation: str


# ============================================================
# ROOT
# ============================================================

@app.get("/")
def root():

    return {
        "application": "Hardtec Intelligent Ticketing",
        "status": "running"
    }


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
def health():

    models_loaded = (
        type_model is not None
        and priority_data is not None
        and queue_data is not None
    )

    return {

        "application": "Hardtec Intelligent Ticketing",

        "status": "healthy",

        "models_loaded": models_loaded

    }


# ============================================================
# PREDICTION
# ============================================================

@app.post(
    "/predict",
    response_model=TicketResponse
)
def predict(ticket: TicketRequest):


    # ========================================================
    # 1. VALIDATION
    # ========================================================

    if not ticket.ticket_text.strip():

        raise HTTPException(
            status_code=400,
            detail="ticket_text ne peut pas être vide"
        )


    if (
        type_model is None
        or priority_data is None
        or queue_data is None
    ):

        raise HTTPException(
            status_code=500,
            detail="Les modèles ML ne sont pas correctement chargés"
        )


    text = ticket.ticket_text.strip()


    # ========================================================
    # 2. TYPE CLASSIFICATION
    # ========================================================

    predicted_type = type_model.predict(
        [text]
    )[0]


    predicted_type = str(
        predicted_type
    )


    # ========================================================
    # 3. CHARGEMENT DU MODÈLE PRIORITY
    # ========================================================

    priority_tfidf = priority_data["tfidf"]

    priority_encoder = priority_data["encoder"]

    priority_classifier = priority_data["classifier"]


    # ========================================================
    # 4. TF-IDF DU TICKET
    # ========================================================

    priority_text_vector = priority_tfidf.transform(
        [text]
    )


    # ========================================================
    # 5. PREMIÈRE PRÉDICTION PRIORITY
    # ========================================================

    # Le modèle Priority utilise :
    #
    # ticket_text
    # type
    # queue
    #
    # Mais la queue n'est pas encore connue.
    #
    # On utilise donc Technical Support comme valeur
    # temporaire pour obtenir une première priorité.

    default_queue = "Technical Support"


    priority_meta = pd.DataFrame([
        {
            "type": predicted_type,
            "queue": default_queue
        }
    ])


    priority_meta_vector = priority_encoder.transform(
        priority_meta
    )


    priority_initial_vector = hstack([
        priority_text_vector,
        priority_meta_vector
    ]).tocsr()


    predicted_priority = priority_classifier.predict(
        priority_initial_vector
    )[0]


    predicted_priority = str(
        predicted_priority
    )


    # ========================================================
    # 6. QUEUE CLASSIFICATION
    # ========================================================

    queue_tfidf = queue_data["tfidf"]

    queue_encoder = queue_data["encoder"]

    queue_classifier = queue_data["classifier"]


    queue_text_vector = queue_tfidf.transform(
        [text]
    )


    # Le modèle Queue utilise :
    #
    # ticket_text
    # type
    # priority

    queue_meta = pd.DataFrame([
        {
            "type": predicted_type,
            "priority": predicted_priority
        }
    ])


    queue_meta_vector = queue_encoder.transform(
        queue_meta
    )


    queue_final_vector = hstack([
        queue_text_vector,
        queue_meta_vector
    ]).tocsr()


    predicted_queue = queue_classifier.predict(
        queue_final_vector
    )[0]


    predicted_queue = str(
        predicted_queue
    )


    # ========================================================
    # 7. REFINEMENT DE LA PRIORITY
    # ========================================================

    # Maintenant que nous connaissons la queue réelle,
    # nous recalculons la priorité.

    priority_meta_final = pd.DataFrame([
        {
            "type": predicted_type,
            "queue": predicted_queue
        }
    ])


    priority_meta_final_vector = priority_encoder.transform(
        priority_meta_final
    )


    priority_final_vector = hstack([
        priority_text_vector,
        priority_meta_final_vector
    ]).tocsr()


    predicted_priority = priority_classifier.predict(
        priority_final_vector
    )[0]


    predicted_priority = str(
        predicted_priority
    ).upper()


    # ========================================================
    # 8. LOG CONSOLE
    # ========================================================

    print()
    print("=======================================================")
    print("          AI TICKET CLASSIFICATION")
    print("=======================================================")

    print()
    print(
        f"Ticket : {text}"
    )

    print()
    print("----------------------------------------")

    print(
        f"Type      : {predicted_type}"
    )

    print(
        f"Priority  : {predicted_priority}"
    )

    print(
        f"Queue     : {predicted_queue}"
    )

    print(
        "----------------------------------------"
    )


    # ========================================================
    # 9. SAVE INTO SNOWFLAKE
    # ========================================================

    try:
        save_prediction_to_snowflake(
            text,
            predicted_type,
            predicted_priority,
            predicted_queue
        )
    except Exception as e:
        print(f"⚠️  La persistance Snowflake a été ignorée : {e}")

    # ========================================================
    # 9b. SAVE INTO LOCAL SQLITE
    # ========================================================

    try:
        if ticket_store:
            ticket_store.save_prediction(
                ticket_text=text,
                ticket_type=predicted_type,
                priority=predicted_priority,
                queue=predicted_queue,
                confidence=0.85,
                recommendation=None,
                source="api"
            )
            print("✅ Prédiction enregistrée localement")
    except Exception as e:
        print(f"⚠️  La persistance locale a échoué : {e}")

    # ========================================================
    # 10. RESPONSE
    # ========================================================

    print(
        "✅ Classification terminée."
    )

    if predicted_priority.upper() in {"HIGH", "CRITICAL"}:
        recommendation = "Escalate to the IT operations or infrastructure team immediately."
    elif predicted_queue.lower() in {"billing", "finance"}:
        recommendation = "Route to the finance support queue and verify the account context."
    else:
        recommendation = "Assign to the relevant support queue and monitor the ticket lifecycle."

    return {
        "ticket_text": text,
        "type": predicted_type,
        "priority": predicted_priority,
        "queue": predicted_queue,
        "confidence": 0.85,
        "recommendation": recommendation,
    }


# ============================================================
# TICKET HISTORY
# ============================================================

@app.get("/history")
def get_ticket_history(limit: int = 50):
    """Retrieve recent ticket predictions from the local store."""
    if not ticket_store:
        raise HTTPException(
            status_code=500,
            detail="Ticket store is not available"
        )

    try:
        predictions = ticket_store.get_recent_predictions(limit=limit)
        return {"predictions": predictions, "count": len(predictions)}
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve history: {str(e)}"
        )


# ============================================================
# STATISTICS
# ============================================================

@app.get("/stats")
def get_prediction_statistics():
    """Retrieve summary statistics about ticket predictions."""
    if not ticket_store:
        raise HTTPException(
            status_code=500,
            detail="Ticket store is not available"
        )

    try:
        stats = ticket_store.get_stats()
        return stats
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve statistics: {str(e)}"
        )