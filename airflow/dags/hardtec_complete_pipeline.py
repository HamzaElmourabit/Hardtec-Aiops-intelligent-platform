from datetime import datetime
from pathlib import Path

from airflow import DAG
from airflow.operators.python import PythonOperator


# ============================================================
# CONFIGURATION
# ============================================================

def get_repo_root() -> Path:
    candidates = [
        Path("/workspace"),
        Path(__file__).resolve().parents[2],
    ]

    for candidate in candidates:
        if candidate.exists() and (candidate / "src").exists():
            return candidate

    return Path(__file__).resolve().parents[2]


REPO_ROOT = get_repo_root()
DATA_DIR = REPO_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
TICKET_DATASET = PROCESSED_DIR / "tickets_clean.csv"
TICKET_STORE_DB = DATA_DIR / "lake" / "tickets_predictions.db"


# ============================================================
# TASKS
# ============================================================

def validate_ticket_dataset():
    import pandas as pd

    if not TICKET_DATASET.exists():
        raise FileNotFoundError(f"Ticket dataset missing: {TICKET_DATASET}")

    df = pd.read_csv(TICKET_DATASET)
    required = ["ticket_text", "type", "queue", "priority", "language"]
    missing = [col for col in required if col not in df.columns]

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    if df.empty:
        raise ValueError("Ticket dataset is empty.")

    print(f"Loaded {len(df)} ticket rows from {TICKET_DATASET}")


def run_ticket_prediction():
    import sys

    sys.path.insert(0, str(REPO_ROOT))
    from src.pipeline.predictor import predict_ticket
    from src.data.ticket_store import TicketStore

    sample = (
        "Database server is down and users cannot access the ERP system. "
        "Critical incident affecting finance and operations."
    )

    prediction = predict_ticket(sample)
    ticket_store = TicketStore(str(TICKET_STORE_DB))

    record_id = ticket_store.save_prediction(
        ticket_text=sample,
        ticket_type=prediction["ticket_type"],
        priority=prediction["priority"],
        queue=prediction["queue"],
        confidence=prediction.get("confidence", 0.85),
        recommendation=(
            f"Assign the ticket to the {prediction['queue']} team "
            f"with {prediction['priority']} priority."
        ),
        source="airflow",
    )

    print(f"Ticket prediction stored with id={record_id}")
    print(prediction)


def run_rag_lookup():
    import sys
    import pickle

    sys.path.insert(0, str(REPO_ROOT))
    from src.rag.rag_engine import retrieve_similar_tickets

    query = "network outage affecting multiple users and VPN access"
    results = retrieve_similar_tickets(query, top_k=3)

    if not results:
        raise ValueError("RAG returned no results for the test query.")

    print("RAG search results:")
    for item in results:
        doc = item.get("document", "") if isinstance(item, dict) else str(item)
        print(doc[:180])


def run_aiops_risk_check():
    import sys

    sys.path.insert(0, str(REPO_ROOT))
    from src.aiops.micross.risk_predictor import MicroSSRiskPredictor

    predictor = MicroSSRiskPredictor()
    timestamps = predictor.get_available_timestamps(remove_initial_history=True)

    if not timestamps:
        raise ValueError("No MicroSS timestamps available for prediction.")

    result = predictor.predict(timestamps[0])
    print("AIOps result:")
    print(result)


def summarize_storage():
    import sqlite3

    db_path = TICKET_STORE_DB
    if not db_path.exists():
        raise FileNotFoundError(f"Ticket database not found: {db_path}")

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM ticket_predictions")
    count = cursor.fetchone()[0]
    print(f"Stored prediction rows: {count}")
    conn.close()


# ============================================================
# DAG
# ============================================================

with DAG(
    dag_id="hardtec_complete_pipeline",
    description="End-to-end HARDTEC flow: ticket analysis + RAG + AIOps + storage",
    start_date=datetime(2026, 1, 1),
    schedule="@daily",
    catchup=False,
    tags=["hardtec", "complete", "ticket", "rag", "aiops"],
) as dag:

    validate_data = PythonOperator(
        task_id="validate_ticket_dataset",
        python_callable=validate_ticket_dataset,
    )

    ticket_prediction = PythonOperator(
        task_id="run_ticket_prediction",
        python_callable=run_ticket_prediction,
    )

    rag_lookup = PythonOperator(
        task_id="run_rag_lookup",
        python_callable=run_rag_lookup,
    )

    aiops_risk = PythonOperator(
        task_id="run_aiops_risk_check",
        python_callable=run_aiops_risk_check,
    )

    storage_summary = PythonOperator(
        task_id="summarize_storage",
        python_callable=summarize_storage,
    )

    validate_data >> ticket_prediction >> [rag_lookup, aiops_risk] >> storage_summary
