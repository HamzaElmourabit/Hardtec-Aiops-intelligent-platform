from datetime import datetime
from pathlib import Path

from airflow import DAG
from airflow.operators.python import PythonOperator


# ============================================================
# CONFIGURATION
# ============================================================

def get_repo_root() -> Path:
    """Return the project root regardless of local or container execution."""
    candidate_paths = [
        Path("/workspace"),
        Path("/opt/airflow"),
        Path(__file__).resolve().parents[2],
    ]

    for candidate in candidate_paths:
        if candidate.exists() and (candidate / "data" / "processed").exists():
            return candidate

    return Path(__file__).resolve().parents[2]


REPO_ROOT = get_repo_root()
DATASET_PATH = REPO_ROOT / "data" / "processed" / "tickets_clean.csv"


# ============================================================
# TASK 1 — DATA QUALITY
# ============================================================

def check_ticket_dataset():
    import pandas as pd

    print("=" * 70)
    print("HARDTEC - TICKET DATA QUALITY CHECK")
    print("=" * 70)

    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Dataset introuvable : {DATASET_PATH}")

    df = pd.read_csv(DATASET_PATH)

    print(f"Dataset : {DATASET_PATH}")
    print(f"Rows    : {len(df)}")
    print(f"Columns : {len(df.columns)}")

    required_columns = [
        "ticket_text",
        "type",
        "queue",
        "priority",
        "language",
    ]

    missing_columns = [col for col in required_columns if col not in df.columns]

    if missing_columns:
        raise ValueError(f"Colonnes manquantes : {missing_columns}")

    print("\nRequired columns: OK")

    missing_values = df[required_columns].isnull().sum()
    print("\nMissing values:")
    print(missing_values)

    if missing_values.sum() > 0:
        raise ValueError("Des valeurs manquantes existent dans les colonnes ML.")

    duplicate_texts = df["ticket_text"].duplicated().sum()
    print(f"\nDuplicate ticket texts: {duplicate_texts}")

    print("\nDATA QUALITY: PASSED")
    print("=" * 70)


# ============================================================
# TASK 2 — DUCKDB ANALYTICS
# ============================================================

def run_duckdb_analytics():
    import duckdb

    print("=" * 70)
    print("HARDTEC - DUCKDB ANALYTICS")
    print("=" * 70)

    conn = duckdb.connect()

    try:
        query = f"""
            SELECT
                type AS ticket_type,
                priority,
                queue,
                COUNT(*) AS total_tickets
            FROM read_csv_auto('{DATASET_PATH.as_posix()}')
            GROUP BY
                type,
                priority,
                queue
            ORDER BY
                total_tickets DESC
        """

        result = conn.execute(query).fetchdf()
        print("\nTicket analytics:")
        print(result.to_string(index=False))
        print("\nDUCKDB ANALYTICS: PASSED")

    finally:
        conn.close()

    print("=" * 70)


# ============================================================
# TASK 3 — PIPELINE READY
# ============================================================

def pipeline_ready():
    print("=" * 70)
    print("HARDTEC TICKET PIPELINE")
    print("PIPELINE READY")
    print("=" * 70)

    print("Data quality       : OK")
    print("DuckDB analytics   : OK")
    print("Status             : READY")


# ============================================================
# DAG
# ============================================================

with DAG(
    dag_id="hardtec_ticket_pipeline",
    description="HARDTEC Ticket DataOps pipeline",
    start_date=datetime(2026, 1, 1),
    schedule="@daily",
    catchup=False,
    tags=["hardtec", "dataops", "tickets", "duckdb"],
) as dag:

    check_dataset = PythonOperator(
        task_id="check_ticket_dataset",
        python_callable=check_ticket_dataset,
    )

    duckdb_analytics = PythonOperator(
        task_id="run_duckdb_analytics",
        python_callable=run_duckdb_analytics,
    )

    ready = PythonOperator(
        task_id="pipeline_ready",
        python_callable=pipeline_ready,
    )

    check_dataset >> duckdb_analytics >> ready