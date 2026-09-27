from datetime import datetime
from pathlib import Path

from airflow import DAG
from airflow.operators.python import PythonOperator


# ============================================================
# CONFIGURATION
# ============================================================

def get_repo_root() -> Path:
    """Return the project root for local and Docker runs."""
    candidates = [
        Path("/workspace"),
        Path(__file__).resolve().parents[2],
    ]

    for candidate in candidates:
        if candidate.exists() and (candidate / "data" / "processed").exists():
            return candidate

    return Path(__file__).resolve().parents[2]


REPO_ROOT = get_repo_root()
PROCESSED_DIR = REPO_ROOT / "data" / "processed"
ANOMALIES_PATH = PROCESSED_DIR / "aiops_anomalies.csv"
INCIDENTS_PATH = PROCESSED_DIR / "aiops_incidents.csv"
RISK_OUTPUT_PATH = PROCESSED_DIR / "aiops_incident_risk.csv"


# ============================================================
# TASKS
# ============================================================

def check_aiops_inputs():
    required_files = [ANOMALIES_PATH, INCIDENTS_PATH]

    missing = [str(path) for path in required_files if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing AIOps inputs: {missing}")

    print("AIOps inputs found:")
    for path in required_files:
        print(f"- {path}")


def generate_aiops_risk_summary():
    import subprocess

    script_path = REPO_ROOT / "src" / "aiops" / "predict_incident_risk.py"
    if not script_path.exists():
        raise FileNotFoundError(f"Script missing: {script_path}")

    subprocess.run(
        ["python", str(script_path)],
        cwd=str(REPO_ROOT),
        check=True,
    )

    if not RISK_OUTPUT_PATH.exists():
        raise FileNotFoundError(f"AIOps risk output not created: {RISK_OUTPUT_PATH}")

    print(f"AIOps risk summary written to: {RISK_OUTPUT_PATH}")


def summarize_aiops_risk():
    import pandas as pd

    if not RISK_OUTPUT_PATH.exists():
        raise FileNotFoundError(f"Missing output file: {RISK_OUTPUT_PATH}")

    df = pd.read_csv(RISK_OUTPUT_PATH)
    print(f"Rows: {len(df)}")
    print(df.head().to_string(index=False))
    print("AIOps risk summary: OK")


# ============================================================
# DAG
# ============================================================

with DAG(
    dag_id="hardtec_aiops_pipeline",
    description="HARDTEC AIOps incident risk pipeline",
    start_date=datetime(2026, 1, 1),
    schedule="@daily",
    catchup=False,
    tags=["hardtec", "aiops", "incident-risk"],
) as dag:

    check_inputs = PythonOperator(
        task_id="check_aiops_inputs",
        python_callable=check_aiops_inputs,
    )

    generate_risk = PythonOperator(
        task_id="generate_aiops_risk_summary",
        python_callable=generate_aiops_risk_summary,
    )

    summarize = PythonOperator(
        task_id="summarize_aiops_risk",
        python_callable=summarize_aiops_risk,
    )

    check_inputs >> generate_risk >> summarize