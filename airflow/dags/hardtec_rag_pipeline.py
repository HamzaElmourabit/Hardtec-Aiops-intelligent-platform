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
        if candidate.exists() and (candidate / "models" / "rag").exists():
            return candidate

    return Path(__file__).resolve().parents[2]

REPO_ROOT = get_repo_root()
RAG_DIR = REPO_ROOT / "models" / "rag"
INDEX_PATH = RAG_DIR / "tickets.index"
DOCUMENTS_PATH = RAG_DIR / "documents.pkl"
# ============================================================
# TASKS
# ============================================================
def check_rag_assets():
    required = [INDEX_PATH, DOCUMENTS_PATH]
    missing = [str(path) for path in required if not path.exists()]

    if missing:
        raise FileNotFoundError(f"Missing RAG assets: {missing}")

    print("RAG assets available:")
    for path in required:
        print(f"- {path}")


def run_rag_similarity_search():
    import pickle
    from pathlib import Path

    import faiss
    from sentence_transformers import SentenceTransformer

    if not INDEX_PATH.exists():
        raise FileNotFoundError(f"FAISS index missing: {INDEX_PATH}")
    if not DOCUMENTS_PATH.exists():
        raise FileNotFoundError(f"Documents file missing: {DOCUMENTS_PATH}")

    embedding_model = SentenceTransformer(
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        device="cpu",
    )
    index = faiss.read_index(str(INDEX_PATH))

    with open(DOCUMENTS_PATH, "rb") as handle:
        documents = pickle.load(handle)

    query = "network outage affecting multiple users"
    query_embedding = embedding_model.encode(
        [query],
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype("float32")

    scores, indices = index.search(query_embedding, 3)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0:
            continue
        results.append({"score": float(score), "document": documents[int(idx)]})

    print("RAG search results:")
    for item in results:
        print(item["score"], "->", item["document"][:160])

    if not results:
        raise ValueError("No RAG results returned for the demo query.")


# ============================================================
# DAG
# ============================================================

with DAG(
    dag_id="hardtec_rag_pipeline",
    description="HARDTEC RAG retrieval pipeline",
    start_date=datetime(2026, 1, 1),
    schedule="@daily",
    catchup=False,
    tags=["hardtec", "rag", "retrieval"],
) as dag:

    validate_assets = PythonOperator(
        task_id="check_rag_assets",
        python_callable=check_rag_assets,
    )

    run_similarity = PythonOperator(
        task_id="run_rag_similarity_search",
        python_callable=run_rag_similarity_search,
    )

    validate_assets >> run_similarity