from pathlib import Path
import pickle

import faiss


# ============================================================
# PATHS
# ============================================================

ROOT_DIR = Path(__file__).resolve().parents[2]

RAG_DIR = ROOT_DIR / "models" / "rag"

INDEX_PATH = RAG_DIR / "tickets.index"
DOCUMENTS_PATH = RAG_DIR / "documents.pkl"


# ============================================================
# CONFIGURATION
# ============================================================

EMBEDDING_MODEL = (
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)


# ============================================================
# LOAD MODEL
# ============================================================

embedding_model = None


def _get_embedding_model():
    global embedding_model

    if embedding_model is None:
        from sentence_transformers import SentenceTransformer

        embedding_model = SentenceTransformer(
            EMBEDDING_MODEL,
            device="cpu"
        )

    return embedding_model


# ============================================================
# LOAD FAISS INDEX
# ============================================================

if not INDEX_PATH.exists():
    raise FileNotFoundError(
        f"FAISS index not found: {INDEX_PATH}"
    )

index = faiss.read_index(str(INDEX_PATH))


# ============================================================
# LOAD DOCUMENTS
# ============================================================

if not DOCUMENTS_PATH.exists():
    raise FileNotFoundError(
        f"Documents file not found: {DOCUMENTS_PATH}"
    )

with open(DOCUMENTS_PATH, "rb") as f:
    documents = pickle.load(f)


# ============================================================
# RETRIEVAL
# ============================================================

def retrieve_similar_tickets(
    query: str,
    top_k: int = 5
):
    """
    Retrieve the most similar historical tickets.
    """

    if not query or not query.strip():
        return []

    query_embedding = _get_embedding_model().encode(
        [query],
        normalize_embeddings=True,
        convert_to_numpy=True
    ).astype("float32")

    scores, indices = index.search(
        query_embedding,
        top_k
    )

    results = []

    for score, idx in zip(scores[0], indices[0]):

        if idx < 0:
            continue

        results.append(
            {
                "score": float(score),
                "document": documents[int(idx)],
                "index": int(idx)
            }
        )

    return results