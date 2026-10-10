import pytest

from src.rag import rag_engine


def test_embedding_model_is_not_loaded_until_retrieval():
    assert rag_engine.embedding_model is None
    assert rag_engine.index is None
    assert rag_engine.documents is None


def test_missing_rag_assets_fail_only_when_retrieval_is_requested(tmp_path, monkeypatch):
    monkeypatch.setattr(rag_engine, "INDEX_PATH", tmp_path / "missing.index")

    with pytest.raises(FileNotFoundError, match="FAISS index not found"):
        rag_engine.retrieve_similar_tickets("server unavailable")

    assert rag_engine.embedding_model is None
