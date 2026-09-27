from src.rag import rag_engine


def test_embedding_model_is_not_loaded_until_retrieval():
    assert rag_engine.embedding_model is None
