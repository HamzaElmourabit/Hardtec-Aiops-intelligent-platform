from pathlib import Path


def test_rag_runtime_dependencies_are_declared_for_container_build():
    req_file = Path(__file__).resolve().parents[1] / "requirements.txt"
    requirements = req_file.read_text(encoding="utf-8")

    assert "faiss-cpu" in requirements
    assert "sentence-transformers" in requirements
