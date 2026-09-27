"""Shared MLflow tracking helpers for HARDTEC training scripts."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import mlflow
import pandas as pd
from sklearn.metrics import classification_report


def start_training_run(model_name: str, data_path: Path | str) -> mlflow.ActiveRun:
    """Start a run and record reproducibility metadata."""

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
    experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME", "hardtec-ticketing")

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    run = mlflow.start_run(run_name=model_name)
    dataset_path = Path(data_path)

    mlflow.set_tags(
        {
            "model_name": model_name,
            "training_script": os.path.basename(os.getenv("PYTHON_SCRIPT", "unknown")),
        }
    )
    mlflow.log_param("dataset", str(dataset_path))

    if dataset_path.exists():
        mlflow.log_param("dataset_rows", len(pd.read_csv(dataset_path)))
        mlflow.log_param("dataset_sha256", _sha256(dataset_path))

    return run


def log_classification_result(
    y_true: Any,
    y_pred: Any,
    model_path: Path | str | None = None,
) -> dict[str, float]:
    """Log standard classification metrics and model artifacts."""

    report = classification_report(
        y_true,
        y_pred,
        zero_division=0,
        output_dict=True,
    )

    metrics = {
        "accuracy": float(report["accuracy"]),
        "macro_f1": float(report["macro avg"]["f1-score"]),
        "weighted_f1": float(report["weighted avg"]["f1-score"]),
        "macro_precision": float(report["macro avg"]["precision"]),
        "macro_recall": float(report["macro avg"]["recall"]),
    }
    mlflow.log_metrics(metrics)

    report_path = Path("reports") / "mlflow" / "classification_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    mlflow.log_artifact(str(report_path), artifact_path="evaluation")

    if model_path is not None and Path(model_path).exists():
        mlflow.log_artifact(str(model_path), artifact_path="models")

    return metrics


def end_training_run() -> None:
    """Close the active run without failing the training script."""

    if mlflow.active_run() is not None:
        mlflow.end_run()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()