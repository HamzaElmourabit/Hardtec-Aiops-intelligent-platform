
"""
HARDTEC - MicroSS Temporal Transformer V4
=========================================

IMPORTANT
---------
This experiment is completely independent from MicroSS XGBoost V3.1.

V3.1 is NOT modified.
No V3.1 model, metadata, feature matrix or prediction code is overwritten.

Purpose
-------
Train a lightweight Temporal Transformer on the current MicroSS dataset.

Current telemetry dataset:
    data/processed/micross/micross_multisource_features_5m.csv

The current telemetry CSV does NOT contain the target column
"incident_next_15m".

Therefore this script builds the target from the available incident/anomaly
event files if they exist.

Target definition:
    incident_next_15m = 1
    when an incident occurs during the next 15 minutes after timestamp t.

Otherwise:
    incident_next_15m = 0

Temporal input:
    Previous SEQUENCE_LENGTH timestamps.

With the default:
    SEQUENCE_LENGTH = 6
    frequency = 5 minutes

the Transformer observes:

    t-25m
    t-20m
    t-15m
    t-10m
    t-5m
    t

and predicts:

    incident during (t, t+15m]

Architecture:
    Input telemetry
        ↓
    Linear projection
        ↓
    Positional embeddings
        ↓
    Transformer Encoder
        ↓
    Last timestep representation
        ↓
    Binary classifier
        ↓
    P(incident_next_15m)

Output:
    models/micross/transformer_v4/
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    average_precision_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "micross"
    / "micross_multisource_features_5m.csv"
)

MICROSS_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "micross"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "models"
    / "micross"
    / "transformer_v4"
)

MODEL_PATH = OUTPUT_DIR / "temporal_transformer_v4.pt"
SCALER_MEAN_PATH = OUTPUT_DIR / "scaler_mean.npy"
SCALER_SCALE_PATH = OUTPUT_DIR / "scaler_scale.npy"
METADATA_PATH = OUTPUT_DIR / "metadata.json"

# ============================================================
# CONFIGURATION
# ============================================================

TARGET_COLUMN = "incident_next_15m"

# MicroSS frequency = 5 minutes.
FREQUENCY_MINUTES = 5

# 6 x 5 minutes = 30 minutes historical context.
SEQUENCE_LENGTH = 6

# Prediction horizon = next 15 minutes.
PREDICTION_HORIZON_STEPS = 3

# Chronological split.
TRAIN_RATIO = 0.70
VALID_RATIO = 0.15
TEST_RATIO = 0.15

# Transformer.
D_MODEL = 64
N_HEADS = 4
N_LAYERS = 2
DIM_FEEDFORWARD = 128
DROPOUT = 0.20

# Training.
BATCH_SIZE = 64
EPOCHS = 30
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
PATIENCE = 6

# Random seed.
RANDOM_SEED = 42

# Training device.
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# DISPLAY
# ============================================================


def print_header(title: str) -> None:
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


# ============================================================
# REPRODUCIBILITY
# ============================================================


def set_seed(seed: int = RANDOM_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Deterministic behavior where possible.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================
# DATASET LOADING
# ============================================================


def load_dataset() -> pd.DataFrame:
    """
    Load the current MicroSS telemetry dataset.

    The target is intentionally NOT required because the current
    telemetry CSV does not contain incident_next_15m.
    """

    print_header("LOADING MICROss DATASET")

    print(f"Dataset: {DATASET_PATH}")

    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"\nMicroSS dataset not found:\n{DATASET_PATH}\n"
        )

    df = pd.read_csv(DATASET_PATH)

    print(f"Rows    : {len(df):,}")
    print(f"Columns : {len(df.columns):,}")

    if "timestamp" not in df.columns:
        raise ValueError(
            "The MicroSS dataset must contain a 'timestamp' column."
        )

    # Parse timestamp.
    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        errors="coerce",
    )

    invalid_timestamps = df["timestamp"].isna().sum()

    if invalid_timestamps > 0:
        print(
            f"WARNING: removing {invalid_timestamps:,} rows "
            "with invalid timestamps."
        )

        df = df.dropna(subset=["timestamp"]).copy()

    # Sort chronologically.
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Remove duplicated timestamps.
    duplicate_timestamps = df["timestamp"].duplicated().sum()

    if duplicate_timestamps > 0:
        print(
            f"WARNING: removing {duplicate_timestamps:,} "
            "duplicated timestamps."
        )

        df = df.drop_duplicates(
            subset=["timestamp"],
            keep="first",
        ).reset_index(drop=True)

    print()
    print("Timestamp range:")
    print(f"  Start : {df['timestamp'].min()}")
    print(f"  End   : {df['timestamp'].max()}")

    print()
    print(
        "Target column present in telemetry CSV:",
        TARGET_COLUMN in df.columns,
    )

    if TARGET_COLUMN in df.columns:
        print(
            "WARNING: target already exists. "
            "It will NOT be overwritten."
        )

    return df


# ============================================================
# INCIDENT FILE DISCOVERY
# ============================================================


def discover_incident_files() -> list[Path]:
    """
    Search for incident/event CSV files in the HARDTEC project.

    We intentionally do NOT modify any existing files.

    Candidate names include:
        aiops_incidents.csv
        aiops_anomalies.csv
        incidents.csv
        incident_events.csv
        micross_incidents.csv
    """

    print_header("SEARCHING FOR INCIDENT / EVENT DATA")

    candidates = [
        MICROSS_DIR / "aiops_incidents.csv",
        MICROSS_DIR / "aiops_anomalies.csv",
        MICROSS_DIR / "incidents.csv",
        MICROSS_DIR / "incident_events.csv",
        MICROSS_DIR / "micross_incidents.csv",
        MICROSS_DIR / "micross_incident_events.csv",

        PROJECT_ROOT / "data" / "processed" / "aiops_incidents.csv",
        PROJECT_ROOT / "data" / "processed" / "aiops_anomalies.csv",

        PROJECT_ROOT / "data" / "aiops_incidents.csv",
        PROJECT_ROOT / "data" / "aiops_anomalies.csv",

        PROJECT_ROOT / "data" / "raw" / "aiops_incidents.csv",
        PROJECT_ROOT / "data" / "raw" / "aiops_anomalies.csv",
    ]

    existing = []

    for path in candidates:
        if path.exists():
            existing.append(path)

    # Also search recursively, but only for likely names.
    if not existing:
        patterns = [
            "*incident*.csv",
            "*anomal*.csv",
        ]

        for pattern in patterns:
            for path in PROJECT_ROOT.rglob(pattern):
                if path.is_file() and path not in existing:
                    existing.append(path)

    if existing:
        print("Candidate event files found:")

        for path in existing:
            print(f"  - {path}")

    else:
        print("No incident/event CSV was found.")

    return existing


# ============================================================
# EVENT TIMESTAMP DETECTION
# ============================================================


def detect_timestamp_column(
    df: pd.DataFrame,
) -> str | None:
    """
    Automatically detect an event timestamp column.
    """

    preferred = [
        "timestamp",
        "incident_timestamp",
        "event_timestamp",
        "datetime",
        "date",
        "time",
        "start_time",
        "incident_time",
        "event_time",
    ]

    for column in preferred:
        if column in df.columns:
            return column

    # Fallback: inspect columns containing time/date.
    for column in df.columns:
        name = column.lower()

        if any(
            keyword in name
            for keyword in [
                "timestamp",
                "datetime",
                "date",
                "time",
            ]
        ):
            return column

    return None


# ============================================================
# INCIDENT LABEL COLUMN DETECTION
# ============================================================


def detect_incident_label_column(
    df: pd.DataFrame,
) -> str | None:
    """
    Detect an optional incident/event label column.

    If the event CSV already contains one row per incident, no label
    column is required.

    If it contains multiple event types, we try to identify a useful
    incident indicator.
    """

    preferred = [
        "incident",
        "is_incident",
        "incident_flag",
        "label",
        "target",
        "event_type",
        "type",
        "status",
    ]

    for column in preferred:
        if column in df.columns:
            return column

    return None


# ============================================================
# BUILD INCIDENT TIMESTAMPS
# ============================================================


def load_incident_timestamps(
    event_files: Iterable[Path],
) -> pd.DatetimeIndex:
    """
    Load incident timestamps from available event files.

    Strategy:
    1. Find timestamp column.
    2. If an incident/event label exists, keep rows that look positive.
    3. Otherwise assume each row represents an event/incident.
    4. Return unique timestamps.
    """

    print_header("BUILDING INCIDENT TIMELINE")

    all_timestamps: list[pd.Timestamp] = []

    for path in event_files:
        print()
        print(f"Reading: {path}")

        try:
            events = pd.read_csv(path)
        except Exception as exc:
            print(f"WARNING: could not read {path}: {exc}")
            continue

        print(
            f"  Rows    : {len(events):,}"
        )
        print(
            f"  Columns : {len(events.columns):,}"
        )

        timestamp_column = detect_timestamp_column(events)

        if timestamp_column is None:
            print(
                "  WARNING: no timestamp column detected. "
                "Skipping file."
            )
            continue

        print(
            f"  Timestamp column: {timestamp_column}"
        )

        events[timestamp_column] = pd.to_datetime(
            events[timestamp_column],
            errors="coerce",
        )

        events = events.dropna(
            subset=[timestamp_column]
        ).copy()

        label_column = detect_incident_label_column(events)

        if label_column is not None:
            print(
                f"  Event/label column: {label_column}"
            )

            values = events[label_column]

            # Normalize strings.
            normalized = (
                values.astype(str)
                .str.strip()
                .str.lower()
            )

            positive_values = {
                "1",
                "true",
                "yes",
                "incident",
                "anomaly",
                "failure",
                "fault",
                "error",
                "critical",
                "high",
                "outage",
            }

            mask = normalized.isin(positive_values)

            # If nothing matches, do NOT blindly discard the data.
            if mask.any():
                events = events.loc[mask].copy()

                print(
                    f"  Positive events kept: {len(events):,}"
                )

            else:
                print(
                    "  No recognized positive label values. "
                    "Assuming each row is an event."
                )

        timestamps = events[timestamp_column]

        all_timestamps.extend(
            timestamps.tolist()
        )

        print(
            f"  Incident/event timestamps added: "
            f"{len(timestamps):,}"
        )

    if not all_timestamps:
        raise FileNotFoundError(
            """
No usable incident/event timestamps were found.

The current MicroSS telemetry file does not contain
'incident_next_15m'.

Therefore V4 needs an event/incident source.

Expected examples:

    data/processed/micross/aiops_incidents.csv
    data/processed/micross/aiops_anomalies.csv

or another CSV containing incident timestamps.

V3.1 has NOT been modified.
"""
        )

    index = pd.DatetimeIndex(
        sorted(
            set(all_timestamps)
        )
    )

    print()
    print(
        f"Unique incident/event timestamps: {len(index):,}"
    )

    print(
        f"First incident: {index.min()}"
    )

    print(
        f"Last incident : {index.max()}"
    )

    return index


# ============================================================
# ALIGN EVENTS TO MICROSS 5-MINUTE GRID
# ============================================================


def build_target(
    df: pd.DataFrame,
    incident_timestamps: pd.DatetimeIndex,
) -> pd.DataFrame:
    """
    Build incident_next_15m.

    For each telemetry timestamp t:

        target = 1

    if an incident exists during:

        (t, t+15 minutes]

    Otherwise:

        target = 0

    IMPORTANT:
    This function only creates a target in memory.
    It does NOT modify the original MicroSS CSV.
    """

    print_header("BUILDING incident_next_15m TARGET")

    result = df.copy()

    telemetry_timestamps = result["timestamp"]

    incident_series = pd.Series(
        1,
        index=pd.DatetimeIndex(incident_timestamps),
        dtype="int8",
    )

    # Remove duplicates.
    incident_series = (
        incident_series[~incident_series.index.duplicated()]
    )

    # For each telemetry timestamp, find the first incident strictly
    # after t.
    incident_values = incident_series.index.values.astype(
        "datetime64[ns]"
    )

    telemetry_values = telemetry_timestamps.values.astype(
        "datetime64[ns]"
    )

    # Search position of first incident >= timestamp.
    positions = np.searchsorted(
        incident_values,
        telemetry_values,
        side="right",
    )

    targets = np.zeros(
        len(result),
        dtype=np.int8,
    )

    horizon_ns = np.timedelta64(
        FREQUENCY_MINUTES
        * PREDICTION_HORIZON_STEPS,
        "m",
    )

    for i, position in enumerate(positions):
        if position >= len(incident_values):
            continue

        next_incident = incident_values[position]

        current_time = telemetry_values[i]

        if (
            next_incident > current_time
            and next_incident <= current_time + horizon_ns
        ):
            targets[i] = 1

    result[TARGET_COLUMN] = targets

    positives = int(targets.sum())
    negatives = int(len(targets) - positives)

    print()
    print("Target definition:")
    print(
        "  1 = incident occurs within the next 15 minutes"
    )
    print(
        "  0 = no incident within the next 15 minutes"
    )

    print()
    print("Target distribution:")
    print(
        f"  0 : {negatives:,}"
    )
    print(
        f"  1 : {positives:,}"
    )

    print()
    print(
        f"Positive ratio: "
        f"{positives / max(len(targets), 1):.4%}"
    )

    if positives == 0:
        raise ValueError(
            """
The generated target contains ZERO positive examples.

Possible causes:
- wrong incident file
- wrong timestamp column
- incident timestamps outside the telemetry period
- incorrect event labels
- timezone mismatch
"""
        )

    return result


# ============================================================
# DETECT TELEMETRY METRICS
# ============================================================


def detect_telemetry_metrics(
    df: pd.DataFrame,
) -> list[str]:
    """
    Detect actual MicroSS telemetry metric columns.

    The current dataset contains many numeric columns, including
    temporal/context columns.

    We deliberately exclude:
        timestamp
        target
        day_of_month
        day_of_week
        hour
        is_weekend

    We also exclude obvious non-metric identifiers.

    This keeps V4 focused on telemetry rather than accidentally
    feeding every numeric metadata column into the Transformer.
    """

    print_header("DETECTING MICROss TELEMETRY FEATURES")

    excluded = {
        "timestamp",
        TARGET_COLUMN,
        "day_of_month",
        "day_of_week",
        "hour",
        "is_weekend",
        "year",
        "month",
        "minute",
    }

    numeric_columns = df.select_dtypes(
        include=[np.number]
    ).columns.tolist()

    metrics = []

    for column in numeric_columns:
        if column in excluded:
            continue

        name = column.lower()

        # Exclude obvious target-like columns.
        if (
            "incident_next" in name
            or name in {
                "incident",
                "label",
                "target",
                "is_incident",
            }
        ):
            continue

        metrics.append(column)

    if not metrics:
        raise ValueError(
            "No telemetry metric columns were detected."
        )

    print(
        f"Numeric columns found : {len(numeric_columns):,}"
    )

    print(
        f"Telemetry metrics used : {len(metrics):,}"
    )

    print()
    print("First 20 telemetry features:")

    for column in metrics[:20]:
        print(f"  - {column}")

    if len(metrics) > 20:
        print(
            f"  ... and {len(metrics) - 20:,} more"
        )

    return metrics


# ============================================================
# PREPARE TELEMETRY DATA
# ============================================================


def prepare_telemetry_data(
    df: pd.DataFrame,
    feature_columns: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert telemetry to float32 and clean NaN/Inf values.

    Returns:
        X : shape (rows, features)
        y : shape (rows,)
    """

    print_header("PREPARING TELEMETRY DATA")

    X_df = df[feature_columns].copy()

    # Convert to numeric.
    for column in feature_columns:
        X_df[column] = pd.to_numeric(
            X_df[column],
            errors="coerce",
        )

    # Replace infinite values.
    X_df = X_df.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    # Forward/backward fill temporal telemetry.
    X_df = X_df.ffill().bfill()

    # Any remaining NaN becomes zero.
    X_df = X_df.fillna(0.0)

    X = X_df.to_numpy(
        dtype=np.float32,
        copy=True,
    )

    y = df[TARGET_COLUMN].to_numpy(
        dtype=np.float32,
    )

    print(
        f"Telemetry matrix : {X.shape}"
    )

    print(
        f"Dtype            : {X.dtype}"
    )

    print(
        f"Target shape     : {y.shape}"
    )

    print(
        f"NaN count        : {np.isnan(X).sum():,}"
    )

    print(
        f"Inf count        : {np.isinf(X).sum():,}"
    )

    return X, y


# ============================================================
# TEMPORAL SPLIT
# ============================================================


def temporal_split(
    X: np.ndarray,
    y: np.ndarray,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """
    Chronological 70/15/15 split.

    No random shuffling across time.
    """

    print_header("CHRONOLOGICAL TRAIN / VALID / TEST SPLIT")

    n = len(X)

    train_end = int(
        n * TRAIN_RATIO
    )

    valid_end = int(
        n * (TRAIN_RATIO + VALID_RATIO)
    )

    X_train = X[:train_end]
    y_train = y[:train_end]

    X_valid = X[train_end:valid_end]
    y_valid = y[train_end:valid_end]

    X_test = X[valid_end:]
    y_test = y[valid_end:]

    print(
        f"Train : {len(X_train):,} "
        f"({len(X_train) / n:.1%})"
    )

    print(
        f"Valid : {len(X_valid):,} "
        f"({len(X_valid) / n:.1%})"
    )

    print(
        f"Test  : {len(X_test):,} "
        f"({len(X_test) / n:.1%})"
    )

    print()
    print("Class distribution:")

    print(
        f"Train positives: "
        f"{int(y_train.sum()):,}"
    )

    print(
        f"Valid positives: "
        f"{int(y_valid.sum()):,}"
    )

    print(
        f"Test positives : "
        f"{int(y_test.sum()):,}"
    )

    return (
        X_train,
        y_train,
        X_valid,
        y_valid,
        X_test,
        y_test,
    )


# ============================================================
# SCALE DATA
# ============================================================


def scale_sequences(
    X_train: np.ndarray,
    X_valid: np.ndarray,
    X_test: np.ndarray,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    StandardScaler,
]:
    """
    Fit scaler ONLY on training data.

    Input shape:
        (rows, features)

    Output:
        scaled arrays
    """

    print_header("STANDARDIZATION")

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(
        X_train
    ).astype(
        np.float32
    )

    X_valid_scaled = scaler.transform(
        X_valid
    ).astype(
        np.float32
    )

    X_test_scaled = scaler.transform(
        X_test
    ).astype(
        np.float32
    )

    print(
        "Scaler fitted on TRAIN data only."
    )

    return (
        X_train_scaled,
        X_valid_scaled,
        X_test_scaled,
        scaler,
    )


# ============================================================
# CREATE TEMPORAL SEQUENCES
# ============================================================


def create_sequences(
    X: np.ndarray,
    y: np.ndarray,
    sequence_length: int = SEQUENCE_LENGTH,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert tabular telemetry into temporal windows.

    Example:

        t0
        t1
        t2
        t3
        t4
        t5

        -> target at t5

    Shape:

        X:
            (samples, sequence_length, features)

        y:
            (samples,)
    """

    if len(X) <= sequence_length:
        raise ValueError(
            "Not enough rows to create temporal sequences."
        )

    samples = len(X) - sequence_length + 1

    n_features = X.shape[1]

    sequences = np.empty(
        (
            samples,
            sequence_length,
            n_features,
        ),
        dtype=np.float32,
    )

    targets = np.empty(
        samples,
        dtype=np.float32,
    )

    for i in range(samples):
        end = i + sequence_length

        sequences[i] = X[i:end]
        targets[i] = y[end - 1]

    return sequences, targets


# ============================================================
# DATA LOADERS
# ============================================================


def create_dataloader(
    X: np.ndarray,
    y: np.ndarray,
    shuffle: bool,
) -> DataLoader:
    dataset = TensorDataset(
        torch.from_numpy(X),
        torch.from_numpy(y),
    )

    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )


# ============================================================
# TRANSFORMER MODEL
# ============================================================


class TemporalTransformer(nn.Module):
    """
    Lightweight Transformer Encoder for temporal telemetry.

    Input:
        [batch, sequence_length, n_features]

    Output:
        [batch]
    """

    def __init__(
        self,
        n_features: int,
        sequence_length: int,
        d_model: int = D_MODEL,
        n_heads: int = N_HEADS,
        n_layers: int = N_LAYERS,
        dim_feedforward: int = DIM_FEEDFORWARD,
        dropout: float = DROPOUT,
    ):
        super().__init__()

        self.n_features = n_features
        self.sequence_length = sequence_length
        self.d_model = d_model

        # Project raw telemetry into Transformer representation.
        self.input_projection = nn.Linear(
            n_features,
            d_model,
        )

        # Learnable positional representation.
        self.position_embedding = nn.Parameter(
            torch.zeros(
                1,
                sequence_length,
                d_model,
            )
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers,
        )

        self.norm = nn.LayerNorm(
            d_model
        )

        self.dropout = nn.Dropout(
            dropout
        )

        self.classifier = nn.Linear(
            d_model,
            1,
        )

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        nn.init.xavier_uniform_(
            self.input_projection.weight
        )

        nn.init.zeros_(
            self.input_projection.bias
        )

        nn.init.normal_(
            self.position_embedding,
            mean=0.0,
            std=0.02,
        )

        nn.init.xavier_uniform_(
            self.classifier.weight
        )

        nn.init.zeros_(
            self.classifier.bias
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        # [B, T, F]
        x = self.input_projection(x)

        # [B, T, D]
        x = x + self.position_embedding

        # Temporal attention.
        x = self.encoder(x)

        # Normalize.
        x = self.norm(x)

        # Use last timestep representation.
        x = x[:, -1, :]

        x = self.dropout(x)

        logits = self.classifier(x)

        return logits.squeeze(-1)


# ============================================================
# POSITIVE CLASS WEIGHT
# ============================================================


def calculate_pos_weight(
    y_train: np.ndarray,
) -> float:
    """
    Compute BCE positive-class weight.
    """

    positives = float(
        np.sum(y_train == 1)
    )

    negatives = float(
        np.sum(y_train == 0)
    )

    if positives <= 0:
        return 1.0

    weight = negatives / positives

    # Avoid extreme numerical instability.
    weight = float(
        np.clip(
            weight,
            1.0,
            50.0,
        )
    )

    print()
    print(
        f"Positive class weight: {weight:.4f}"
    )

    return weight


# ============================================================
# PREDICTION
# ============================================================


@torch.no_grad()
def predict_probabilities(
    model: nn.Module,
    loader: DataLoader,
) -> tuple[np.ndarray, np.ndarray]:

    model.eval()

    probabilities = []
    targets = []

    for X_batch, y_batch in loader:

        X_batch = X_batch.to(
            DEVICE,
            non_blocking=True,
        )

        logits = model(
            X_batch
        )

        probs = torch.sigmoid(
            logits
        )

        probabilities.extend(
            probs.detach()
            .cpu()
            .numpy()
            .tolist()
        )

        targets.extend(
            y_batch.numpy().tolist()
        )

    return (
        np.asarray(
            probabilities,
            dtype=np.float32,
        ),
        np.asarray(
            targets,
            dtype=np.float32,
        ),
    )


# ============================================================
# METRICS
# ============================================================


def calculate_metrics(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    threshold: float = 0.5,
) -> dict:

    predictions = (
        probabilities >= threshold
    ).astype(
        np.int32
    )

    metrics = {}

    # ROC-AUC only if both classes exist.
    if len(np.unique(y_true)) >= 2:
        metrics["roc_auc"] = float(
            roc_auc_score(
                y_true,
                probabilities,
            )
        )

        metrics["pr_auc"] = float(
            average_precision_score(
                y_true,
                probabilities,
            )
        )
    else:
        metrics["roc_auc"] = None
        metrics["pr_auc"] = None

    metrics["accuracy"] = float(
        accuracy_score(
            y_true,
            predictions,
        )
    )

    metrics["precision"] = float(
        precision_score(
            y_true,
            predictions,
            zero_division=0,
        )
    )

    metrics["recall"] = float(
        recall_score(
            y_true,
            predictions,
            zero_division=0,
        )
    )

    metrics["f1"] = float(
        f1_score(
            y_true,
            predictions,
            zero_division=0,
        )
    )

    metrics["confusion_matrix"] = (
        confusion_matrix(
            y_true,
            predictions,
            labels=[0, 1],
        ).tolist()
    )

    return metrics


# ============================================================
# TRAINING
# ============================================================


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    pos_weight: float,
) -> tuple[nn.Module, list[dict]]:

    print_header("TRAINING TEMPORAL TRANSFORMER V4")

    print(
        f"Device         : {DEVICE}"
    )

    print(
        f"Epochs         : {EPOCHS}"
    )

    print(
        f"Batch size     : {BATCH_SIZE}"
    )

    print(
        f"Learning rate  : {LEARNING_RATE}"
    )

    print(
        f"Weight decay   : {WEIGHT_DECAY}"
    )

    print()

    model = model.to(
        DEVICE
    )

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(
            pos_weight,
            dtype=torch.float32,
            device=DEVICE,
        )
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=2,
    )

    best_score = -np.inf
    best_state = None
    patience_counter = 0

    history = []

    for epoch in range(
        1,
        EPOCHS + 1,
    ):

        model.train()

        running_loss = 0.0
        sample_count = 0

        for X_batch, y_batch in train_loader:

            X_batch = X_batch.to(
                DEVICE,
                non_blocking=True,
            )

            y_batch = y_batch.to(
                DEVICE,
                non_blocking=True,
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            logits = model(
                X_batch
            )

            loss = criterion(
                logits,
                y_batch,
            )

            loss.backward()

            # Gradient clipping for stability.
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0,
            )

            optimizer.step()

            batch_size = X_batch.size(0)

            running_loss += (
                loss.item()
                * batch_size
            )

            sample_count += batch_size

        train_loss = (
            running_loss
            / max(sample_count, 1)
        )

        valid_probabilities, valid_targets = (
            predict_probabilities(
                model,
                valid_loader,
            )
        )

        valid_metrics = calculate_metrics(
            valid_targets,
            valid_probabilities,
        )

        roc_auc = valid_metrics[
            "roc_auc"
        ]

        if roc_auc is None:
            monitor_score = (
                valid_metrics["f1"]
            )
        else:
            monitor_score = roc_auc

        scheduler.step(
            monitor_score
        )

        current_lr = optimizer.param_groups[
            0
        ]["lr"]

        epoch_info = {
            "epoch": epoch,
            "train_loss": float(
                train_loss
            ),
            "valid_roc_auc": roc_auc,
            "valid_pr_auc": valid_metrics[
                "pr_auc"
            ],
            "valid_f1": valid_metrics[
                "f1"
            ],
            "valid_precision": valid_metrics[
                "precision"
            ],
            "valid_recall": valid_metrics[
                "recall"
            ],
            "learning_rate": float(
                current_lr
            ),
        }

        history.append(
            epoch_info
        )

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"Loss={train_loss:.5f} | "
            f"Val ROC-AUC="
            f"{roc_auc if roc_auc is not None else float('nan'):.4f} | "
            f"Val PR-AUC="
            f"{valid_metrics['pr_auc'] if valid_metrics['pr_auc'] is not None else float('nan'):.4f} | "
            f"Val F1="
            f"{valid_metrics['f1']:.4f} | "
            f"LR={current_lr:.2e}"
        )

        # Early stopping.
        if monitor_score > best_score:
            best_score = monitor_score

            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }

            patience_counter = 0

        else:
            patience_counter += 1

            if patience_counter >= PATIENCE:
                print()
                print(
                    f"Early stopping after "
                    f"{epoch} epochs."
                )
                break

    if best_state is not None:
        model.load_state_dict(
            best_state
        )

    print()
    print(
        f"Best validation score: "
        f"{best_score:.4f}"
    )

    return model, history


# ============================================================
# SAVE MODEL
# ============================================================


def save_model(
    model: TemporalTransformer,
    scaler: StandardScaler,
    feature_columns: list[str],
    history: list[dict],
    train_metrics: dict,
    valid_metrics: dict,
    test_metrics: dict,
    target_positive_ratio: float,
) -> None:

    print_header("SAVING TRANSFORMER V4")

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Save PyTorch model
    # --------------------------------------------------------

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "model_config": {
            "n_features": len(feature_columns),
            "sequence_length": SEQUENCE_LENGTH,
            "d_model": D_MODEL,
            "n_heads": N_HEADS,
            "n_layers": N_LAYERS,
            "dim_feedforward": DIM_FEEDFORWARD,
            "dropout": DROPOUT,
        },
        "target_column": TARGET_COLUMN,
        "prediction_horizon_steps": (
            PREDICTION_HORIZON_STEPS
        ),
        "frequency_minutes": FREQUENCY_MINUTES,
        "feature_columns": feature_columns,
        "version": "V4",
    }

    torch.save(
        checkpoint,
        MODEL_PATH,
    )

    # --------------------------------------------------------
    # Save scaler
    # --------------------------------------------------------

    np.save(
        SCALER_MEAN_PATH,
        scaler.mean_.astype(
            np.float32
        ),
    )

    np.save(
        SCALER_SCALE_PATH,
        scaler.scale_.astype(
            np.float32
        ),
    )

    # --------------------------------------------------------
    # Metadata
    # --------------------------------------------------------

    metadata = {
        "project": "HARDTEC",
        "model_name": "MicroSS Temporal Transformer",
        "model_version": "V4",

        "v31_protected": True,

        "dataset": str(
            DATASET_PATH
        ),

        "target": TARGET_COLUMN,

        "target_definition": (
            "1 when an incident occurs during "
            "the next 15 minutes; 0 otherwise."
        ),

        "frequency_minutes": FREQUENCY_MINUTES,

        "prediction_horizon_minutes": (
            FREQUENCY_MINUTES
            * PREDICTION_HORIZON_STEPS
        ),

        "sequence_length": SEQUENCE_LENGTH,

        "historical_context_minutes": (
            SEQUENCE_LENGTH
            * FREQUENCY_MINUTES
        ),

        "number_of_features": len(
            feature_columns
        ),

        "feature_columns": feature_columns,

        "architecture": {
            "d_model": D_MODEL,
            "n_heads": N_HEADS,
            "n_layers": N_LAYERS,
            "dim_feedforward": DIM_FEEDFORWARD,
            "dropout": DROPOUT,
        },

        "training": {
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "patience": PATIENCE,
            "random_seed": RANDOM_SEED,
            "device": str(DEVICE),
        },

        "target_positive_ratio": float(
            target_positive_ratio
        ),

        "metrics": {
            "train": train_metrics,
            "validation": valid_metrics,
            "test": test_metrics,
        },

        "history": history,
    }

    with open(
        METADATA_PATH,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            metadata,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print(
        f"Model   : {MODEL_PATH}"
    )

    print(
        f"Scaler  : {SCALER_MEAN_PATH}"
    )

    print(
        f"Metadata: {METADATA_PATH}"
    )


# ============================================================
# EVALUATION
# ============================================================


def print_evaluation(
    name: str,
    metrics: dict,
) -> None:

    print()
    print(
        f"{name.upper()} EVALUATION"
    )

    print(
        "-" * 50
    )

    print(
        f"ROC-AUC   : "
        f"{metrics['roc_auc']}"
    )

    print(
        f"PR-AUC    : "
        f"{metrics['pr_auc']}"
    )

    print(
        f"Accuracy  : "
        f"{metrics['accuracy']:.4f}"
    )

    print(
        f"Precision : "
        f"{metrics['precision']:.4f}"
    )

    print(
        f"Recall    : "
        f"{metrics['recall']:.4f}"
    )

    print(
        f"F1        : "
        f"{metrics['f1']:.4f}"
    )

    print()
    print(
        "Confusion matrix:"
    )

    print(
        np.asarray(
            metrics["confusion_matrix"]
        )
    )


# ============================================================
# MAIN
# ============================================================


def main() -> None:

    set_seed()

    print()
    print(
        "=" * 70
    )
    print(
        "HARDTEC - MICROss TEMPORAL TRANSFORMER V4"
    )
    print(
        "=" * 70
    )

    print()
    print(
        "IMPORTANT:"
    )

    print(
        "This experiment is completely independent from V3.1."
    )

    print(
        "The existing XGBoost V3.1 model will NOT be modified."
    )

    print(
        "=" * 70
    )

    # --------------------------------------------------------
    # 1. Load telemetry
    # --------------------------------------------------------

    df = load_dataset()

    # --------------------------------------------------------
    # 2. Discover incident/event source
    # --------------------------------------------------------

    event_files = discover_incident_files()

    if not event_files:
        raise FileNotFoundError(
            """
No incident/event file was found.

Your current file:

    micross_multisource_features_5m.csv

contains telemetry but does not contain:

    incident_next_15m

Therefore the Transformer cannot be supervised until incident
timestamps are available.

Expected examples:

    data/processed/micross/aiops_incidents.csv
    data/processed/micross/aiops_anomalies.csv

Do NOT modify V3.1.
"""
        )

    # --------------------------------------------------------
    # 3. Load incident timestamps
    # --------------------------------------------------------

    incident_timestamps = load_incident_timestamps(
        event_files
    )

    # --------------------------------------------------------
    # 4. Build target in memory
    # --------------------------------------------------------

    df = build_target(
        df,
        incident_timestamps,
    )

    # --------------------------------------------------------
    # 5. Detect telemetry features
    # --------------------------------------------------------

    feature_columns = detect_telemetry_metrics(
        df
    )

    # --------------------------------------------------------
    # 6. Prepare data
    # --------------------------------------------------------

    X, y = prepare_telemetry_data(
        df,
        feature_columns,
    )

    # --------------------------------------------------------
    # 7. Chronological split
    # --------------------------------------------------------

    (
        X_train,
        y_train,
        X_valid,
        y_valid,
        X_test,
        y_test,
    ) = temporal_split(
        X,
        y,
    )

    # --------------------------------------------------------
    # 8. Scale
    # --------------------------------------------------------

    (
        X_train,
        X_valid,
        X_test,
        scaler,
    ) = scale_sequences(
        X_train,
        X_valid,
        X_test,
    )

    # --------------------------------------------------------
    # 9. Create sequences
    # --------------------------------------------------------

    print_header(
        "CREATING TEMPORAL SEQUENCES"
    )

    print(
        f"Sequence length       : "
        f"{SEQUENCE_LENGTH}"
    )

    print(
        f"Frequency             : "
        f"{FREQUENCY_MINUTES} minutes"
    )

    print(
        f"Historical context    : "
        f"{SEQUENCE_LENGTH * FREQUENCY_MINUTES} minutes"
    )

    print(
        f"Prediction horizon    : "
        f"{PREDICTION_HORIZON_STEPS * FREQUENCY_MINUTES} minutes"
    )

    (
        X_train_seq,
        y_train_seq,
    ) = create_sequences(
        X_train,
        y_train,
    )

    (
        X_valid_seq,
        y_valid_seq,
    ) = create_sequences(
        X_valid,
        y_valid,
    )

    (
        X_test_seq,
        y_test_seq,
    ) = create_sequences(
        X_test,
        y_test,
    )

    print()
    print(
        f"Train sequences : "
        f"{X_train_seq.shape}"
    )

    print(
        f"Valid sequences : "
        f"{X_valid_seq.shape}"
    )

    print(
        f"Test sequences  : "
        f"{X_test_seq.shape}"
    )

    # --------------------------------------------------------
    # 10. DataLoaders
    # --------------------------------------------------------

    train_loader = create_dataloader(
        X_train_seq,
        y_train_seq,
        shuffle=True,
    )

    valid_loader = create_dataloader(
        X_valid_seq,
        y_valid_seq,
        shuffle=False,
    )

    test_loader = create_dataloader(
        X_test_seq,
        y_test_seq,
        shuffle=False,
    )

    # --------------------------------------------------------
    # 11. Model
    # --------------------------------------------------------

    print_header(
        "INITIALIZING TEMPORAL TRANSFORMER V4"
    )

    n_features = len(
        feature_columns
    )

    model = TemporalTransformer(
        n_features=n_features,
        sequence_length=SEQUENCE_LENGTH,
        d_model=D_MODEL,
        n_heads=N_HEADS,
        n_layers=N_LAYERS,
        dim_feedforward=DIM_FEEDFORWARD,
        dropout=DROPOUT,
    )

    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    print(
        f"Telemetry features : {n_features:,}"
    )

    print(
        f"d_model            : {D_MODEL}"
    )

    print(
        f"attention heads     : {N_HEADS}"
    )

    print(
        f"encoder layers      : {N_LAYERS}"
    )

    print(
        f"parameters          : {parameter_count:,}"
    )

    # --------------------------------------------------------
    # 12. Class weighting
    # --------------------------------------------------------

    pos_weight = calculate_pos_weight(
        y_train_seq
    )

    # --------------------------------------------------------
    # 13. Train
    # --------------------------------------------------------

    model, history = train_model(
        model,
        train_loader,
        valid_loader,
        pos_weight,
    )

    # --------------------------------------------------------
    # 14. Final evaluation
    # --------------------------------------------------------

    print_header(
        "FINAL TEST EVALUATION"
    )

    train_probabilities, train_targets = (
        predict_probabilities(
            model,
            create_dataloader(
                X_train_seq,
                y_train_seq,
                shuffle=False,
            ),
        )
    )

    valid_probabilities, valid_targets = (
        predict_probabilities(
            model,
            valid_loader,
        )
    )

    test_probabilities, test_targets = (
        predict_probabilities(
            model,
            test_loader,
        )
    )

    train_metrics = calculate_metrics(
        train_targets,
        train_probabilities,
    )

    valid_metrics = calculate_metrics(
        valid_targets,
        valid_probabilities,
    )

    test_metrics = calculate_metrics(
        test_targets,
        test_probabilities,
    )

    print_evaluation(
        "Train",
        train_metrics,
    )

    print_evaluation(
        "Validation",
        valid_metrics,
    )

    print_evaluation(
        "Test",
        test_metrics,
    )

    # --------------------------------------------------------
    # 15. Save
    # --------------------------------------------------------

    target_positive_ratio = float(
        np.mean(y)
    )

    save_model(
        model=model,
        scaler=scaler,
        feature_columns=feature_columns,
        history=history,
        train_metrics=train_metrics,
        valid_metrics=valid_metrics,
        test_metrics=test_metrics,
        target_positive_ratio=target_positive_ratio,
    )

    # --------------------------------------------------------
    # 16. Final summary
    # --------------------------------------------------------

    print()
    print(
        "=" * 70
    )
    print(
        "MICROss TEMPORAL TRANSFORMER V4 COMPLETE"
    )
    print(
        "=" * 70
    )

    print()
    print(
        "V3.1 STATUS:"
    )

    print(
        "  PROTECTED / UNCHANGED"
    )

    print()
    print(
        "V4 MODEL:"
    )

    print(
        f"  Features       : {n_features:,}"
    )

    print(
        f"  Sequence       : {SEQUENCE_LENGTH} x "
        f"{FREQUENCY_MINUTES} min"
    )

    print(
        f"  Horizon        : "
        f"{PREDICTION_HORIZON_STEPS * FREQUENCY_MINUTES} min"
    )

    print(
        f"  Test ROC-AUC   : "
        f"{test_metrics['roc_auc']}"
    )

    print(
        f"  Test PR-AUC    : "
        f"{test_metrics['pr_auc']}"
    )

    print(
        f"  Test F1        : "
        f"{test_metrics['f1']:.4f}"
    )

    print()
    print(
        f"Output directory:"
    )

    print(
        f"  {OUTPUT_DIR}"
    )

    print()
    print(
        "V3.1 was not touched."
    )

    print(
        "=" * 70
    )


# ============================================================
# ENTRY POINT
# ============================================================


if __name__ == "__main__":
    main()

