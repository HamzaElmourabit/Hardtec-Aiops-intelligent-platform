
"""
HARDTEC - Experimental Queue Classification
Sentence Transformer Embeddings + XGBoost

IMPORTANT:
- Experimental model only
- Does NOT modify the production Queue V4 model
- Production model:
    models/ticket_queue_model_v4.pkl

Experimental model:
    models/experimental/ticket_queue_embedding_xgb.pkl
"""

from pathlib import Path
import time
import pickle

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

from sentence_transformers import SentenceTransformer
from xgboost import XGBClassifier


# ============================================================
# CONFIGURATION
# ============================================================

DATA_PATH = Path("data/processed/tickets_clean.csv")

MODEL_PATH = Path(
    "models/experimental/ticket_queue_embedding_xgb.pkl"
)

REPORT_DIR = Path("reports/queue_embeddings")

RANDOM_STATE = 42

TEST_SIZE = 0.20

EMBEDDING_MODEL = "all-MiniLM-L6-v2"

BATCH_SIZE = 32


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("QUEUE CLASSIFICATION")
    print("SENTENCE EMBEDDINGS + XGBOOST")
    print("=" * 70)

    # --------------------------------------------------------
    # Create directories
    # --------------------------------------------------------

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    # ========================================================
    # [1/7] LOAD DATASET
    # ========================================================

    print("\n[1/7] Loading dataset...")

    df = pd.read_csv(DATA_PATH)

    # Keep only required columns
    df = df[["ticket_text", "queue"]].copy()

    # Remove missing values
    df["ticket_text"] = df["ticket_text"].fillna("").astype(str)
    df["queue"] = df["queue"].fillna("").astype(str)

    # Remove empty tickets / labels
    df = df[
        (df["ticket_text"].str.strip() != "")
        & (df["queue"].str.strip() != "")
    ].reset_index(drop=True)

    print(f"Dataset shape: {df.shape}")

    print(f"Number of queues: {df['queue'].nunique()}")

    print("\nQueue distribution:")
    print(df["queue"].value_counts())

    # ========================================================
    # [2/7] ENCODE QUEUE LABELS
    # ========================================================

    print("\n[2/7] Encoding queue labels...")

    label_encoder = LabelEncoder()

    y = label_encoder.fit_transform(
        df["queue"].to_numpy()
    )

    print(f"Classes: {list(label_encoder.classes_)}")

    print(
        f"Number of classes: "
        f"{len(label_encoder.classes_)}"
    )

    # --------------------------------------------------------
    # IMPORTANT FIX
    # Convert ticket_text to NumPy array.
    #
    # This prevents the PyArrow / pandas indexing error:
    #
    # TypeError:
    # only integer scalar arrays can be converted to a scalar index
    # --------------------------------------------------------

    X_text = df["ticket_text"].to_numpy(dtype=str)

    # ========================================================
    # [3/7] TRAIN / TEST SPLIT
    # ========================================================

    print("\n[3/7] Creating train/test split...")

    X_train_text, X_test_text, y_train, y_test = train_test_split(
        X_text,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    print(f"Train samples: {len(X_train_text)}")
    print(f"Test samples : {len(X_test_text)}")

    # ========================================================
    # [4/7] SENTENCE TRANSFORMER
    # ========================================================

    print("\n[4/7] Loading Sentence Transformer...")

    print(
        f"Model: {EMBEDDING_MODEL}"
    )

    embedding_model = SentenceTransformer(
        EMBEDDING_MODEL
    )

    # --------------------------------------------------------
    # Encode training data
    # --------------------------------------------------------

    print("\nEncoding training tickets...")

    start_embedding = time.time()

    X_train_embeddings = embedding_model.encode(
        X_train_text.tolist(),
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )

    train_embedding_time = time.time() - start_embedding

    print(
        f"Training embeddings shape: "
        f"{X_train_embeddings.shape}"
    )

    print(
        f"Training embedding time: "
        f"{train_embedding_time:.2f} sec"
    )

    # --------------------------------------------------------
    # Encode test data
    # --------------------------------------------------------

    print("\nEncoding test tickets...")

    start_embedding = time.time()

    X_test_embeddings = embedding_model.encode(
        X_test_text.tolist(),
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )

    test_embedding_time = time.time() - start_embedding

    print(
        f"Test embeddings shape: "
        f"{X_test_embeddings.shape}"
    )

    print(
        f"Test embedding time: "
        f"{test_embedding_time:.2f} sec"
    )

    # --------------------------------------------------------
    # Ensure float32 for XGBoost
    # --------------------------------------------------------

    X_train_embeddings = np.asarray(
        X_train_embeddings,
        dtype=np.float32,
    )

    X_test_embeddings = np.asarray(
        X_test_embeddings,
        dtype=np.float32,
    )

    print(
        f"\nEmbedding dimension: "
        f"{X_train_embeddings.shape[1]}"
    )

    # ========================================================
    # [5/7] XGBOOST TRAINING
    # ========================================================

    print("\n[5/7] Training XGBoost...")

    print("\nXGBoost configuration:")

    xgb_model = XGBClassifier(
        n_estimators=300,
        max_depth=8,
        learning_rate=0.10,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="multi:softprob",
        num_class=len(label_encoder.classes_),
        eval_metric="mlogloss",
        tree_method="hist",
        n_jobs=4,
        random_state=RANDOM_STATE,
        verbosity=1,
    )

    start_training = time.time()

    xgb_model.fit(
        X_train_embeddings,
        y_train,
    )

    training_time = time.time() - start_training

    print(
        f"\nXGBoost training time: "
        f"{training_time:.2f} sec"
    )

    # ========================================================
    # [6/7] EVALUATION
    # ========================================================

    print("\n[6/7] Evaluating model...")

    start_prediction = time.time()

    y_pred = xgb_model.predict(
        X_test_embeddings
    )

    prediction_time = time.time() - start_prediction

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    accuracy = accuracy_score(
        y_test,
        y_pred,
    )

    precision_macro = precision_score(
        y_test,
        y_pred,
        average="macro",
        zero_division=0,
    )

    recall_macro = recall_score(
        y_test,
        y_pred,
        average="macro",
        zero_division=0,
    )

    f1_macro = f1_score(
        y_test,
        y_pred,
        average="macro",
        zero_division=0,
    )

    f1_weighted = f1_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0,
    )

    errors = int(
        np.sum(y_test != y_pred)
    )

    # ========================================================
    # DISPLAY RESULTS
    # ========================================================

    print("\n" + "=" * 70)
    print("XGBOOST RESULTS")
    print("=" * 70)

    print(
        f"Accuracy        : {accuracy:.4f}"
    )

    print(
        f"Precision Macro : {precision_macro:.4f}"
    )

    print(
        f"Recall Macro    : {recall_macro:.4f}"
    )

    print(
        f"Macro F1        : {f1_macro:.4f}"
    )

    print(
        f"Weighted F1     : {f1_weighted:.4f}"
    )

    print(
        f"Errors          : {errors}"
    )

    print(
        f"Training time   : {training_time:.2f} sec"
    )

    print(
        f"Prediction time : {prediction_time:.4f} sec"
    )

    # ========================================================
    # CLASSIFICATION REPORT
    # ========================================================

    report_dict = classification_report(
        y_test,
        y_pred,
        target_names=label_encoder.classes_,
        output_dict=True,
        zero_division=0,
    )

    classification_report_df = (
        pd.DataFrame(report_dict)
        .transpose()
    )

    classification_report_path = (
        REPORT_DIR
        / "embedding_xgb_classification_report.csv"
    )

    classification_report_df.to_csv(
        classification_report_path
    )

    # ========================================================
    # CONFUSION MATRIX
    # ========================================================

    cm = confusion_matrix(
        y_test,
        y_pred,
    )

    confusion_matrix_df = pd.DataFrame(
        cm,
        index=label_encoder.classes_,
        columns=label_encoder.classes_,
    )

    confusion_matrix_path = (
        REPORT_DIR
        / "embedding_xgb_confusion_matrix.csv"
    )

    confusion_matrix_df.to_csv(
        confusion_matrix_path
    )

    # ========================================================
    # RESULTS CSV
    # ========================================================

    results = pd.DataFrame(
        [
            {
                "Model": "SentenceTransformer + XGBoost",
                "Embedding Model": EMBEDDING_MODEL,
                "Embedding Dimension": X_train_embeddings.shape[1],
                "Accuracy": accuracy,
                "Precision Macro": precision_macro,
                "Recall Macro": recall_macro,
                "Macro F1": f1_macro,
                "Weighted F1": f1_weighted,
                "Errors": errors,
                "Training Time (sec)": training_time,
                "Prediction Time (sec)": prediction_time,
                "Train Samples": len(X_train_text),
                "Test Samples": len(X_test_text),
                "Number of Classes": len(
                    label_encoder.classes_
                ),
            }
        ]
    )

    results_path = (
        REPORT_DIR
        / "embedding_xgb_results.csv"
    )

    results.to_csv(
        results_path,
        index=False,
    )

    # ========================================================
    # [7/7] SAVE EXPERIMENTAL MODEL
    # ========================================================

    print("\n[7/7] Saving experimental model...")

    model_data = {
        "model": xgb_model,
        "label_encoder": label_encoder,
        "embedding_model": EMBEDDING_MODEL,
        "embedding_dimension": X_train_embeddings.shape[1],
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
    }

    with open(
        MODEL_PATH,
        "wb",
    ) as f:

        pickle.dump(
            model_data,
            f,
        )

    print(
        f"Experimental model saved to:\n"
        f"{MODEL_PATH}"
    )

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)

    print(
        f"\nModel              : "
        f"Sentence Transformer + XGBoost"
    )

    print(
        f"Embedding model    : "
        f"{EMBEDDING_MODEL}"
    )

    print(
        f"Embedding dimension: "
        f"{X_train_embeddings.shape[1]}"
    )

    print(
        f"Queues             : "
        f"{len(label_encoder.classes_)}"
    )

    print(
        f"Accuracy            : "
        f"{accuracy:.4f}"
    )

    print(
        f"Macro F1            : "
        f"{f1_macro:.4f}"
    )

    print(
        f"Weighted F1         : "
        f"{f1_weighted:.4f}"
    )

    print(
        f"\nReports saved in:"
        f"\n{REPORT_DIR}"
    )

    print("\n" + "=" * 70)
    print("EXPERIMENT COMPLETED")
    print("=" * 70)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()

