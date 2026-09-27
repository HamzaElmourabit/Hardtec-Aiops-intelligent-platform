import os
import time
import json
import random

import numpy as np
import pandas as pd
import torch

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

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)


# ============================================================
# CONFIGURATION
# ============================================================

DATA_PATH = "data/processed/tickets_clean.csv"

MODEL_NAME = "distilbert-base-uncased"

MODEL_OUTPUT_DIR = "models/experimental/queue_distilbert"
REPORT_DIR = "reports/queue_distilbert"

RANDOM_STATE = 42

TEST_SIZE = 0.20

# CPU-friendly configuration
NUM_EPOCHS = 1
BATCH_SIZE = 8
LEARNING_RATE = 5e-5
MAX_LENGTH = 128

# Set to None to use the complete dataset.
# For a QUICK TEST on CPU, you can temporarily use 5000.
MAX_SAMPLES = None


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(RANDOM_STATE)


# ============================================================
# DIRECTORIES
# ============================================================

os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("QUEUE CLASSIFICATION")
print("DEEP LEARNING - DISTILBERT")
print("=" * 70)


# ============================================================
# DEVICE
# ============================================================

device = "cuda" if torch.cuda.is_available() else "cpu"

print()
print("Device:", device)

if device == "cpu":
    print("WARNING: CPU training can be slow.")
else:
    print("GPU detected.")


# ============================================================
# [1/8] LOAD DATASET
# ============================================================

print()
print("[1/8] Loading dataset...")

df = pd.read_csv(DATA_PATH)

required_columns = ["ticket_text", "queue"]

for column in required_columns:
    if column not in df.columns:
        raise ValueError(
            f"Missing required column '{column}' in {DATA_PATH}"
        )

df = df[required_columns].copy()

df["ticket_text"] = df["ticket_text"].fillna("").astype(str)
df["queue"] = df["queue"].fillna("").astype(str)

# Remove empty tickets / labels
df = df[
    (df["ticket_text"].str.strip() != "")
    & (df["queue"].str.strip() != "")
].copy()

# Optional quick test
if MAX_SAMPLES is not None:
    print(f"Using maximum {MAX_SAMPLES} samples for quick test.")

    # Stratified sample
    if MAX_SAMPLES < len(df):
        df, _ = train_test_split(
            df,
            train_size=MAX_SAMPLES,
            random_state=RANDOM_STATE,
            stratify=df["queue"],
        )

df = df.reset_index(drop=True)

print("Dataset shape:", df.shape)
print("Number of queues:", df["queue"].nunique())

print()
print("Queue distribution:")
print(df["queue"].value_counts())


# ============================================================
# [2/8] ENCODE LABELS
# ============================================================

print()
print("[2/8] Encoding queue labels...")

label_encoder = LabelEncoder()

df["label"] = label_encoder.fit_transform(df["queue"])

num_labels = len(label_encoder.classes_)

print("Classes:", list(label_encoder.classes_))
print("Number of classes:", num_labels)


# ============================================================
# [3/8] TRAIN / TEST SPLIT
# ============================================================

print()
print("[3/8] Creating train/test split...")

train_df, test_df = train_test_split(
    df,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=df["label"],
)

train_df = train_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)

print("Train samples:", len(train_df))
print("Test samples :", len(test_df))


# ============================================================
# [4/8] TOKENIZER
# ============================================================

print()
print("[4/8] Loading DistilBERT tokenizer...")
print("Model:", MODEL_NAME)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


# ============================================================
# DATASET CREATION
# ============================================================

train_dataset = Dataset.from_pandas(
    train_df[["ticket_text", "label"]],
    preserve_index=False,
)

test_dataset = Dataset.from_pandas(
    test_df[["ticket_text", "label"]],
    preserve_index=False,
)


# ============================================================
# TOKENIZATION
# ============================================================

print()
print("Tokenizing training tickets...")

def tokenize_function(examples):
    return tokenizer(
        examples["ticket_text"],
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH,
    )


train_dataset = train_dataset.map(
    tokenize_function,
    batched=True,
    desc="Tokenizing train",
)

print()
print("Tokenizing test tickets...")

test_dataset = test_dataset.map(
    tokenize_function,
    batched=True,
    desc="Tokenizing test",
)


# Keep only required columns
train_dataset = train_dataset.remove_columns(["ticket_text"])
test_dataset = test_dataset.remove_columns(["ticket_text"])

train_dataset.set_format("torch")
test_dataset.set_format("torch")


# ============================================================
# [5/8] LOAD DISTILBERT
# ============================================================

print()
print("[5/8] Loading DistilBERT model...")

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=num_labels,
    id2label={
        i: label
        for i, label in enumerate(label_encoder.classes_)
    },
    label2id={
        label: i
        for i, label in enumerate(label_encoder.classes_)
    },
)

print("Model loaded successfully.")


# ============================================================
# METRICS
# ============================================================

def compute_metrics(eval_pred):

    predictions, labels = eval_pred

    predicted_labels = np.argmax(predictions, axis=1)

    accuracy = accuracy_score(
        labels,
        predicted_labels,
    )

    precision = precision_score(
        labels,
        predicted_labels,
        average="macro",
        zero_division=0,
    )

    recall = recall_score(
        labels,
        predicted_labels,
        average="macro",
        zero_division=0,
    )

    macro_f1 = f1_score(
        labels,
        predicted_labels,
        average="macro",
        zero_division=0,
    )

    weighted_f1 = f1_score(
        labels,
        predicted_labels,
        average="weighted",
        zero_division=0,
    )

    return {
        "accuracy": accuracy,
        "precision_macro": precision,
        "recall_macro": recall,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
    }


# ============================================================
# [6/8] TRAINING
# ============================================================

print()
print("[6/8] Training DistilBERT...")
print()

print("Training configuration:")
print("Epochs       :", NUM_EPOCHS)
print("Batch size   :", BATCH_SIZE)
print("Learning rate:", LEARNING_RATE)
print("Max length   :", MAX_LENGTH)
print("Device       :", device)

training_args = TrainingArguments(
    output_dir=MODEL_OUTPUT_DIR,

    num_train_epochs=NUM_EPOCHS,

    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,

    learning_rate=LEARNING_RATE,

    weight_decay=0.01,

    logging_steps=100,

    eval_strategy="epoch",
    save_strategy="epoch",

    load_best_model_at_end=False,

    report_to="none",

    fp16=False,

    dataloader_num_workers=0,

    save_total_limit=1,

    seed=RANDOM_STATE,
)


trainer = Trainer(
    model=model,
    args=training_args,

    train_dataset=train_dataset,
    eval_dataset=test_dataset,

    compute_metrics=compute_metrics,
)


training_start = time.time()

trainer.train()

training_time = time.time() - training_start

print()
print(f"Training time: {training_time:.2f} sec")


# ============================================================
# [7/8] EVALUATION
# ============================================================

print()
print("[7/8] Evaluating DistilBERT...")

prediction_start = time.time()

predictions = trainer.predict(test_dataset)

prediction_time = time.time() - prediction_start

logits = predictions.predictions
true_labels = predictions.label_ids

predicted_labels = np.argmax(
    logits,
    axis=1,
)


# ============================================================
# METRICS
# ============================================================

accuracy = accuracy_score(
    true_labels,
    predicted_labels,
)

precision_macro = precision_score(
    true_labels,
    predicted_labels,
    average="macro",
    zero_division=0,
)

recall_macro = recall_score(
    true_labels,
    predicted_labels,
    average="macro",
    zero_division=0,
)

macro_f1 = f1_score(
    true_labels,
    predicted_labels,
    average="macro",
    zero_division=0,
)

weighted_f1 = f1_score(
    true_labels,
    predicted_labels,
    average="weighted",
    zero_division=0,
)

errors = int(
    np.sum(
        true_labels != predicted_labels
    )
)


# ============================================================
# RESULTS
# ============================================================

print()
print("=" * 70)
print("DISTILBERT RESULTS")
print("=" * 70)

print(f"Accuracy        : {accuracy:.4f}")
print(f"Precision Macro : {precision_macro:.4f}")
print(f"Recall Macro    : {recall_macro:.4f}")
print(f"Macro F1        : {macro_f1:.4f}")
print(f"Weighted F1     : {weighted_f1:.4f}")
print(f"Errors          : {errors}")
print(f"Training time   : {training_time:.2f} sec")
print(f"Prediction time : {prediction_time:.4f} sec")


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

classification_report_dict = classification_report(
    true_labels,
    predicted_labels,
    target_names=label_encoder.classes_,
    output_dict=True,
    zero_division=0,
)

classification_report_df = pd.DataFrame(
    classification_report_dict
).transpose()

classification_report_path = os.path.join(
    REPORT_DIR,
    "distilbert_classification_report.csv",
)

classification_report_df.to_csv(
    classification_report_path
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    true_labels,
    predicted_labels,
)

confusion_matrix_df = pd.DataFrame(
    cm,
    index=label_encoder.classes_,
    columns=label_encoder.classes_,
)

confusion_matrix_path = os.path.join(
    REPORT_DIR,
    "distilbert_confusion_matrix.csv",
)

confusion_matrix_df.to_csv(
    confusion_matrix_path
)


# ============================================================
# RESULTS CSV
# ============================================================

results = pd.DataFrame(
    [
        {
            "Model": "DistilBERT",
            "Base Model": MODEL_NAME,
            "Accuracy": accuracy,
            "Precision Macro": precision_macro,
            "Recall Macro": recall_macro,
            "Macro F1": macro_f1,
            "Weighted F1": weighted_f1,
            "Errors": errors,
            "Training Time (sec)": training_time,
            "Prediction Time (sec)": prediction_time,
            "Train Samples": len(train_df),
            "Test Samples": len(test_df),
            "Number of Queues": num_labels,
            "Max Length": MAX_LENGTH,
            "Batch Size": BATCH_SIZE,
            "Epochs": NUM_EPOCHS,
        }
    ]
)

results_path = os.path.join(
    REPORT_DIR,
    "distilbert_results.csv",
)

results.to_csv(
    results_path,
    index=False,
)


# ============================================================
# METADATA
# ============================================================

metadata = {
    "model": MODEL_NAME,
    "task": "Queue classification",
    "dataset": DATA_PATH,
    "train_samples": len(train_df),
    "test_samples": len(test_df),
    "number_of_queues": num_labels,
    "classes": list(label_encoder.classes_),
    "max_length": MAX_LENGTH,
    "batch_size": BATCH_SIZE,
    "epochs": NUM_EPOCHS,
    "learning_rate": LEARNING_RATE,
    "random_state": RANDOM_STATE,
    "device": device,
    "accuracy": float(accuracy),
    "precision_macro": float(precision_macro),
    "recall_macro": float(recall_macro),
    "macro_f1": float(macro_f1),
    "weighted_f1": float(weighted_f1),
    "errors": errors,
    "training_time_seconds": float(training_time),
    "prediction_time_seconds": float(prediction_time),
}

metadata_path = os.path.join(
    REPORT_DIR,
    "distilbert_metadata.json",
)

with open(
    metadata_path,
    "w",
    encoding="utf-8",
) as f:
    json.dump(
        metadata,
        f,
        indent=4,
    )


# ============================================================
# SAVE FINAL EXPERIMENTAL MODEL
# ============================================================

print()
print("Saving experimental DistilBERT model...")

trainer.save_model(MODEL_OUTPUT_DIR)
tokenizer.save_pretrained(MODEL_OUTPUT_DIR)

label_encoder_path = os.path.join(
    MODEL_OUTPUT_DIR,
    "label_encoder_classes.json",
)

with open(
    label_encoder_path,
    "w",
    encoding="utf-8",
) as f:
    json.dump(
        list(label_encoder.classes_),
        f,
        indent=4,
    )


print()
print("Experimental model saved to:")
print(MODEL_OUTPUT_DIR)


# ============================================================
# FINAL SUMMARY
# ============================================================

print()
print("=" * 70)
print("FINAL SUMMARY")
print("=" * 70)

print()
print("Model              : DistilBERT")
print("Base model         :", MODEL_NAME)
print("Task               : Queue classification")
print("Queues             :", num_labels)

print()
print(f"Accuracy           : {accuracy:.4f}")
print(f"Macro F1           : {macro_f1:.4f}")
print(f"Weighted F1        : {weighted_f1:.4f}")

print()
print("Reports saved in:")
print(REPORT_DIR)

print()
print("=" * 70)
print("DEEP LEARNING EXPERIMENT COMPLETED")
print("=" * 70)