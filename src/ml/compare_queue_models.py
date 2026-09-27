from pathlib import Path
import time

import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)

from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    from lightgbm import LGBMClassifier
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False


# ============================================================
# CONFIGURATION
# ============================================================

DATA_PATH = Path("data/processed/tickets_clean.csv")
REPORT_DIR = Path("reports/model_comparison")

RANDOM_STATE = 42
TEST_SIZE = 0.20

# Standard TF-IDF
TFIDF_MAX_FEATURES = 30000

# XGBoost-specific reduction
XGB_MAX_FEATURES = 10000

REPORT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("QUEUE MODEL COMPARISON")
print("=" * 70)

print("\n[1/7] Loading dataset...")

df = pd.read_csv(DATA_PATH)

required_columns = ["ticket_text", "type", "queue"]

for col in required_columns:
    if col not in df.columns:
        raise ValueError(f"Missing required column: {col}")

df = df.dropna(subset=required_columns).copy()

df["ticket_text"] = df["ticket_text"].astype(str)
df["type"] = df["type"].astype(str)
df["queue"] = df["queue"].astype(str)

print(f"Dataset shape: {df.shape}")
print(f"Number of queues: {df['queue'].nunique()}")
print(f"Number of types: {df['type'].nunique()}")


# ============================================================
# SPLIT
# ============================================================

print("\n[2/7] Train/test split...")

X_text = df["ticket_text"]
X_type = df[["type"]]
y = df["queue"]

X_text_train, X_text_test, X_type_train, X_type_test, y_train, y_test = (
    train_test_split(
        X_text,
        X_type,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )
)

print(f"Train samples: {len(y_train)}")
print(f"Test samples : {len(y_test)}")


# ============================================================
# TF-IDF
# ============================================================

print("\n[3/7] Building TF-IDF features...")

vectorizer = TfidfVectorizer(
    lowercase=True,
    max_features=TFIDF_MAX_FEATURES,
    ngram_range=(1, 2),
    min_df=2,
    sublinear_tf=True,
)

X_text_train_tfidf = vectorizer.fit_transform(X_text_train)
X_text_test_tfidf = vectorizer.transform(X_text_test)

print(f"TF-IDF train shape: {X_text_train_tfidf.shape}")
print(f"TF-IDF test shape : {X_text_test_tfidf.shape}")


# ============================================================
# TYPE ENCODING
# ============================================================

print("\n[4/7] Encoding type metadata...")

encoder = OneHotEncoder(
    handle_unknown="ignore",
    sparse_output=True,
)

X_type_train_encoded = encoder.fit_transform(X_type_train)
X_type_test_encoded = encoder.transform(X_type_test)

X_train = hstack(
    [X_text_train_tfidf, X_type_train_encoded],
    format="csr",
)

X_test = hstack(
    [X_text_test_tfidf, X_type_test_encoded],
    format="csr",
)

print(f"Combined train shape: {X_train.shape}")
print(f"Combined test shape : {X_test.shape}")


# ============================================================
# LABEL ENCODING FOR XGBOOST
# ============================================================

print("\n[5/7] Preparing XGBoost labels...")

label_encoder = LabelEncoder()

label_encoder.fit(y_train)

y_train_encoded = label_encoder.transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

print(f"XGBoost classes: {len(label_encoder.classes_)}")


# ============================================================
# METRICS
# ============================================================

results = []


def evaluate_model(name, model, X_train_data, X_test_data, y_train_data, y_test_data):
    print("\n" + "-" * 70)
    print(f"Training: {name}")
    print("-" * 70)

    start = time.time()

    model.fit(X_train_data, y_train_data)

    training_time = time.time() - start

    pred = model.predict(X_test_data)

    accuracy = accuracy_score(y_test_data, pred)

    precision_macro = precision_score(
        y_test_data,
        pred,
        average="macro",
        zero_division=0,
    )

    recall_macro = recall_score(
        y_test_data,
        pred,
        average="macro",
        zero_division=0,
    )

    f1_macro = f1_score(
        y_test_data,
        pred,
        average="macro",
        zero_division=0,
    )

    f1_weighted = f1_score(
        y_test_data,
        pred,
        average="weighted",
        zero_division=0,
    )

    errors = int(np.sum(pred != y_test_data))

    print(f"Accuracy        : {accuracy:.4f}")
    print(f"Precision Macro : {precision_macro:.4f}")
    print(f"Recall Macro    : {recall_macro:.4f}")
    print(f"Macro F1        : {f1_macro:.4f}")
    print(f"Weighted F1     : {f1_weighted:.4f}")
    print(f"Errors          : {errors}")
    print(f"Training time   : {training_time:.2f} sec")

    results.append(
        {
            "Model": name,
            "Accuracy": accuracy,
            "Precision Macro": precision_macro,
            "Recall Macro": recall_macro,
            "Macro F1": f1_macro,
            "Weighted F1": f1_weighted,
            "Errors": errors,
            "Training Time (sec)": training_time,
        }
    )

    report = classification_report(
        y_test_data,
        pred,
        output_dict=True,
        zero_division=0,
    )

    report_df = pd.DataFrame(report).transpose()

    report_path = REPORT_DIR / f"{name.lower().replace(' ', '_')}_classification_report.csv"
    report_df.to_csv(report_path)

    return model


# ============================================================
# LINEAR SVC
# ============================================================

linear_svc = LinearSVC(
    C=1.5,
    class_weight="balanced",
    random_state=RANDOM_STATE,
)

evaluate_model(
    "LinearSVC",
    linear_svc,
    X_train,
    X_test,
    y_train,
    y_test,
)


# ============================================================
# LOGISTIC REGRESSION
# ============================================================

logistic = LogisticRegression(
    C=1.0,
    class_weight="balanced",
    max_iter=1000,
    solver="lbfgs",
    random_state=RANDOM_STATE,
)

evaluate_model(
    "LogisticRegression",
    logistic,
    X_train,
    X_test,
    y_train,
    y_test,
)


# ============================================================
# RANDOM FOREST
# ============================================================

random_forest = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    class_weight="balanced",
    n_jobs=-1,
    random_state=RANDOM_STATE,
)

evaluate_model(
    "RandomForest",
    random_forest,
    X_train,
    X_test,
    y_train,
    y_test,
)


# ============================================================
# XGBOOST - OPTIMIZED FOR CPU / 16 GB RAM
# ============================================================

if XGBOOST_AVAILABLE:

    print("\n" + "=" * 70)
    print("XGBOOST CPU OPTIMIZED")
    print("=" * 70)

    print(f"Original feature count: {X_train.shape[1]}")

    # --------------------------------------------------------
    # Reduce TF-IDF features
    # --------------------------------------------------------

    xgb_text_vectorizer = TfidfVectorizer(
        lowercase=True,
        max_features=XGB_MAX_FEATURES,
        ngram_range=(1, 2),
        min_df=2,
        sublinear_tf=True,
    )

    xgb_text_train = xgb_text_vectorizer.fit_transform(X_text_train)
    xgb_text_test = xgb_text_vectorizer.transform(X_text_test)

    xgb_train = hstack(
        [xgb_text_train, X_type_train_encoded],
        format="csr",
    )

    xgb_test = hstack(
        [xgb_text_test, X_type_test_encoded],
        format="csr",
    )

    print(f"Reduced XGBoost feature count: {xgb_train.shape[1]}")

    # --------------------------------------------------------
    # Convert sparse matrix to dense
    #
    # Important:
    # 10k features x 16k samples ≈ 640 MB float32
    # This is acceptable for a 16 GB RAM machine.
    # --------------------------------------------------------

    print("Converting reduced matrices to float32...")

    xgb_train_dense = xgb_train.astype(np.float32).toarray()
    xgb_test_dense = xgb_test.astype(np.float32).toarray()

    print(
        f"XGBoost train memory: "
        f"{xgb_train_dense.nbytes / (1024 ** 2):.1f} MB"
    )

    print(
        f"XGBoost test memory : "
        f"{xgb_test_dense.nbytes / (1024 ** 2):.1f} MB"
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    xgb_model = XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.10,
        subsample=0.8,
        colsample_bytree=0.5,
        objective="multi:softmax",
        num_class=len(label_encoder.classes_),
        eval_metric="mlogloss",
        tree_method="hist",
        max_bin=64,
        min_child_weight=3,
        reg_lambda=2.0,
        random_state=RANDOM_STATE,
        n_jobs=4,
        verbosity=1,
    )

    print("\nTraining optimized XGBoost...")

    start = time.time()

    xgb_model.fit(
        xgb_train_dense,
        y_train_encoded,
    )

    training_time = time.time() - start

    print(f"XGBoost training time: {training_time:.2f} sec")

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    pred_encoded = xgb_model.predict(xgb_test_dense)

    pred_encoded = np.asarray(pred_encoded).astype(int).reshape(-1)

    pred = label_encoder.inverse_transform(pred_encoded)

    accuracy = accuracy_score(y_test, pred)

    precision_macro = precision_score(
        y_test,
        pred,
        average="macro",
        zero_division=0,
    )

    recall_macro = recall_score(
        y_test,
        pred,
        average="macro",
        zero_division=0,
    )

    f1_macro = f1_score(
        y_test,
        pred,
        average="macro",
        zero_division=0,
    )

    f1_weighted = f1_score(
        y_test,
        pred,
        average="weighted",
        zero_division=0,
    )

    errors = int(np.sum(pred != y_test))

    print("\nXGBoost results:")
    print(f"Accuracy        : {accuracy:.4f}")
    print(f"Precision Macro : {precision_macro:.4f}")
    print(f"Recall Macro    : {recall_macro:.4f}")
    print(f"Macro F1        : {f1_macro:.4f}")
    print(f"Weighted F1     : {f1_weighted:.4f}")
    print(f"Errors          : {errors}")

    results.append(
        {
            "Model": "XGBoost",
            "Accuracy": accuracy,
            "Precision Macro": precision_macro,
            "Recall Macro": recall_macro,
            "Macro F1": f1_macro,
            "Weighted F1": f1_weighted,
            "Errors": errors,
            "Training Time (sec)": training_time,
        }
    )

    report = classification_report(
        y_test,
        pred,
        output_dict=True,
        zero_division=0,
    )

    report_df = pd.DataFrame(report).transpose()

    report_path = (
        REPORT_DIR
        / "xgboost_classification_report.csv"
    )

    report_df.to_csv(report_path)

else:
    print("\nXGBoost is not installed. Skipping.")


# ============================================================
# LIGHTGBM
# ============================================================

if LIGHTGBM_AVAILABLE:

    lightgbm = LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        max_depth=-1,
        subsample=0.8,
        colsample_bytree=0.8,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbosity=-1,
    )

    evaluate_model(
        "LightGBM",
        lightgbm,
        X_train,
        X_test,
        y_train,
        y_test,
    )

else:
    print("\nLightGBM is not installed. Skipping.")


# ============================================================
# FINAL COMPARISON
# ============================================================

print("\n" + "=" * 70)
print("FINAL MODEL COMPARISON")
print("=" * 70)

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="Macro F1",
    ascending=False,
)

print("\n")
print(results_df.to_string(index=False))

# Save results
results_path = REPORT_DIR / "queue_models_comparison.csv"

results_df.to_csv(
    results_path,
    index=False,
)

print(f"\nResults saved to:")
print(results_path)

print("\n" + "=" * 70)
print("BEST MODEL")
print("=" * 70)

best = results_df.iloc[0]

print(f"Model      : {best['Model']}")
print(f"Accuracy   : {best['Accuracy']:.4f}")
print(f"Macro F1   : {best['Macro F1']:.4f}")
print(f"Weighted F1: {best['Weighted F1']:.4f}")

print("\nComparison completed.")