import pandas as pd
import re
from pathlib import Path


# ==============================
# Paths
# ==============================

BASE_DIR = Path(__file__).resolve().parents[2]

RAW_FILE = BASE_DIR / "data" / "raw" / "tickets.csv"
PROCESSED_FILE = BASE_DIR / "data" / "processed" / "tickets_clean.csv"


# ==============================
# Text cleaning
# ==============================

def clean_text(text):
    """
    Nettoie le texte d'un ticket.
    """

    if pd.isna(text):
        return ""

    text = str(text)

    # Convertir en minuscules
    text = text.lower()

    # Supprimer les URLs
    text = re.sub(r"http\S+|www\S+", " ", text)

    # Supprimer les emails
    text = re.sub(r"\S+@\S+", " ", text)

    # Supprimer les caractères spéciaux
    text = re.sub(r"[^a-zA-ZÀ-ÿ0-9\s]", " ", text)

    # Supprimer les espaces multiples
    text = re.sub(r"\s+", " ", text)

    return text.strip()


# ==============================
# Load data
# ==============================

def load_data():
    print("Loading dataset...")

    df = pd.read_csv(RAW_FILE)

    print(f"Dataset shape: {df.shape}")

    return df


# ==============================
# Preprocessing
# ==============================

def preprocess_data(df):

    print("Starting preprocessing...")

    # Supprimer les lignes sans sujet ni description
    df = df.dropna(
        subset=["subject", "body"],
        how="all"
    ).copy()

    # Remplacer les valeurs manquantes
    df["subject"] = df["subject"].fillna("")
    df["body"] = df["body"].fillna("")

    # Combiner subject + body
    df["ticket_text"] = (
        df["subject"].astype(str)
        + " "
        + df["body"].astype(str)
    )

    # Nettoyage du texte
    df["ticket_text"] = df["ticket_text"].apply(clean_text)

    # Supprimer les tickets dont le texte est vide
    df = df[df["ticket_text"].str.len() > 0].copy()

    # Nettoyer les colonnes importantes
    df["type"] = df["type"].fillna("Unknown")
    df["queue"] = df["queue"].fillna("Unknown")
    df["priority"] = df["priority"].fillna("Unknown")
    df["language"] = df["language"].fillna("Unknown")

    # Garder uniquement les colonnes utiles
    df = df[
        [
            "ticket_text",
            "type",
            "queue",
            "priority",
            "language",
            "answer"
        ]
    ].copy()

    return df


# ==============================
# Save processed dataset
# ==============================

def save_data(df):

    PROCESSED_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(
        PROCESSED_FILE,
        index=False
    )

    print(f"Processed dataset saved to: {PROCESSED_FILE}")


# ==============================
# Main
# ==============================

if __name__ == "__main__":

    df = load_data()

    df_clean = preprocess_data(df)

    save_data(df_clean)

    print("\nPreprocessing completed!")
    print(f"Final shape: {df_clean.shape}")

    print("\nColumns:")
    print(df_clean.columns.tolist())

    print("\nPriority distribution:")
    print(df_clean["priority"].value_counts())

    print("\nType distribution:")
    print(df_clean["type"].value_counts())

    print("\nQueue distribution:")
    print(df_clean["queue"].value_counts())