"""
HARDTEC - Snowflake Data Loader

Fonctionnalités :
1. Charger le dataset source HARDTEC dans Snowflake.
2. Sauvegarder les prédictions ML de FastAPI dans Snowflake.

La connexion Snowflake n'est jamais créée lors de l'import du module.
"""

from pathlib import Path
from typing import Optional

import pandas as pd
import snowflake.connector

from src.config import get_snowflake_config


# ============================================================
# CONFIGURATION
# ============================================================

CSV_PATH = Path("data/raw/tickets.csv")

RAW_TABLE = "TICKETS_RAW"
PREDICTIONS_TABLE = "TICKET_PREDICTIONS"


# ============================================================
# SNOWFLAKE CONNECTION
# ============================================================


def get_connection():
    """
    Crée une connexion Snowflake uniquement lorsque nécessaire.
    """

    config = get_snowflake_config()

    return snowflake.connector.connect(
        account=config["account"],
        user=config["user"],
        password=config["password"],
        warehouse=config["warehouse"],
        database=config["database"],
        schema=config["schema"],
        role=config["role"],
    )


# ============================================================
# RAW TABLE
# ============================================================


def create_raw_table(cursor):
    """
    Crée la table Snowflake correspondant au dataset source
    de 20 000 tickets et 15 colonnes.
    """

    cursor.execute(f"""
        CREATE OR REPLACE TABLE {RAW_TABLE} (
            SUBJECT VARCHAR,
            BODY VARCHAR,
            ANSWER VARCHAR,
            TYPE VARCHAR,
            QUEUE VARCHAR,
            PRIORITY VARCHAR,
            LANGUAGE VARCHAR,
            TAG_1 VARCHAR,
            TAG_2 VARCHAR,
            TAG_3 VARCHAR,
            TAG_4 VARCHAR,
            TAG_5 VARCHAR,
            TAG_6 VARCHAR,
            TAG_7 VARCHAR,
            TAG_8 VARCHAR
        )
    """)


# ============================================================
# LOAD RAW DATASET
# ============================================================


def load_raw_dataset() -> int:
    """
    Charge data/raw/tickets.csv dans Snowflake.

    Le fichier attendu contient les colonnes :

    subject
    body
    answer
    type
    queue
    priority
    language
    tag_1 ... tag_8

    Retourne le nombre de lignes chargées.
    """

    if not CSV_PATH.exists():
        raise FileNotFoundError(
            f"Dataset introuvable : {CSV_PATH}"
        )

    print("=" * 60)
    print("HARDTEC - Chargement du dataset vers Snowflake")
    print("=" * 60)

    # ========================================================
    # LECTURE DU CSV
    # ========================================================

    df = pd.read_csv(CSV_PATH)

    print(f"Dataset chargé : {df.shape[0]} lignes")
    print(f"Nombre de colonnes : {df.shape[1]}")

    print("\nColonnes détectées :")

    for column in df.columns:
        print(f" - {column}")

    # ========================================================
    # COLONNES ATTENDUES
    # ========================================================

    required_columns = [
        "subject",
        "body",
        "answer",
        "type",
        "queue",
        "priority",
        "language",
        "tag_1",
        "tag_2",
        "tag_3",
        "tag_4",
        "tag_5",
        "tag_6",
        "tag_7",
        "tag_8",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "Colonnes manquantes dans le dataset : "
            + ", ".join(missing_columns)
        )

    # ========================================================
    # ORDRE DES COLONNES
    # ========================================================

    df = df[required_columns].copy()

    # ========================================================
    # CONNEXION
    # ========================================================

    conn = None
    cursor = None

    try:

        print("\nConnexion à Snowflake...")

        conn = get_connection()

        cursor = conn.cursor()

        print("Connexion Snowflake réussie.")

        # ====================================================
        # CREATION TABLE
        # ====================================================

        print(f"\nCréation de la table {RAW_TABLE}...")

        create_raw_table(cursor)

        print(f"Table {RAW_TABLE} créée.")

        # ====================================================
        # INSERTION
        # ====================================================

        insert_sql = f"""
            INSERT INTO {RAW_TABLE} (
                SUBJECT,
                BODY,
                ANSWER,
                TYPE,
                QUEUE,
                PRIORITY,
                LANGUAGE,
                TAG_1,
                TAG_2,
                TAG_3,
                TAG_4,
                TAG_5,
                TAG_6,
                TAG_7,
                TAG_8
            )
            VALUES (
                %s, %s, %s, %s, %s,
                %s, %s, %s, %s, %s,
                %s, %s, %s, %s, %s
            )
        """

        data = [
            tuple(
                None if pd.isna(value) else str(value)
                for value in row
            )
            for row in df.itertuples(
                index=False,
                name=None
            )
        ]

        print(f"\nInsertion de {len(data)} lignes...")

        cursor.executemany(
            insert_sql,
            data
        )

        conn.commit()

        print("Données chargées avec succès.")

        # ====================================================
        # VERIFICATION
        # ====================================================

        cursor.execute(
            f"SELECT COUNT(*) FROM {RAW_TABLE}"
        )

        count = cursor.fetchone()[0]

        print(
            f"\nNombre de lignes dans Snowflake : {count}"
        )

        if count != len(df):

            raise RuntimeError(
                f"Erreur de vérification : "
                f"{len(df)} lignes attendues, "
                f"{count} trouvées."
            )

        print("\n" + "=" * 60)
        print("CHARGEMENT TERMINÉ AVEC SUCCÈS")
        print("=" * 60)

        return count

    except Exception:

        if conn is not None:
            conn.rollback()

        raise

    finally:

        if cursor is not None:
            cursor.close()

        if conn is not None:
            conn.close()


# ============================================================
# PREDICTIONS TABLE
# ============================================================


def create_predictions_table(cursor):
    """
    Crée la table utilisée par FastAPI pour enregistrer
    les prédictions ML.
    """

    cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS {PREDICTIONS_TABLE} (
            ID NUMBER AUTOINCREMENT,
            TICKET_TEXT VARCHAR,
            TICKET_TYPE VARCHAR,
            PRIORITY VARCHAR,
            QUEUE VARCHAR,
            CONFIDENCE FLOAT,
            RECOMMENDATION VARCHAR,
            SOURCE VARCHAR DEFAULT 'FASTAPI',
            CREATED_AT TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP()
        )
    """)


# ============================================================
# SAVE PREDICTION
# ============================================================


def save_prediction_to_snowflake(
    ticket_text: str,
    ticket_type: str,
    priority: str,
    queue: str,
    confidence: float = 0.85,
    recommendation: Optional[str] = None,
    source: str = "FASTAPI",
) -> bool:
    """
    Sauvegarde une prédiction ML dans Snowflake.

    IMPORTANT :
    confidence=0.85 est actuellement une valeur placeholder
    dans le projet HARDTEC. Elle ne représente pas une
    probabilité ML calibrée.
    """

    if not isinstance(ticket_text, str) or not ticket_text.strip():

        raise ValueError(
            "ticket_text doit être une chaîne non vide."
        )

    conn = None
    cursor = None

    try:

        conn = get_connection()

        cursor = conn.cursor()

        # ----------------------------------------------------
        # Création de la table si elle n'existe pas
        # ----------------------------------------------------

        create_predictions_table(cursor)

        # ----------------------------------------------------
        # INSERTION
        # ----------------------------------------------------

        insert_sql = f"""
            INSERT INTO {PREDICTIONS_TABLE} (
                TICKET_TEXT,
                TICKET_TYPE,
                PRIORITY,
                QUEUE,
                CONFIDENCE,
                RECOMMENDATION,
                SOURCE
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s)
        """

        cursor.execute(
            insert_sql,
            (
                ticket_text,
                ticket_type,
                priority,
                queue,
                float(confidence),
                recommendation,
                source,
            ),
        )

        conn.commit()

        return True

    except Exception:

        if conn is not None:
            conn.rollback()

        raise

    finally:

        if cursor is not None:
            cursor.close()

        if conn is not None:
            conn.close()


# ============================================================
# TEST CONNECTION
# ============================================================


def test_connection() -> bool:
    """
    Vérifie uniquement que la connexion Snowflake fonctionne.
    """

    conn = None
    cursor = None

    try:

        conn = get_connection()

        cursor = conn.cursor()

        cursor.execute(
            "SELECT CURRENT_VERSION()"
        )

        version = cursor.fetchone()[0]

        print(
            f"Connexion Snowflake OK - version : {version}"
        )

        return True

    finally:

        if cursor is not None:
            cursor.close()

        if conn is not None:
            conn.close()


# ============================================================
# MAIN
# ============================================================


if __name__ == "__main__":

    load_raw_dataset()