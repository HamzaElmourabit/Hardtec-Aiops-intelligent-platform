from pathlib import Path

import duckdb
import pandas as pd
import sqlite3


# ============================================================
# PATHS
# ============================================================

ROOT_DIR = Path(__file__).resolve().parent.parent

DATASET_PATH = (
    ROOT_DIR
    / "data"
    / "processed"
    / "tickets_clean.csv"
)

SQLITE_PATH = (
    ROOT_DIR
    / "data"
    / "lake"
    / "tickets_predictions.db"
)


# ============================================================
# DUCKDB CONNECTION
# ============================================================

def get_connection():
    """
    Creates a local in-memory DuckDB connection.

    DuckDB is used for ticket dataset analytics.
    """

    return duckdb.connect()


# ============================================================
# TICKET ANALYTICS
# ============================================================

def get_ticket_analytics():
    """
    Generates ticket analytics directly from tickets_clean.csv
    using DuckDB SQL.
    """

    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset introuvable : {DATASET_PATH}"
        )

    conn = get_connection()

    try:
        query = """
            SELECT
                type AS TICKET_TYPE,
                priority AS PRIORITY,
                queue AS QUEUE,

                COUNT(*) AS TOTAL_TICKETS,

                COUNT(
                    CASE
                        WHEN LOWER(priority) = 'high'
                        THEN 1
                    END
                ) AS HIGH_PRIORITY_TICKETS,

                COUNT(
                    CASE
                        WHEN LOWER(priority) = 'medium'
                        THEN 1
                    END
                ) AS MEDIUM_PRIORITY_TICKETS,

                COUNT(
                    CASE
                        WHEN LOWER(priority) = 'low'
                        THEN 1
                    END
                ) AS LOW_PRIORITY_TICKETS

            FROM read_csv_auto(?)

            GROUP BY
                type,
                priority,
                queue

            ORDER BY
                TOTAL_TICKETS DESC
        """

        df = conn.execute(
            query,
            [str(DATASET_PATH)]
        ).df()

        return df

    finally:
        conn.close()


# ============================================================
# SQLITE CONNECTION
# ============================================================

def get_sqlite_connection():
    """
    Creates a connection to the SQLite prediction database.
    """

    SQLITE_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    return sqlite3.connect(
        SQLITE_PATH
    )


# ============================================================
# AI PREDICTION HISTORY
# ============================================================

def get_ai_predictions(limit=100):
    """
    Loads AI prediction history from SQLite.

    SQLite is the single source of truth for predictions.
    Predictions are inserted by FastAPI /predict.
    """

    if not SQLITE_PATH.exists():
        return pd.DataFrame(
            columns=[
                "TICKET_ID",
                "TICKET_TEXT",
                "TYPE",
                "PRIORITY",
                "QUEUE",
                "CONFIDENCE",
                "RECOMMENDATION",
                "CREATED_AT",
            ]
        )

    conn = get_sqlite_connection()

    try:
        query = """
            SELECT
                id AS TICKET_ID,
                ticket_text AS TICKET_TEXT,
                ticket_type AS TYPE,
                priority AS PRIORITY,
                queue AS QUEUE,
                confidence AS CONFIDENCE,
                recommendation AS RECOMMENDATION,
                created_at AS CREATED_AT

            FROM ticket_predictions

            ORDER BY created_at DESC

            LIMIT ?
        """

        df = pd.read_sql_query(
            query,
            conn,
            params=[limit],
        )

        return df

    finally:
        conn.close()


# ============================================================
# LAST PREDICTION
# ============================================================

def get_last_prediction():
    """
    Returns the most recent AI prediction
    stored in SQLite.
    """

    df = get_ai_predictions(limit=1)

    return df