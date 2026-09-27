"""
SQLite-based ticket prediction store for local persistence.
Allows ticket history to survive across Streamlit reruns and API restarts.
"""

import sqlite3
import os
from datetime import datetime
from typing import List, Dict, Optional


class TicketStore:
    """Manage ticket predictions in a local SQLite database."""

    def __init__(self, db_path: str = None):
        """
        Initialize the ticket store.

        Args:
            db_path: Path to SQLite database file. Defaults to data/lake/tickets_predictions.db
        """
        if db_path is None:
            base_dir = os.path.dirname(
                os.path.dirname(
                    os.path.dirname(os.path.abspath(__file__))
                )
            )
            db_path = os.path.join(base_dir, "data", "lake", "tickets_predictions.db")

        self.db_path = db_path
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._init_schema()

    def _init_schema(self):
        """Create the prediction table if it does not exist."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS ticket_predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticket_text TEXT NOT NULL,
                ticket_type TEXT NOT NULL,
                priority TEXT NOT NULL,
                queue TEXT NOT NULL,
                confidence REAL NOT NULL,
                recommendation TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                source TEXT DEFAULT 'api'
            )
            """
        )

        # Create an index on created_at for faster queries
        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_created_at 
            ON ticket_predictions(created_at DESC)
            """
        )

        conn.commit()
        conn.close()

    def save_prediction(
        self,
        ticket_text: str,
        ticket_type: str,
        priority: str,
        queue: str,
        confidence: float = 0.85,
        recommendation: str = None,
        source: str = "api",
    ) -> int:
        """
        Save a ticket prediction to the database.

        Args:
            ticket_text: The ticket description
            ticket_type: The predicted ticket type
            priority: The predicted priority
            queue: The predicted queue
            confidence: Confidence score (0.0 - 1.0)
            recommendation: Recommended action text
            source: Source of prediction ('api', 'dashboard', etc.)

        Returns:
            The ID of the inserted record
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT INTO ticket_predictions
            (ticket_text, ticket_type, priority, queue, confidence, recommendation, source)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (ticket_text, ticket_type, priority, queue, confidence, recommendation, source),
        )

        conn.commit()
        record_id = cursor.lastrowid
        conn.close()

        return record_id

    def get_recent_predictions(self, limit: int = 50) -> List[Dict]:
        """
        Retrieve recent ticket predictions.

        Args:
            limit: Maximum number of records to return

        Returns:
            List of prediction dictionaries, ordered by most recent first
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT 
                id, 
                ticket_text, 
                ticket_type, 
                priority, 
                queue, 
                confidence,
                recommendation,
                created_at
            FROM ticket_predictions
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (limit,),
        )

        rows = cursor.fetchall()
        conn.close()

        return [dict(row) for row in rows]

    def get_prediction_by_id(self, prediction_id: int) -> Optional[Dict]:
        """
        Retrieve a single prediction by ID.

        Args:
            prediction_id: The prediction ID

        Returns:
            Prediction dictionary or None if not found
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT 
                id, 
                ticket_text, 
                ticket_type, 
                priority, 
                queue, 
                confidence,
                recommendation,
                created_at
            FROM ticket_predictions
            WHERE id = ?
            """,
            (prediction_id,),
        )

        row = cursor.fetchone()
        conn.close()

        return dict(row) if row else None

    def get_stats(self) -> Dict:
        """
        Get summary statistics about predictions.

        Returns:
            Dictionary with stats (total count, priority distribution, etc.)
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Total predictions
        cursor.execute("SELECT COUNT(*) FROM ticket_predictions")
        total = cursor.fetchone()[0]

        # Priority distribution
        cursor.execute(
            """
            SELECT priority, COUNT(*) as count
            FROM ticket_predictions
            GROUP BY priority
            ORDER BY count DESC
            """
        )
        priority_dist = {row[0]: row[1] for row in cursor.fetchall()}

        # Type distribution
        cursor.execute(
            """
            SELECT ticket_type, COUNT(*) as count
            FROM ticket_predictions
            GROUP BY ticket_type
            ORDER BY count DESC
            """
        )
        type_dist = {row[0]: row[1] for row in cursor.fetchall()}

        # Queue distribution
        cursor.execute(
            """
            SELECT queue, COUNT(*) as count
            FROM ticket_predictions
            GROUP BY queue
            ORDER BY count DESC
            """
        )
        queue_dist = {row[0]: row[1] for row in cursor.fetchall()}

        # Average confidence
        cursor.execute("SELECT AVG(confidence) FROM ticket_predictions")
        avg_confidence = cursor.fetchone()[0] or 0.0

        conn.close()

        return {
            "total_predictions": total,
            "priority_distribution": priority_dist,
            "type_distribution": type_dist,
            "queue_distribution": queue_dist,
            "average_confidence": round(avg_confidence, 4),
        }

    def clear_old_records(self, days: int = 90) -> int:
        """
        Delete predictions older than a specified number of days.

        Args:
            days: Number of days to keep

        Returns:
            Number of records deleted
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            DELETE FROM ticket_predictions
            WHERE created_at < datetime('now', '-' || ? || ' days')
            """,
            (days,),
        )

        conn.commit()
        deleted = cursor.rowcount
        conn.close()

        return deleted
