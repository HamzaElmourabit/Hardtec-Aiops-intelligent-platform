"""
Unit tests for the TicketStore persistence layer.
"""

import unittest
import os
import tempfile
import shutil
from src.data.ticket_store import TicketStore


class TicketStoreTests(unittest.TestCase):
    def setUp(self):
        """Create a fresh temporary database for each test."""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "test_tickets.db")
        self.store = TicketStore(db_path=self.db_path)

    def tearDown(self):
        """Clean up the temporary database after each test."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_save_and_retrieve_prediction(self):
        """Test saving and retrieving a single prediction."""
        prediction_id = self.store.save_prediction(
            ticket_text="Test ticket",
            ticket_type="Problem",
            priority="HIGH",
            queue="Technical Support",
            confidence=0.95,
            recommendation="Escalate immediately",
        )

        self.assertIsNotNone(prediction_id)
        self.assertGreater(prediction_id, 0)

        retrieved = self.store.get_prediction_by_id(prediction_id)
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved["ticket_text"], "Test ticket")
        self.assertEqual(retrieved["ticket_type"], "Problem")
        self.assertEqual(retrieved["priority"], "HIGH")
        self.assertEqual(retrieved["queue"], "Technical Support")

    def test_get_recent_predictions(self):
        """Test retrieving recent predictions."""
        # Save multiple predictions
        for i in range(5):
            self.store.save_prediction(
                ticket_text=f"Ticket {i}",
                ticket_type="Problem",
                priority="MEDIUM",
                queue="Queue",
            )

        # Retrieve recent
        recent = self.store.get_recent_predictions(limit=3)
        self.assertEqual(len(recent), 3)

        # Verify we get 3 of the 5 tickets (order may vary if timestamps are identical)
        ticket_texts = [r["ticket_text"] for r in recent]
        self.assertTrue(all(t.startswith("Ticket") for t in ticket_texts))

    def test_get_stats(self):
        """Test retrieving statistics."""
        self.store.save_prediction(
            ticket_text="Ticket 1",
            ticket_type="Problem",
            priority="HIGH",
            queue="Technical Support",
        )
        self.store.save_prediction(
            ticket_text="Ticket 2",
            ticket_type="Problem",
            priority="MEDIUM",
            queue="Technical Support",
        )
        self.store.save_prediction(
            ticket_text="Ticket 3",
            ticket_type="Request",
            priority="LOW",
            queue="Service Desk",
        )

        stats = self.store.get_stats()

        self.assertEqual(stats["total_predictions"], 3)
        self.assertIn("HIGH", stats["priority_distribution"])
        self.assertEqual(stats["priority_distribution"]["HIGH"], 1)
        self.assertEqual(stats["priority_distribution"]["MEDIUM"], 1)
        self.assertEqual(stats["priority_distribution"]["LOW"], 1)
        self.assertIn("Problem", stats["type_distribution"])
        self.assertEqual(stats["type_distribution"]["Problem"], 2)
        self.assertIn("Request", stats["type_distribution"])

    def test_get_prediction_by_nonexistent_id(self):
        """Test retrieving a non-existent prediction."""
        result = self.store.get_prediction_by_id(9999)
        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
