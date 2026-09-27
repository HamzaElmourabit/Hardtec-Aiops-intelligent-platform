import unittest

from src.pipeline.predictor import predict_ticket


class PredictTicketTests(unittest.TestCase):
    def test_predict_ticket_returns_classification_payload(self):
        result = predict_ticket("VPN is down and employees cannot connect to the network")

        self.assertIsInstance(result, dict)
        self.assertIn("ticket_type", result)
        self.assertIn("priority", result)
        self.assertIn("queue", result)


if __name__ == "__main__":
    unittest.main()
