import unittest

from fastapi.testclient import TestClient

from src.api.main import app


class ApiEndpointTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def test_health_endpoint_reports_service_status(self):
        response = self.client.get("/health")

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["status"], "healthy")
        self.assertEqual(payload["services"]["ticket_ml"], "available")
        self.assertEqual(payload["services"]["database"], "SQLite")

    def test_predict_endpoint_returns_confidence_and_recommendation(self):
        response = self.client.post(
            "/predict",
            json={"ticket_text": "VPN is down and employees cannot connect to the network"},
        )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertIn("confidence", payload)
        self.assertIn("recommendation", payload)
        self.assertGreaterEqual(payload["confidence"], 0.0)
        self.assertLessEqual(payload["confidence"], 1.0)


if __name__ == "__main__":
    unittest.main()
