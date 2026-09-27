import unittest

from fastapi.testclient import TestClient

from src.agent.main import app


class AgentEndpointTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def test_health_endpoint_reports_agent(self):
        response = self.client.get("/health")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["service"], "hardtec-agent")


if __name__ == "__main__":
    unittest.main()