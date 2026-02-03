import unittest

from fastapi.testclient import TestClient
from src.api.dashboard import app


class TestDashboard(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_dashboard_ui(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn("ML Zoo Dashboard", response.text)

    def test_api_runs(self):
        response = self.client.get("/api/runs")
        self.assertEqual(response.status_code, 200)
        runs = response.json()
        self.assertIsInstance(runs, list)
        print(f"\nFound {len(runs)} runs in MLflow directory.")


if __name__ == "__main__":
    unittest.main()
