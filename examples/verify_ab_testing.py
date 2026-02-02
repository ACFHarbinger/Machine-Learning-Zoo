import unittest
from fastapi.testclient import TestClient
from src.api.server import app


class TestABTesting(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_ab_experiment_creation(self):
        # Create an experiment
        response = self.client.post(
            "/v1/ab_experiment",
            json={
                "test_id": "test_exp_1",
                "variants": [
                    {"model_path": "model_a", "engine": "torch"},
                    {"model_path": "model_b", "engine": "torch"},
                ],
                "traffic_split": [0.5, 0.5],
            },
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "success")

    def test_ab_traffic_splitting(self):
        # Create an experiment with 100% split to variant B
        self.client.post(
            "/v1/ab_experiment",
            json={
                "test_id": "test_exp_2",
                "variants": [
                    {"model_path": "model_a", "engine": "torch"},
                    {"model_path": "model_b", "engine": "torch"},
                ],
                "traffic_split": [0.0, 1.0],
            },
        )

        # This will fail in predict unless we mock get_model or have real models
        # But we can verify the variants are selected correctly in units
        from src.api.server import AB_MANAGER

        variant = AB_MANAGER.get_variant("test_exp_2", session_id="user_123")
        self.assertEqual(variant["model_path"], "model_b")


if __name__ == "__main__":
    unittest.main()
