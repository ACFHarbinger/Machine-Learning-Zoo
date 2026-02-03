import unittest

import torch
from src.models.hub import ModelHub
from src.pipeline.training.fairness import FairnessAuditor


class TestHubFairness(unittest.TestCase):
    def setUp(self):
        self.hub = ModelHub()
        self.auditor = FairnessAuditor()

    def test_hub_stats(self):
        print("\nTesting ModelHub storage stats...")
        stats = self.hub.get_storage_stats()
        self.assertIn("hub_size_gb", stats)
        self.assertIn("disk_free_gb", stats)
        print(f"Hub Status: {stats}")

    def test_hub_info(self):
        print("\nTesting ModelHub metadata...")
        info = self.hub.get_model_info("llama-3-8b-instruct")
        self.assertIsNotNone(info)
        self.assertEqual(info["metadata"]["author"], "Meta")
        print(f"Model Info: {info['metadata']}")

    def test_fairness_auditor(self):
        print("\nTesting FairnessAuditor...")
        # Synthetic data: 100 samples
        # Group 0: 50 samples, 40 positives (80% rate)
        # Group 1: 50 samples, 10 positives (20% rate)
        y_true = torch.randint(0, 2, (100,))
        y_pred = torch.zeros(100, dtype=torch.long)
        y_pred[:40] = 1  # Group 0 positive
        y_pred[50:60] = 1  # Group 1 positive

        sensitive_attr = torch.zeros(100, dtype=torch.long)
        sensitive_attr[50:] = 1  # Second half is group 1

        results = self.auditor.audit(y_true, y_pred, sensitive_attr)

        print(f"Fairness Audit: {results}")

        # Demographic Parity: 0.8 - 0.2 = 0.6
        self.assertAlmostEqual(results["demographic_parity_diff"], 0.6)
        # Disparate Impact: 0.2 / 0.8 = 0.25
        self.assertAlmostEqual(results["disparate_impact_ratio"], 0.25)
        self.assertIn("equalized_odds_diff", results)


if __name__ == "__main__":
    unittest.main()
