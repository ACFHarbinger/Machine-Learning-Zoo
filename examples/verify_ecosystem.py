"""Verification script for Phase 9: Ecosystem Growth."""

import unittest
from unittest.mock import MagicMock, patch

from src.pipeline.training.automl import HyperparameterOptimizer
from src.utils.registry import Registry


class TestEcosystem(unittest.TestCase):
    def test_plugin_loading(self):
        print("Testing Plugin System...")

        # Create a mock entry point
        mock_ep = MagicMock()
        mock_ep.name = "MockPlugin"
        mock_ep.load.return_value = str  # Load str class as a mock model

        # Mock importlib.metadata.entry_points
        with patch("importlib.metadata.entry_points") as mock_eps:
            # Setup mock return value for 3.10+ select()
            mock_select = MagicMock()
            mock_select.select.return_value = [mock_ep]
            mock_eps.return_value = mock_select

            # Create a localized registry for testing
            reg = Registry("TestRegistry", entry_point_group="ml_zoo.test")

            # Trigger loading
            available = reg.list_available()
            print(f"Available in TestRegistry: {available}")

            self.assertIn("MockPlugin", available)
            cls = reg.get("MockPlugin")
            self.assertEqual(cls, str)
            print("Plugin loading successful.")

    def test_automl_optimizer(self):
        print("\nTesting AutoML Optimizer...")
        optimizer = HyperparameterOptimizer(
            study_name="test_study", direction="minimize"
        )

        def objective(trial):
            x = trial.suggest_float("x", -10, 10)
            return (x - 2) ** 2  # Min at x=2

        best_params = optimizer.optimize(objective, n_trials=50)
        print(f"Best params found: {best_params}")

        self.assertAlmostEqual(best_params["x"], 2, delta=0.5)
        print("AutoML optimization successful.")


if __name__ == "__main__":
    unittest.main(argv=["first-arg-is-ignored"], exit=False)
