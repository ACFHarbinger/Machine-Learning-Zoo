"""Verification script for Phase 4 & 6: Performance & Tracking."""

import unittest
from unittest.mock import MagicMock, patch

from src.pipeline.training.lightning_module import PiLightningModule
from src.pipeline.training.trainer import TrainingOrchestrator


class TestPerformanceTracking(unittest.TestCase):
    def test_distillation_setup(self):
        print("Testing Knowledge Distillation Setup...")

        distill_config = {
            "teacher_name": "gpt2",  # Use a small real model for loading test
            "temperature": 2.0,
            "alpha": 0.5,
        }

        # Patch AutoModelForCausalLM.from_pretrained to avoid heavy download in tests
        with patch("transformers.AutoModelForCausalLM.from_pretrained") as mock_hf:
            mock_model = MagicMock()
            mock_model.config = MagicMock()
            mock_hf.return_value = mock_model

            module = PiLightningModule(
                model_name="gpt2", distillation_config=distill_config
            )

            self.assertIsNotNone(module.teacher_model)
            self.assertEqual(module.distill_config["alpha"], 0.5)
            print("Distillation setup successful.")

    def test_tracking_integration(self):
        print("\nTesting Experiment Tracking Integration...")

        tracking_config = {
            "use_wandb": True,
            "use_mlflow": True,
            "project": "test-project",
            "name": "test-run",
        }

        orchestrator = TrainingOrchestrator(model_name="gpt2")

        with (
            patch("src.pipeline.training.trainer.WandbLogger") as mock_wandb,
            patch("src.pipeline.training.trainer.MLFlowLogger") as mock_mlflow,
            patch("src.pipeline.training.trainer.pl.Trainer") as mock_trainer,
        ):
            orchestrator.train(
                train_texts=["hello world"], epochs=1, tracking_config=tracking_config
            )

            # Check if loggers were initialized
            mock_wandb.assert_called_once()
            mock_mlflow.assert_called_once()

            # Check if trainer was initialized with loggers
            args, kwargs = mock_trainer.call_args
            self.assertIn("logger", kwargs)
            self.assertEqual(len(kwargs["logger"]), 2)
            print("Tracking integration successful.")


if __name__ == "__main__":
    unittest.main(argv=["first-arg-is-ignored"], exit=False)
