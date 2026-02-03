"""
Active Learning Core Module.

wraps a backbone model to support uncertainty estimation for active learning.
"""

from typing import Any, Dict, cast

import torch
import torch.nn as nn

from ..base import BaseModule
from .uncertainty import mc_dropout_predict


class ActiveLearningModule(BaseModule):
    """
    Lightning Module for Active Learning.

    Wraps a standard supervised model but adds capabilities for
    uncertainty estimation (e.g., via MC Dropout) during inference/prediction.
    """

    def __init__(
        self,
        backbone: nn.Module,
        cfg: dict[str, Any],
    ) -> None:
        """
        Initialize Active Learning Module.

        Args:
            backbone: The base PyTorch model.
            cfg: Configuration dictionary.
        """
        super().__init__(cfg)
        self.backbone = backbone
        self.mc_dropout_samples = int(cfg.get("active_learning", {}).get("mc_samples", 10))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the backbone."""
        return cast(torch.Tensor, self.backbone(x))

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        """Standard supervised training step."""
        # Assuming batch is (x, y) or dict
        if isinstance(batch, (list, tuple)):
            x, y = batch
        elif isinstance(batch, dict):
            x = batch["data"]
            y = batch["target"]
        else:
            raise ValueError(f"Unsupported batch type: {type(batch)}")

        preds = self(x)
        loss = nn.functional.mse_loss(preds, y)  # Defaulting to MSE, could make configurable
        self.log("train/loss", loss)
        return loss

    def validation_step(self, batch: Any, batch_idx: int) -> None:
        """Validation step."""
        if isinstance(batch, (list, tuple)):
            x, y = batch
        elif isinstance(batch, dict):
            x = batch["data"]
            y = batch["target"]
        else:
            return

        preds = self(x)
        loss = nn.functional.mse_loss(preds, y)
        self.log("val/loss", loss)

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Dict[str, torch.Tensor]:
        """
        Prediction step with optional uncertainty estimation.
        """
        if isinstance(batch, (list, tuple)):
            x = batch[0]
        elif isinstance(batch, dict):
            x = batch["data"]
        else:
            x = batch

        # Perform MC Dropout if configured
        if self.mc_dropout_samples > 1:
            return mc_dropout_predict(self.backbone, x, n_samples=self.mc_dropout_samples)

        # Standard prediction
        preds = self(x)
        return {"mean": preds, "variance": torch.zeros_like(preds), "std": torch.zeros_like(preds)}
