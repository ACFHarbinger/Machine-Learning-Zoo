"""
Online Learning Core Module.

Adapts online learning strategies to the Lightning API.
"""

from typing import Any, cast

import torch
import torch.nn as nn

from ..base import BaseModule


class OnlineLearningModule(BaseModule):
    """
    Lightning Module for Online Learning.

    Wraps an OnlineTrainer to perform incremental updates within a PL loop.
    Note: PL isn't ideal for true streaming, but this module bridges the gap
    for batch-based simulation of online learning.
    """

    def __init__(
        self,
        backbone: nn.Module,  # Must support partial_fit if used with OnlineTrainer scikit-learn mode, or be a torch model
        cfg: dict[str, Any],
    ) -> None:
        """
        Initialize Online Learning Module.
        """
        super().__init__(cfg)
        self.backbone = backbone

        # We wrap the OnlineTrainer.
        # Note: OnlineTrainer currently expects sklearn-like BaseEstimator.
        # If backbone is a Torch model, we need an adapter or OnlineTrainer needs update.
        # For now, we assume the user might be using a Torch-compatible wrapper or we implement
        # a simple Torch-based online update here.
        #
        # Let's implement a direct Torch online update mechanism here for simplicity and robustness.

        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return cast(torch.Tensor, self.backbone(x))

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        """
        Online update step.
        Ideally batch_size is small (1) for true online, or mini-batch.
        """
        if isinstance(batch, (list, tuple)):
            x, y = batch
        elif isinstance(batch, dict):
            x = batch["data"]
            y = batch["target"]
        else:
            raise ValueError(f"Unsupported batch type: {type(batch)}")

        # For online learning, we typically take a gradient step and then
        # potentially discard data/history, or update a buffer.
        # Standard PL training_step IS a gradient step, so it fits mini-batch online learning.

        preds = self(x)
        loss = nn.functional.mse_loss(preds, y)
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
