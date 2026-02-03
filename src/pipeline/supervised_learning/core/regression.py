"""
Regression module for supervised learning.

Provides a Lightning module for regression tasks with configurable loss functions.
"""

from typing import Any, Dict, Optional, cast

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import nn

from ..common.losses import get_loss


class RegressionModule(pl.LightningModule):
    """
    Lightning Module for regression tasks.

    Supports various loss functions and backbones for time series
    and tabular regression.
    """

    def __init__(
        self,
        backbone: Optional[nn.Module] = None,
        cfg: Optional[Dict[str, Any]] = None,
        loss_fn: str = "mse",
        learning_rate: float = 1e-3,
        **kwargs,
    ):
        """
        Initialize the regression module.

        Args:
            backbone: Pre-trained backbone model.
            cfg: Configuration dictionary.
            loss_fn: Loss function name ('mse', 'l1', 'huber').
            learning_rate: Learning rate for optimizer.
            **kwargs: Additional arguments.
        """
        super().__init__()
        cfg = cfg or {}
        self.save_hyperparameters(ignore=["backbone"])

        self.backbone = backbone
        self.learning_rate = learning_rate

        # Get output dimension
        hidden_dim = int(cfg.get("hidden_dim", 128))
        output_dim = int(cfg.get("output_dim", 1))

        # Prediction head
        self.head = nn.Linear(hidden_dim, output_dim)

        # Loss function
        self.loss_fn = get_loss(loss_fn)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through backbone and head."""
        if self.backbone is not None:
            feat = self.backbone(x)
        else:
            feat = x
        return cast(torch.Tensor, self.head(feat))

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        """Training step."""
        if isinstance(batch, dict):
            x = batch["observation"]
            y = batch["target"]
        else:
            x, y = batch

        pred = self(x)
        loss = self.loss_fn(pred, y)

        self.log("train/loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        """Validation step."""
        if isinstance(batch, dict):
            x = batch["observation"]
            y = batch["target"]
        else:
            x, y = batch

        pred = self(x)
        loss = self.loss_fn(pred, y)

        self.log("val/loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        """Configure optimizer."""
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
