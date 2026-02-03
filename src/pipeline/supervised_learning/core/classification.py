"""
Classification module for supervised learning.

Provides a Lightning module for classification tasks.
"""

from typing import Any, Dict, Optional, cast

import pytorch_lightning as pl
import torch
from torch import nn

from ..common.losses import get_loss


class ClassificationModule(pl.LightningModule):
    """
    Lightning Module for classification tasks.

    Supports multi-class and binary classification with various loss functions.
    """

    def __init__(
        self,
        backbone: Optional[nn.Module] = None,
        cfg: Optional[Dict[str, Any]] = None,
        loss_fn: str = "cross_entropy",
        num_classes: int = 2,
        learning_rate: float = 1e-3,
        **kwargs,
    ):
        """
        Initialize the classification module.

        Args:
            backbone: Pre-trained backbone model.
            cfg: Configuration dictionary.
            loss_fn: Loss function name ('cross_entropy', 'focal', 'bce').
            num_classes: Number of output classes.
            learning_rate: Learning rate for optimizer.
            **kwargs: Additional arguments.
        """
        super().__init__()
        cfg = cfg or {}
        self.save_hyperparameters(ignore=["backbone"])

        self.backbone = backbone
        self.learning_rate = learning_rate
        self.num_classes = num_classes

        # Get hidden dimension
        hidden_dim = int(cfg.get("hidden_dim", 128))

        # Classification head
        self.head = nn.Linear(hidden_dim, num_classes)

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

        logits = self(x)
        loss = self.loss_fn(logits, y)

        # Compute accuracy
        preds = torch.argmax(logits, dim=-1)
        acc = (preds == y).float().mean()

        self.log("train/loss", loss, prog_bar=True)
        self.log("train/acc", acc, prog_bar=True)
        return loss

    def validation_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        """Validation step."""
        if isinstance(batch, dict):
            x = batch["observation"]
            y = batch["target"]
        else:
            x, y = batch

        logits = self(x)
        loss = self.loss_fn(logits, y)

        preds = torch.argmax(logits, dim=-1)
        acc = (preds == y).float().mean()

        self.log("val/loss", loss, prog_bar=True)
        self.log("val/acc", acc, prog_bar=True)
        return loss

    def configure_optimizers(self):
        """Configure optimizer."""
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
