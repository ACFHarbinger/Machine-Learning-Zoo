"""
Continual Learning Core Module.

Implements strategies for Continual Learning (e.g., EWC).
"""

from typing import Any, Optional, cast

import torch
import torch.nn as nn
from pytorch_lightning.callbacks import Callback

from ..base import BaseModule
from .engine import EWCCallback


class ContinualLearningModule(BaseModule):
    """
    Lightning Module for Continual Learning.

    Integrates elastic weight consolidation (EWC) to prevent catastrophic forgetting.
    """

    def __init__(
        self,
        backbone: nn.Module,
        cfg: dict[str, Any],
    ) -> None:
        """
        Initialize Continual Learning Module.

        Args:
            backbone: The base PyTorch model.
            cfg: Configuration dictionary.
        """
        super().__init__(cfg)
        self.backbone = backbone

        # EWC Configuration
        self.ewc_lambda = float(cfg.get("continual_learning", {}).get("ewc_lambda", 0.0))
        self.ewc_callback: Optional[EWCCallback] = None

        if self.ewc_lambda > 0:
            self.ewc_callback = EWCCallback(ewc_lambda=self.ewc_lambda)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return cast(torch.Tensor, self.backbone(x))

    def configure_callbacks(self) -> list[Callback]:
        """Register EWC callback if enabled."""
        callbacks = super().configure_callbacks() if hasattr(super(), "configure_callbacks") else []

        # Normalize to list
        if not isinstance(callbacks, list):
            callbacks = [callbacks] if callbacks else []

        if self.ewc_callback:
            callbacks.append(self.ewc_callback)

        return cast(list[Callback], callbacks)

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        """Training step with EWC penalty."""
        if isinstance(batch, (list, tuple)):
            x, y = batch
        elif isinstance(batch, dict):
            x = batch["data"]
            y = batch["target"]
        else:
            raise ValueError(f"Unsupported batch type: {type(batch)}")

        preds = self(x)
        # Task loss (MSE default)
        task_loss = nn.functional.mse_loss(preds, y)

        # EWC Penalty
        ewc_loss = torch.tensor(0.0, device=self.device)
        if self.ewc_callback:
            ewc_loss = self.ewc_callback.get_ewc_loss(self)

        total_loss = task_loss + ewc_loss

        self.log("train/task_loss", task_loss)
        self.log("train/ewc_loss", ewc_loss)
        self.log("train/total_loss", total_loss)

        return total_loss

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
