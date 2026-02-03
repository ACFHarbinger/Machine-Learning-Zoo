"""
Federated Learning Core Module.

Implements client-side logic for Federated Learning.
"""

from typing import Any, Dict, cast

import torch
import torch.nn as nn

from ..base import BaseModule


class FederatedClientModule(BaseModule):
    """
    Lightning Module for Federated Learning Client.

    Provides interfaces for getting/setting weights for aggregation.
    """

    def __init__(
        self,
        backbone: nn.Module,
        cfg: dict[str, Any],
    ) -> None:
        """
        Initialize Federated Client Module.

        Args:
            backbone: The base PyTorch model.
            cfg: Configuration dictionary.
        """
        super().__init__(cfg)
        self.backbone = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return cast(torch.Tensor, self.backbone(x))

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        """Standard local training step."""
        if isinstance(batch, (list, tuple)):
            x, y = batch
        elif isinstance(batch, dict):
            x = batch["data"]
            y = batch["target"]
        else:
            raise ValueError(f"Unsupported batch type: {type(batch)}")

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

    def get_weights(self) -> Dict[str, torch.Tensor]:
        """Return state dict for aggregation."""
        return self.backbone.state_dict()

    def set_weights(self, state_dict: Dict[str, torch.Tensor]) -> None:
        """Load aggregated weights."""
        self.backbone.load_state_dict(state_dict)
