"""
Loss functions for supervised learning.

Provides a registry of common loss functions for regression and classification tasks.
"""

from typing import Any, Callable, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss for imbalanced classification.

    Reference:
        Lin, T. Y., et al. (2017). Focal loss for dense object detection.
        ICCV 2017.
    """

    def __init__(self, alpha: float = 1.0, gamma: float = 2.0, reduction: str = "mean"):
        """
        Initialize Focal Loss.

        Args:
            alpha: Weighting factor for class imbalance.
            gamma: Focusing parameter for hard examples.
            reduction: Reduction method ('mean', 'sum', 'none').
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute focal loss."""
        ce_loss = F.cross_entropy(inputs, targets, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss


class HuberLoss(nn.Module):
    """Huber loss (smooth L1 loss)."""

    def __init__(self, delta: float = 1.0, reduction: str = "mean"):
        super().__init__()
        self.delta = delta
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return F.smooth_l1_loss(inputs, targets, beta=self.delta, reduction=self.reduction)


# Loss function registry
LOSS_REGISTRY: Dict[str, Callable[..., nn.Module]] = {
    "mse": lambda **kwargs: nn.MSELoss(**kwargs),
    "l1": lambda **kwargs: nn.L1Loss(**kwargs),
    "huber": lambda **kwargs: HuberLoss(**kwargs),
    "cross_entropy": lambda **kwargs: nn.CrossEntropyLoss(**kwargs),
    "bce": lambda **kwargs: nn.BCEWithLogitsLoss(**kwargs),
    "focal": lambda **kwargs: FocalLoss(**kwargs),
    "kl_div": lambda **kwargs: nn.KLDivLoss(**kwargs),
}


def get_loss(name: str, **kwargs) -> nn.Module:
    """
    Get loss function by name.

    Args:
        name: Loss function name.
        **kwargs: Additional arguments for the loss function.

    Returns:
        nn.Module: Loss function instance.

    Raises:
        ValueError: If loss name is not in registry.
    """
    if name not in LOSS_REGISTRY:
        raise ValueError(f"Unknown loss: {name}. Available: {list(LOSS_REGISTRY.keys())}")

    return LOSS_REGISTRY[name](**kwargs)
