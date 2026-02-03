"""Common utilities for supervised learning."""

from .losses import LOSS_REGISTRY, get_loss

__all__ = [
    "LOSS_REGISTRY",
    "get_loss",
]
