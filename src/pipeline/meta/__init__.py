"""
Meta-Learning Module.

Provides factories for Meta-Learning models (MAML).
"""

from typing import Any
import pytorch_lightning as pl
import torch.nn as nn
from .maml import MAMLLightningModule

__all__ = ["MAMLLightningModule", "create_meta_learning_model"]


def create_meta_learning_model(cfg: Any) -> pl.LightningModule:
    """
    Create a Meta-Learning model.

    Args:
        cfg: Configuration.

    Returns:
        pl.LightningModule: Configured MAMLLightningModule.
    """
    # Placeholder inner model construction
    # In production, use `get_model` from src.models
    # For now we mock it or extract from config
    input_dim = getattr(cfg.model, "input_dim", 10)
    hidden_dim = getattr(cfg.model, "hidden_dim", 32)
    output_dim = getattr(cfg.model, "output_dim", 1)

    inner_model = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, output_dim))

    return MAMLLightningModule(model=inner_model, cfg=vars(cfg) if not isinstance(cfg, dict) else cfg)
