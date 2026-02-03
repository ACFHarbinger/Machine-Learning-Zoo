"""
Online Learning Module.

Provides factories for Online Learning models.
"""

from typing import Any
import pytorch_lightning as pl
import torch.nn as nn
from .core import OnlineLearningModule
from .online_trainer import OnlineTrainer

__all__ = ["OnlineTrainer", "OnlineLearningModule", "create_online_learning_model"]


def create_online_learning_model(cfg: Any) -> pl.LightningModule:
    """
    Create an Online Learning model.

    Args:
        cfg: Configuration.

    Returns:
        pl.LightningModule: Configured OnlineLearningModule.
    """
    # Placeholder backbone
    input_dim = getattr(cfg.model, "input_dim", 10)
    output_dim = getattr(cfg.model, "output_dim", 1)

    backbone = nn.Linear(input_dim, output_dim)

    return OnlineLearningModule(backbone=backbone, cfg=vars(cfg) if not isinstance(cfg, dict) else cfg)
