"""
Continual Learning Module.

Provides factories for creating models with Continual Learning capabilities
(e.g., EWC, Replay Buffers).
"""

from typing import Any
import pytorch_lightning as pl
from .core import ContinualLearningModule


def create_continual_learning_model(cfg: Any) -> pl.LightningModule:
    """
    Create a Continual Learning model.

    Args:
        cfg: Configuration.

    Returns:
        pl.LightningModule: Configured ContinualLearningModule.
    """
    from ..supervised_learning import create_supervised_model

    sl_module = create_supervised_model(cfg)
    backbone = getattr(sl_module, "model", sl_module)

    return ContinualLearningModule(backbone=backbone, cfg=vars(cfg) if not isinstance(cfg, dict) else cfg)
