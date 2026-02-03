"""
Active Learning Module.

Provides factories for creating models suitable for active learning loops
(e.g., with MC Dropout enabled).
"""

from typing import Any
import pytorch_lightning as pl
from .core import ActiveLearningModule


def create_active_learning_model(cfg: Any) -> pl.LightningModule:
    """
    Create an Active Learning model.

    Args:
        cfg: Configuration.

    Returns:
        pl.LightningModule: Configured ActiveLearningModule.
    """
    from ..supervised_learning import create_supervised_model

    # Create the backbone using standard supervised factory or valid approach
    # Here we assume create_supervised_model returns a wrapper, but we want the inner backbone
    # Or more simply, we assume cfg defines the model architecture.

    # We'll re-use the supervised factory to get the BACKBONE if possible,
    # but `create_supervised_model` returns a PL module.
    # We might need to extract the backbone.

    sl_module = create_supervised_model(cfg)
    # Assuming sl_module has a 'model' or 'backbone' attribute, or IS the model.
    # BaseModule usually has self.model?? Actually SLLightningModule likely has self.model.

    backbone = getattr(sl_module, "model", sl_module)

    return ActiveLearningModule(backbone=backbone, cfg=vars(cfg) if not isinstance(cfg, dict) else cfg)
