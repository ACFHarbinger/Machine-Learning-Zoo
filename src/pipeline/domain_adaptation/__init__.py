"""
Domain Adaptation Module.

Provides factories for Domain Adaptation models (DANN, MMD).
"""

from typing import Any
import pytorch_lightning as pl
from .core import DomainAdaptationModule


def create_domain_adaptation_model(cfg: Any) -> pl.LightningModule:
    """
    Create a Domain Adaptation model.

    Args:
        cfg: Configuration.

    Returns:
        pl.LightningModule: Configured DomainAdaptationModule.
    """
    from ..supervised_learning import create_supervised_model

    sl_module = create_supervised_model(cfg)
    backbone = getattr(sl_module, "model", sl_module)

    return DomainAdaptationModule(backbone=backbone, cfg=vars(cfg) if not isinstance(cfg, dict) else cfg)
