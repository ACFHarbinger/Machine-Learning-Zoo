"""
Federated Learning Module.

Provides factories for Federated Learning client models.
"""

from typing import Any
import pytorch_lightning as pl
from .core import FederatedClientModule
from .engine import FederatedAggregator, FederatedClient

__all__ = ["FederatedAggregator", "FederatedClient", "FederatedClientModule", "create_federated_learning_model"]


def create_federated_learning_model(cfg: Any) -> pl.LightningModule:
    """
    Create a Federated Learning client model.

    Args:
        cfg: Configuration.

    Returns:
        pl.LightningModule: Configured FederatedClientModule.
    """
    from ..supervised_learning import create_supervised_model

    sl_module = create_supervised_model(cfg)
    backbone = getattr(sl_module, "model", sl_module)

    return FederatedClientModule(backbone=backbone, cfg=vars(cfg) if not isinstance(cfg, dict) else cfg)
