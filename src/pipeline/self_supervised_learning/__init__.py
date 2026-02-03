"""
Self-Supervised Learning Module for Machine Learning Zoo.

This package contains Lightning modules for self-supervised learning
including contrastive learning and masked modeling.
"""

from typing import Any, cast
import pytorch_lightning as pl
from .core import ContrastiveModule, SimCLRModule, SelfSupervisedModule

# Algorithm registry for self-supervised learning
ALGO_REGISTRY = {
    "contrastive": ContrastiveModule,
    "simclr": SimCLRModule,
    "self_supervised": SelfSupervisedModule,
}


def create_self_supervised_model(cfg: Any) -> pl.LightningModule:
    """
    Factory function to create a self-supervised learning model.

    Args:
        cfg: Configuration object with model and training settings.

    Returns:
        pl.LightningModule: Configured self-supervised training module.
    """
    algo_name = cfg.train.algorithm if hasattr(cfg.train, "algorithm") else "contrastive"

    if algo_name not in ALGO_REGISTRY:
        raise ValueError(f"Unknown self-supervised algorithm: {algo_name}. Available: {list(ALGO_REGISTRY.keys())}")

    algo_cls = ALGO_REGISTRY[algo_name]
    cfg_dict = vars(cfg.train) if hasattr(cfg.train, "__dict__") else dict(cfg.train)

    return cast(pl.LightningModule, algo_cls(cfg=cfg_dict))


__all__ = [
    "ContrastiveModule",
    "SimCLRModule",
    "ALGO_REGISTRY",
    "create_self_supervised_model",
]
