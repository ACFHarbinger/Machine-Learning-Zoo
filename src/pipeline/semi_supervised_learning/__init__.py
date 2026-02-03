"""
Semi-Supervised Learning Module for Machine Learning Zoo.

This package contains Lightning modules for semi-supervised learning
using both labeled and unlabeled data.
"""

from typing import Any, cast
import pytorch_lightning as pl
from .core import PseudoLabelingModule, MixMatchModule, SemiSupervisedModule

# Algorithm registry for semi-supervised learning
ALGO_REGISTRY = {
    "pseudo_labeling": PseudoLabelingModule,
    "mixmatch": MixMatchModule,
    "semi_supervised": SemiSupervisedModule,
}


def create_semi_supervised_model(cfg: Any) -> pl.LightningModule:
    """
    Factory function to create a semi-supervised learning model.

    Args:
        cfg: Configuration object with model and training settings.

    Returns:
        pl.LightningModule: Configured semi-supervised training module.
    """
    algo_name = cfg.train.algorithm if hasattr(cfg.train, "algorithm") else "pseudo_labeling"

    if algo_name not in ALGO_REGISTRY:
        raise ValueError(f"Unknown semi-supervised algorithm: {algo_name}. Available: {list(ALGO_REGISTRY.keys())}")

    algo_cls = ALGO_REGISTRY[algo_name]
    cfg_dict = vars(cfg.train) if hasattr(cfg.train, "__dict__") else dict(cfg.train)

    return cast(pl.LightningModule, algo_cls(cfg=cfg_dict))


__all__ = [
    "PseudoLabelingModule",
    "MixMatchModule",
    "ALGO_REGISTRY",
    "create_semi_supervised_model",
]
