"""
Supervised Learning Module for Machine Learning Zoo.

This package contains Lightning modules for supervised learning tasks
including regression, classification, and time series forecasting.
"""

from typing import Any, cast
import pytorch_lightning as pl
from .core import RegressionModule, ClassificationModule, SLLightningModule, PiLightningModule
from .common.losses import LOSS_REGISTRY, get_loss

# Algorithm registry for supervised learning
ALGO_REGISTRY = {
    "regression": RegressionModule,
    "classification": ClassificationModule,
    "sl": SLLightningModule,
    "llm": PiLightningModule,
}


def create_supervised_model(cfg: Any) -> pl.LightningModule:
    """
    Factory function to create a supervised learning model.

    Args:
        cfg: Configuration object with model and training settings.

    Returns:
        pl.LightningModule: Configured supervised training module.
    """
    algo_name = cfg.train.algorithm if hasattr(cfg.train, "algorithm") else "regression"

    if algo_name not in ALGO_REGISTRY:
        raise ValueError(f"Unknown supervised algorithm: {algo_name}. Available: {list(ALGO_REGISTRY.keys())}")

    algo_cls = ALGO_REGISTRY[algo_name]

    common_kwargs = {
        "backbone": None,  # Will be configured based on cfg.model
        "cfg": vars(cfg.train) if hasattr(cfg.train, "__dict__") else dict(cfg.train),
    }

    return cast(pl.LightningModule, algo_cls(**common_kwargs))


__all__ = [
    "RegressionModule",
    "ClassificationModule",
    "LOSS_REGISTRY",
    "get_loss",
    "ALGO_REGISTRY",
    "create_supervised_model",
]
