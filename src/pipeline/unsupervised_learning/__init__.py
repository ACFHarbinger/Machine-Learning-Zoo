"""
Unsupervised Learning Module for Machine Learning Zoo.

This package contains Lightning modules for unsupervised learning tasks
including clustering, dimensionality reduction, and representation learning.
"""

from typing import Any, cast
import pytorch_lightning as pl
from .core import ClusteringModule, AutoencoderModule, UnsupervisedModule, VAEModule, GANModule, DiffusionModule

# Algorithm registry for unsupervised learning
ALGO_REGISTRY = {
    "clustering": ClusteringModule,
    "autoencoder": AutoencoderModule,
    "unsupervised": UnsupervisedModule,
    "vae": VAEModule,
    "gan": GANModule,
    "diffusion": DiffusionModule,
}


def create_unsupervised_model(cfg: Any) -> pl.LightningModule:
    """
    Factory function to create an unsupervised learning model.

    Args:
        cfg: Configuration object with model and training settings.

    Returns:
        pl.LightningModule: Configured unsupervised training module.
    """
    algo_name = cfg.train.algorithm if hasattr(cfg.train, "algorithm") else "autoencoder"

    if algo_name not in ALGO_REGISTRY:
        raise ValueError(f"Unknown unsupervised algorithm: {algo_name}. Available: {list(ALGO_REGISTRY.keys())}")

    algo_cls = ALGO_REGISTRY[algo_name]
    cfg_dict = vars(cfg.train) if hasattr(cfg.train, "__dict__") else dict(cfg.train)

    return cast(pl.LightningModule, algo_cls(cfg=cfg_dict))


__all__ = [
    "ClusteringModule",
    "AutoencoderModule",
    "ALGO_REGISTRY",
    "create_unsupervised_model",
]
