"""
Reinforcement Learning Module for Machine Learning Zoo.

This package contains Lightning modules, algorithms, and utilities
for training agents using reinforcement learning.

Core Modules:
- core/: RL algorithms (REINFORCE, PPO, variants)
- common/: Base classes, baselines, utilities
"""

from typing import Any, cast
import pytorch_lightning as pl
from .common.base import RLBaseModule
from .common.baselines import (
    BASELINE_REGISTRY,
    Baseline,
    CriticBaseline,
    ExponentialBaseline,
    NoBaseline,
    POMOBaseline,
    RolloutBaseline,
    WarmupBaseline,
    get_baseline,
)
from .core import REINFORCE, PPO, MultiAgentRLModule

# Algorithm registry for RL training
ALGO_REGISTRY = {
    "reinforce": REINFORCE,
    "ppo": PPO,
    "multi_agent_rl": MultiAgentRLModule,
}


def create_rl_model(cfg: Any) -> pl.LightningModule:
    """
    Factory function to create an RL model based on configuration.

    Args:
        cfg: Configuration object with env, model, and rl settings.

    Returns:
        pl.LightningModule: Configured RL training module.
    """
    from ...envs import get_env
    from ...models import get_policy

    # Initialize environment
    env = get_env(cfg.env.name, **vars(cfg.env))

    # Initialize policy
    policy = get_policy(cfg.model.name, **vars(cfg.model))

    # Get algorithm class
    algo_name = cfg.rl.algorithm
    if algo_name not in ALGO_REGISTRY:
        raise ValueError(f"Unknown RL algorithm: {algo_name}. Available: {list(ALGO_REGISTRY.keys())}")

    algo_cls = ALGO_REGISTRY[algo_name]

    # Build common kwargs
    common_kwargs = {
        "env": env,
        "policy": policy,
        "baseline": cfg.rl.baseline,
        "optimizer": cfg.optim.optimizer,
        "optimizer_kwargs": {"lr": cfg.optim.lr},
        "lr_scheduler": cfg.optim.lr_scheduler,
        "train_data_size": cfg.train.train_data_size,
        "val_data_size": cfg.train.val_data_size,
        "batch_size": cfg.train.batch_size,
    }

    return cast(pl.LightningModule, algo_cls(**common_kwargs))


__all__ = [
    "RLBaseModule",
    "REINFORCE",
    "Baseline",
    "NoBaseline",
    "ExponentialBaseline",
    "RolloutBaseline",
    "CriticBaseline",
    "WarmupBaseline",
    "POMOBaseline",
    "get_baseline",
    "BASELINE_REGISTRY",
    "ALGO_REGISTRY",
    "create_rl_model",
]
