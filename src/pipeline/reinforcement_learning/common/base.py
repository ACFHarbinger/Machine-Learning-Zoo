"""
PyTorch Lightning base module for RL training.

Inspired by RL4COLitModule from WSmartPlus-Route.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Union, cast

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


class RLBaseModule(pl.LightningModule, ABC):
    """
    Base PyTorch Lightning module for RL training.

    This module handles:
    - Training/validation/test loops
    - Optimizer configuration
    - Data loading
    - Metric logging

    Subclasses must implement `calculate_loss()` for algorithm-specific loss computation.
    """

    def __init__(
        self,
        env: Any,
        policy: torch.nn.Module,
        baseline: Optional[str] = "rollout",
        optimizer: str = "adam",
        optimizer_kwargs: Optional[dict] = None,
        lr_scheduler: Optional[str] = None,
        lr_scheduler_kwargs: Optional[dict] = None,
        train_data_size: int = 100000,
        val_data_size: int = 10000,
        batch_size: int = 256,
        num_workers: int = 4,
        **kwargs,
    ):
        """
        Initialize the RL base module.

        Args:
            env: The RL environment.
            policy: The neural network policy.
            baseline: Type of baseline for variance reduction.
            optimizer: Optimizer name ('adam' or 'adamw').
            optimizer_kwargs: Keyword arguments for the optimizer.
            lr_scheduler: Learning rate scheduler name.
            lr_scheduler_kwargs: Keyword arguments for the scheduler.
            train_data_size: Number of training samples per epoch.
            val_data_size: Number of validation samples.
            batch_size: Batch size for training.
            num_workers: Number of workers for data loading.
            **kwargs: Additional keyword arguments.
        """
        super().__init__()
        self.save_hyperparameters(ignore=["env", "policy"])

        self.env = env
        self.policy = policy
        self.baseline_type = baseline
        self.train_dataset: Optional[Any] = None
        self.val_dataset: Optional[Any] = None

        # Data params
        self.train_data_size = train_data_size
        self.val_data_size = val_data_size
        self.batch_size = batch_size
        self.num_workers = num_workers

        # Optimizer params
        self.optimizer_name = optimizer
        self.optimizer_kwargs = optimizer_kwargs or {"lr": 1e-4}
        self.lr_scheduler_name = lr_scheduler
        self.lr_scheduler_kwargs = lr_scheduler_kwargs or {}

        # Initialize baseline
        self._init_baseline()

    def _init_baseline(self):
        """Initialize baseline for advantage estimation."""
        from .baselines import WarmupBaseline, get_baseline

        if self.baseline_type is None:
            self.baseline_type = "rollout"

        baseline = get_baseline(self.baseline_type, self.policy, **self.hparams)

        # Handle warmup
        warmup_epochs = self.hparams.get("bl_warmup_epochs", 0)
        if warmup_epochs > 0:
            baseline = WarmupBaseline(baseline, warmup_epochs)

        self.baseline = baseline

    @abstractmethod
    def calculate_loss(
        self,
        td: Any,
        out: dict,
        batch_idx: int,
        env: Optional[Any] = None,
    ) -> torch.Tensor:
        """
        Compute algorithm-specific RL loss.

        Args:
            td: TensorDict with environment state.
            out: Policy output dictionary.
            batch_idx: Current batch index.
            env: Environment (optional).

        Returns:
            Loss tensor.
        """
        raise NotImplementedError

    def shared_step(
        self,
        batch: Any,
        batch_idx: int,
        phase: str,
    ) -> dict:
        """
        Common step for train/val/test.

        Args:
            batch: Input batch.
            batch_idx: Batch index.
            phase: One of "train", "val", "test".

        Returns:
            Output dictionary with loss, reward, etc.
        """
        # Unwrap batch if from baseline dataset
        batch, baseline_val = self.baseline.unwrap_batch(batch)

        # Move to device
        if hasattr(batch, "to"):
            batch = batch.to(self.device)

        if baseline_val is not None:
            baseline_val = baseline_val.to(self.device)
        self._current_baseline_val = baseline_val

        # Reset environment
        td = self.env.reset(batch)

        # Run policy
        out = self.policy(
            td,
            self.env,
            decode_type="sampling" if phase == "train" else "greedy",
        )

        # Compute loss for training
        if phase == "train":
            out["loss"] = self.calculate_loss(td, out, batch_idx, env=self.env)

        # Log metrics
        reward_mean = out["reward"].mean()
        batch_size = out["reward"].shape[0]

        self.log(
            f"{phase}/reward",
            reward_mean,
            prog_bar=True,
            sync_dist=True,
            batch_size=batch_size,
        )

        return out

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        """Execute a single training step."""
        out = self.shared_step(batch, batch_idx, phase="train")
        return out["loss"]

    def validation_step(self, batch: Any, batch_idx: int) -> dict:
        """Execute a single validation step."""
        return self.shared_step(batch, batch_idx, phase="val")

    def test_step(self, batch: Any, batch_idx: int) -> dict:
        """Execute a single test step."""
        return self.shared_step(batch, batch_idx, phase="test")

    def on_train_epoch_end(self):
        """Update baseline at end of epoch."""
        if hasattr(self.baseline, "epoch_callback"):
            self.baseline.epoch_callback(
                self.policy,
                self.current_epoch,
                val_dataset=self.val_dataset,
                env=self.env,
            )

    def configure_optimizers(self) -> Any:
        """Configure optimizer and optional scheduler."""
        if self.optimizer_name.lower() == "adam":
            optimizer = torch.optim.Adam(self.policy.parameters(), **self.optimizer_kwargs)
        elif self.optimizer_name.lower() == "adamw":
            optimizer = torch.optim.AdamW(self.policy.parameters(), **self.optimizer_kwargs)
        else:
            raise ValueError(f"Unknown optimizer: {self.optimizer_name}")

        if self.lr_scheduler_name is None:
            return optimizer

        # Configure scheduler
        if self.lr_scheduler_name.lower() == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, **self.lr_scheduler_kwargs)
        elif self.lr_scheduler_name.lower() == "step":
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, **self.lr_scheduler_kwargs)
        else:
            raise ValueError(f"Unknown scheduler: {self.lr_scheduler_name}")

        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def train_dataloader(self) -> DataLoader:
        """Create the training DataLoader."""
        return DataLoader(
            cast(Any, self.train_dataset),
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self) -> DataLoader:
        """Create the validation DataLoader."""
        return DataLoader(
            cast(Any, self.val_dataset),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )
