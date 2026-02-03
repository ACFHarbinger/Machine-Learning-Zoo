from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import pytorch_lightning as pl
import torch

__all__ = ["BaseCallback", "BaseEvaluator", "BasePipeline", "BaseTrainer", "BaseModule"]


class BasePipeline(ABC):
    """Base class for all pipelines."""

    @abstractmethod
    def run(self, **kwargs: Any) -> Any:
        """Run the pipeline."""
        pass


class BaseTrainer(BasePipeline):
    """Base trainer interface."""

    @abstractmethod
    def train(self, **kwargs: Any) -> Any:
        """Train the model."""
        pass

    def run(self, **kwargs: Any) -> Any:
        return self.train(**kwargs)


class BaseEvaluator(BasePipeline):
    """Base evaluator interface."""

    @abstractmethod
    def evaluate(self, **kwargs: Any) -> Any:
        """Evaluate the model."""
        pass

    def run(self, **kwargs: Any) -> Any:
        return self.evaluate(**kwargs)


class BaseCallback(ABC):
    """Base class for all training callbacks."""

    def on_train_begin(self, **kwargs: Any) -> None:
        """Called when training begins."""
        pass

    def on_train_end(self, **kwargs: Any) -> None:
        """Called when training ends."""
        pass

    def on_epoch_begin(self, epoch: int, **kwargs: Any) -> None:
        """Called when an epoch begins."""
        pass

    def on_epoch_end(self, epoch: int, metrics: dict[str, float], **kwargs: Any) -> None:
        """Called when an epoch ends."""
        pass

    def on_batch_begin(self, batch_idx: int, **kwargs: Any) -> None:
        """Called when a batch begins."""
        pass

    def on_batch_end(self, batch_idx: int, metrics: dict[str, float], **kwargs: Any) -> None:
        """Called when a batch ends."""
        pass


class BaseModule(pl.LightningModule):
    """
    Base LightningModule with shared functionality for logging and configuration.
    """

    def __init__(self, cfg: dict[str, Any] | None = None) -> None:
        """
        Initialize the base module.

        Args:
            cfg (Dict[str, Any], optional): Configuration dictionary containing learning rate, etc.
        """
        super().__init__()
        self.save_hyperparameters()
        # Ensure cfg is set and accessible
        self.cfg = cfg or {}
        self.learning_rate = float(self.cfg.get("learning_rate", 1e-3))

    def configure_optimizers(self) -> Any:
        """
        Configure the default Adam optimizer.

        Returns:
            Any: Optimizer or dict containing optimizer and scheduler.
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def training_step(self, batch: Any, batch_idx: int) -> Any:
        """
        Abstract training step.
        """
        raise NotImplementedError

    def validation_step(self, batch: Any, batch_idx: int) -> Any:
        """
        Abstract validation step.
        """
        raise NotImplementedError
