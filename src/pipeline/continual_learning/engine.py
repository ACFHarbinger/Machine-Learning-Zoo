"""Continual learning strategies to prevent catastrophic forgetting."""

import logging
from typing import Any, Dict, List

import torch
from pytorch_lightning import Callback, LightningModule, Trainer

logger = logging.getLogger(__name__)


class EWCCallback(Callback):
    """
    Elastic Weight Consolidation (EWC) callback.
    Prevents catastrophic forgetting by penalizing changes to weights that were
    important for previous tasks.
    """

    def __init__(self, ewc_lambda: float = 0.4):
        """
        Initialize EWC callback.
        Args:
            ewc_lambda: Importance of the EWC penalty.
        """
        self.ewc_lambda = ewc_lambda
        self.fisher_matrices: List[Dict[str, torch.Tensor]] = []
        self.optimal_weights: List[Dict[str, torch.Tensor]] = []

    def on_train_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Initialize EWC penalty if previous task data exists."""
        if not self.fisher_matrices:
            return

        logger.info(
            "Enabling EWC penalty for %d previous tasks.", len(self.fisher_matrices)
        )

    def on_after_backward(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Apply EWC penalty to gradients or directly to loss?
        Better to add to loss in training_step, but we can also manually
        adjust gradients here if needed. Standard EWC adds to loss.
        Since we can't easily modify the loss here without returning it,
        we'll expect the LightningModule to cooperate or we add it to grads.
        """
        pass

    def compute_fisher(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        dataloader: torch.utils.data.DataLoader,
    ) -> None:
        """
        Compute the Fisher Information Matrix and save optimal weights after a task is finished.
        """
        logger.info("Computing Fisher Information Matrix for the current task...")
        pl_module.eval()
        fisher = {}
        params = {n: p for n, p in pl_module.named_parameters() if p.requires_grad}

        for name, param in params.items():
            fisher[name] = torch.zeros_like(param)

        # Use a subset of data to estimate Fisher
        count = 0
        device = next(pl_module.parameters()).device

        for batch in dataloader:
            pl_module.zero_grad()
            # Handle different batch formats
            if isinstance(batch, (list, tuple)):
                outputs = pl_module(batch[0].to(device))
            elif isinstance(batch, dict):
                outputs = pl_module(**{k: v.to(device) for k, v in batch.items()})
            else:
                outputs = pl_module(batch.to(device))

            # Simple heuristic: use log-likelihood (or squared gradients)
            # For generative models, we might need more specific logic
            if hasattr(outputs, "loss"):
                loss = outputs.loss
            elif isinstance(outputs, torch.Tensor):
                loss = outputs.mean()
            else:
                continue

            loss.backward()

            for name, param in params.items():
                if param.grad is not None:
                    fisher[name] += param.grad.data**2

            count += 1
            if count >= 100:  # Limit samples for speed
                break

        for name in fisher:
            fisher[name] /= count

        self.fisher_matrices.append(fisher)
        self.optimal_weights.append({n: p.data.clone() for n, p in params.items()})
        logger.info("Fisher Information Matrix computed and stored.")

    def get_ewc_loss(self, pl_module: LightningModule) -> torch.Tensor:
        """Calculate the EWC penalty."""
        loss = 0
        if not self.fisher_matrices:
            return torch.tensor(0.0, device=next(pl_module.parameters()).device)

        params = {n: p for n, p in pl_module.named_parameters() if p.requires_grad}

        for i in range(len(self.fisher_matrices)):
            fisher = self.fisher_matrices[i]
            opt_weights = self.optimal_weights[i]

            for name, param in params.items():
                if name in fisher:
                    loss += (fisher[name] * (param - opt_weights[name]) ** 2).sum()

        return self.ewc_lambda * loss


class ReplayBuffer:
    """
    Simple experience replay buffer for storing samples from previous tasks.
    """

    def __init__(self, capacity: int = 1000):
        self.capacity = capacity
        self.buffer: List[Any] = []

    def add_samples(self, samples: List[Any]) -> None:
        """Add new samples to the buffer, maintaining capacity."""
        self.buffer.extend(samples)
        if len(self.buffer) > self.capacity:
            # Simple random replacement or FIFO? FIFO for now.
            self.buffer = self.buffer[-self.capacity :]

    def sample(self, count: int) -> List[Any]:
        """Get a random sample from the buffer."""
        import random

        if not self.buffer:
            return []
        return random.sample(self.buffer, min(count, len(self.buffer)))
