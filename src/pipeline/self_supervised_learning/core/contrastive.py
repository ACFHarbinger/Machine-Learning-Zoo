"""
Contrastive learning module for self-supervised learning.

Generic contrastive learning with InfoNCE loss.
"""

from typing import Any, Dict, Optional, cast

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import nn


class ContrastiveModule(pl.LightningModule):
    """
    Contrastive Learning Module.

    Learns representations by maximizing agreement between
    positive pairs and minimizing agreement with negatives.
    """

    def __init__(
        self,
        encoder: Optional[nn.Module] = None,
        cfg: Optional[Dict[str, Any]] = None,
        hidden_dim: int = 128,
        projection_dim: int = 64,
        temperature: float = 0.07,
        learning_rate: float = 1e-3,
        **kwargs,
    ):
        """
        Initialize the contrastive module.

        Args:
            encoder: Encoder network for feature extraction.
            cfg: Configuration dictionary.
            hidden_dim: Hidden dimension from encoder.
            projection_dim: Dimension of projection head output.
            temperature: Temperature for InfoNCE loss.
            learning_rate: Learning rate for optimizer.
        """
        super().__init__()
        cfg = cfg or {}
        self.save_hyperparameters(ignore=["encoder"])

        self.encoder = encoder
        self.temperature = temperature
        self.learning_rate = learning_rate

        # Projection head
        self.projector = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, projection_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode and project."""
        if self.encoder is not None:
            h = self.encoder(x)
        else:
            h = x
        return cast(torch.Tensor, self.projector(h))

    def info_nce_loss(self, z_i: torch.Tensor, z_j: torch.Tensor) -> torch.Tensor:
        """Compute InfoNCE loss for positive pairs."""
        batch_size = z_i.shape[0]

        # Normalize embeddings
        z_i = F.normalize(z_i, dim=-1)
        z_j = F.normalize(z_j, dim=-1)

        # Concatenate representations
        representations = torch.cat([z_i, z_j], dim=0)

        # Compute similarity matrix
        similarity = torch.matmul(representations, representations.T) / self.temperature

        # Create labels for positive pairs
        labels = torch.cat([torch.arange(batch_size) + batch_size, torch.arange(batch_size)], dim=0)
        labels = labels.to(similarity.device)

        # Mask out self-similarities
        mask = torch.eye(2 * batch_size, device=similarity.device).bool()
        similarity = similarity.masked_fill(mask, float("-inf"))

        return F.cross_entropy(similarity, labels)

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        """Training step with contrastive loss."""
        # Expect two augmented views
        if isinstance(batch, dict):
            x_i = batch["view1"]
            x_j = batch["view2"]
        elif isinstance(batch, (list, tuple)) and len(batch) >= 2:
            x_i, x_j = batch[0], batch[1]
        else:
            # Single view: create dummy pair (not recommended)
            x_i = x_j = batch

        z_i = self(x_i)
        z_j = self(x_j)

        loss = self.info_nce_loss(z_i, z_j)

        self.log("train/loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        """Configure optimizer."""
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
