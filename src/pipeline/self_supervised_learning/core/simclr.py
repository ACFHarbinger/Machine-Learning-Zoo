"""
SimCLR module for self-supervised learning.

Reference:
    Chen, T., et al. (2020). A Simple Framework for Contrastive Learning
    of Visual Representations. ICML 2020.
"""

from typing import Any, Dict, Optional, cast

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import nn


class SimCLRModule(pl.LightningModule):
    """
    SimCLR Module for self-supervised visual representation learning.

    Implements the SimCLR framework with NT-Xent loss.
    """

    def __init__(
        self,
        encoder: Optional[nn.Module] = None,
        cfg: Optional[Dict[str, Any]] = None,
        hidden_dim: int = 2048,
        projection_dim: int = 128,
        temperature: float = 0.5,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-6,
        **kwargs,
    ):
        """
        Initialize the SimCLR module.

        Args:
            encoder: Backbone encoder (e.g., ResNet).
            cfg: Configuration dictionary.
            hidden_dim: Hidden dimension from encoder.
            projection_dim: Output dimension of projection head.
            temperature: Temperature for NT-Xent loss.
            learning_rate: Learning rate for optimizer.
            weight_decay: Weight decay for optimizer.
        """
        super().__init__()
        cfg = cfg or {}
        self.save_hyperparameters(ignore=["encoder"])

        self.encoder = encoder
        self.temperature = temperature
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        # Projection head (MLP)
        self.projector = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, projection_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode and project."""
        if self.encoder is not None:
            h = self.encoder(x)
        else:
            h = x
        return cast(torch.Tensor, self.projector(h))

    def nt_xent_loss(self, z_i: torch.Tensor, z_j: torch.Tensor) -> torch.Tensor:
        """Compute normalized temperature-scaled cross entropy loss."""
        batch_size = z_i.shape[0]

        # Normalize
        z_i = F.normalize(z_i, dim=-1)
        z_j = F.normalize(z_j, dim=-1)

        # Gather all embeddings
        z = torch.cat([z_i, z_j], dim=0)

        # Compute similarities
        sim = torch.mm(z, z.T) / self.temperature

        # Positive pairs mask
        sim_i_j = torch.diag(sim, batch_size)
        sim_j_i = torch.diag(sim, -batch_size)
        positives = torch.cat([sim_i_j, sim_j_i], dim=0)

        # Mask out self-similarities
        mask = ~torch.eye(2 * batch_size, dtype=bool, device=z.device)
        negatives = sim[mask].view(2 * batch_size, -1)

        # NT-Xent loss
        logits = torch.cat([positives.unsqueeze(1), negatives], dim=1)
        labels = torch.zeros(2 * batch_size, dtype=torch.long, device=z.device)

        return F.cross_entropy(logits, labels)

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        """Training step with NT-Xent loss."""
        if isinstance(batch, dict):
            x_i, x_j = batch["view1"], batch["view2"]
        elif isinstance(batch, (list, tuple)) and len(batch) >= 2:
            x_i, x_j = batch[0], batch[1]
        else:
            x_i = x_j = batch

        z_i = self(x_i)
        z_j = self(x_j)

        loss = self.nt_xent_loss(z_i, z_j)

        self.log("train/loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        """Configure LARS optimizer (simplified as Adam here)."""
        return torch.optim.Adam(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
