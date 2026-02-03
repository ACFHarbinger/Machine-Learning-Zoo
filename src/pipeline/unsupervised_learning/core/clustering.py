"""
Clustering module for unsupervised learning.

Provides deep clustering with learned representations.
"""

from typing import Any, Dict, Optional

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import nn


class ClusteringModule(pl.LightningModule):
    """
    Deep Clustering Module.

    Combines representation learning with clustering objective.
    """

    def __init__(
        self,
        encoder: Optional[nn.Module] = None,
        cfg: Optional[Dict[str, Any]] = None,
        n_clusters: int = 10,
        hidden_dim: int = 128,
        learning_rate: float = 1e-3,
        **kwargs,
    ):
        """
        Initialize the clustering module.

        Args:
            encoder: Encoder network for feature extraction.
            cfg: Configuration dictionary.
            n_clusters: Number of clusters.
            hidden_dim: Hidden dimension for embeddings.
            learning_rate: Learning rate for optimizer.
        """
        super().__init__()
        cfg = cfg or {}
        self.save_hyperparameters(ignore=["encoder"])

        self.encoder = encoder
        self.n_clusters = n_clusters
        self.learning_rate = learning_rate

        # Cluster centers (learnable)
        self.cluster_centers = nn.Parameter(torch.randn(n_clusters, hidden_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Get cluster assignments."""
        if self.encoder is not None:
            z = self.encoder(x)
        else:
            z = x

        # Compute distances to cluster centers
        distances = torch.cdist(z, self.cluster_centers)
        return F.softmax(-distances, dim=-1)

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        """Training step with clustering loss."""
        if isinstance(batch, dict):
            x = batch["observation"]
        else:
            x = batch[0] if isinstance(batch, (list, tuple)) else batch

        # Get cluster probabilities
        q = self(x)

        # Target distribution (sharpened)
        p = self._target_distribution(q)

        # KL divergence loss
        loss = F.kl_div(q.log(), p, reduction="batchmean")

        self.log("train/loss", loss, prog_bar=True)
        return loss

    def _target_distribution(self, q: torch.Tensor) -> torch.Tensor:
        """Compute target distribution for self-training."""
        weight = q**2 / q.sum(0)
        return (weight.t() / weight.sum(1)).t()

    def configure_optimizers(self):
        """Configure optimizer."""
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
