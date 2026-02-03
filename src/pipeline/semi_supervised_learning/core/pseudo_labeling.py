"""
Pseudo-labeling module for semi-supervised learning.

Uses confident predictions on unlabeled data as pseudo-labels.
"""

from typing import Any, Dict, Optional

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import nn


class PseudoLabelingModule(pl.LightningModule):
    """
    Pseudo-Labeling Module for semi-supervised learning.

    Uses model predictions on unlabeled data as training targets
    when confidence exceeds a threshold.
    """

    def __init__(
        self,
        backbone: Optional[nn.Module] = None,
        cfg: Optional[Dict[str, Any]] = None,
        num_classes: int = 10,
        confidence_threshold: float = 0.95,
        unlabeled_weight: float = 1.0,
        warmup_epochs: int = 0,
        learning_rate: float = 1e-3,
        **kwargs,
    ):
        """
        Initialize the pseudo-labeling module.

        Args:
            backbone: Feature extractor network.
            cfg: Configuration dictionary.
            num_classes: Number of output classes.
            confidence_threshold: Minimum confidence for pseudo-labels.
            unlabeled_weight: Weight for unlabeled loss.
            warmup_epochs: Epochs before using pseudo-labels.
            learning_rate: Learning rate for optimizer.
        """
        super().__init__()
        cfg = cfg or {}
        self.save_hyperparameters(ignore=["backbone"])

        self.backbone = backbone
        self.num_classes = num_classes
        self.confidence_threshold = confidence_threshold
        self.unlabeled_weight = unlabeled_weight
        self.warmup_epochs = warmup_epochs
        self.learning_rate = learning_rate

        # Classification head
        hidden_dim = int(cfg.get("hidden_dim", 128))
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Get class logits."""
        if self.backbone is not None:
            feat = self.backbone(x)
        else:
            feat = x
        return self.classifier(feat)

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        """Training step with labeled and pseudo-labeled data."""
        # Expect batch to contain both labeled and unlabeled data
        if isinstance(batch, dict):
            x_l, y_l = batch["labeled"]
            x_u = batch.get("unlabeled", None)
        else:
            x_l, y_l = batch
            x_u = None

        # Supervised loss on labeled data
        logits_l = self(x_l)
        loss_labeled = F.cross_entropy(logits_l, y_l)

        loss = loss_labeled
        self.log("train/loss_labeled", loss_labeled, prog_bar=True)

        # Pseudo-labeling on unlabeled data (after warmup)
        if x_u is not None and self.current_epoch >= self.warmup_epochs:
            with torch.no_grad():
                logits_u = self(x_u)
                probs_u = F.softmax(logits_u, dim=-1)
                max_probs, pseudo_labels = probs_u.max(dim=-1)

                # Filter by confidence
                mask = max_probs >= self.confidence_threshold

            if mask.sum() > 0:
                logits_u_masked = self(x_u[mask])
                loss_unlabeled = F.cross_entropy(logits_u_masked, pseudo_labels[mask])
                loss = loss + self.unlabeled_weight * loss_unlabeled
                self.log("train/loss_unlabeled", loss_unlabeled)
                self.log("train/pseudo_ratio", mask.float().mean())

        self.log("train/loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        """Configure optimizer."""
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
