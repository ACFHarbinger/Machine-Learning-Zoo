"""
Autoencoder module for unsupervised learning.

Provides variational and standard autoencoders for representation learning.
"""

from typing import Any, Dict, Optional, Tuple

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import nn


class AutoencoderModule(pl.LightningModule):
    """
    Autoencoder Module for unsupervised representation learning.

    Supports standard and variational autoencoder objectives.
    """

    def __init__(
        self,
        encoder: Optional[nn.Module] = None,
        decoder: Optional[nn.Module] = None,
        cfg: Optional[Dict[str, Any]] = None,
        latent_dim: int = 64,
        variational: bool = False,
        beta: float = 1.0,
        learning_rate: float = 1e-3,
        **kwargs,
    ):
        """
        Initialize the autoencoder module.

        Args:
            encoder: Encoder network.
            decoder: Decoder network.
            cfg: Configuration dictionary.
            latent_dim: Latent space dimension.
            variational: Whether to use VAE objective.
            beta: Beta weight for KL term in VAE.
            learning_rate: Learning rate for optimizer.
        """
        super().__init__()
        cfg = cfg or {}
        self.save_hyperparameters(ignore=["encoder", "decoder"])

        self.encoder = encoder
        self.decoder = decoder
        self.latent_dim = latent_dim
        self.variational = variational
        self.beta = beta
        self.learning_rate = learning_rate

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to latent space."""
        if self.encoder is not None:
            return self.encoder(x)
        return x

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent to reconstruction."""
        if self.decoder is not None:
            return self.decoder(z)
        return z

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: encode then decode."""
        z = self.encode(x)
        return self.decode(z)

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        """Training step with reconstruction loss."""
        if isinstance(batch, dict):
            x = batch["observation"]
        else:
            x = batch[0] if isinstance(batch, (list, tuple)) else batch

        # Reconstruction
        x_recon = self(x)
        recon_loss = F.mse_loss(x_recon, x)

        loss = recon_loss

        self.log("train/recon_loss", recon_loss, prog_bar=True)
        self.log("train/loss", loss)
        return loss

    def configure_optimizers(self):
        """Configure optimizer."""
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
