"""
Synthetic Data Generation Module.
Includes GAN and VAE implementations for data augmentation.
"""

import torch
from torch import nn
from typing import Tuple, List, Optional


class GANGenerator(nn.Module):
    """
    Simple GAN Generator for tabular/time-series data.
    """

    def __init__(self, latent_dim: int, output_dim: int, hidden_dims: List[int]):
        super().__init__()
        layers = []
        curr_dim = latent_dim
        for h_dim in hidden_dims:
            layers.extend(
                [nn.Linear(curr_dim, h_dim), nn.ReLU(), nn.BatchNorm1d(h_dim)]
            )
            curr_dim = h_dim
        layers.append(nn.Linear(curr_dim, output_dim))
        self.main = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.main(z)


class GANDiscriminator(nn.Module):
    """
    Simple GAN Discriminator.
    """

    def __init__(self, input_dim: int, hidden_dims: List[int]):
        super().__init__()
        layers = []
        curr_dim = input_dim
        for h_dim in hidden_dims:
            layers.extend(
                [nn.Linear(curr_dim, h_dim), nn.LeakyReLU(0.2), nn.Dropout(0.3)]
            )
            curr_dim = h_dim
        layers.append(nn.Linear(curr_dim, 1))
        layers.append(nn.Sigmoid())
        self.main = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.main(x)


class VAEGenerator(nn.Module):
    """
    Variational Autoencoder for synthetic data generation.
    """

    def __init__(self, input_dim: int, latent_dim: int, hidden_dims: List[int]):
        super().__init__()
        # Encoder
        enc_layers = []
        curr_dim = input_dim
        for h_dim in hidden_dims:
            enc_layers.extend([nn.Linear(curr_dim, h_dim), nn.ReLU()])
            curr_dim = h_dim
        self.encoder = nn.Sequential(*enc_layers)
        self.fc_mu = nn.Linear(curr_dim, latent_dim)
        self.fc_var = nn.Linear(curr_dim, latent_dim)

        # Decoder
        dec_layers = []
        curr_dim = latent_dim
        for h_dim in reversed(hidden_dims):
            dec_layers.extend([nn.Linear(curr_dim, h_dim), nn.ReLU()])
            curr_dim = h_dim
        dec_layers.append(nn.Linear(curr_dim, input_dim))
        self.decoder = nn.Sequential(*dec_layers)

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_var(h)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    def sample(self, num_samples: int, device: str = "cpu") -> torch.Tensor:
        z = torch.randn(num_samples, self.fc_mu.out_features).to(device)
        return self.decode(z)
