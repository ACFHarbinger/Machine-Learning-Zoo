"""
Backbone configurations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass
class BackboneConfig:
    """Configuration for backbone models."""

    hidden_dim: int = 256
    num_layers: int = 4
    dropout: float = 0.1
    input_dim: int | None = None  # Set dynamically based on data
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class TransformerBackboneConfig(BackboneConfig):
    """Configuration for Transformer backbone."""

    num_heads: int = 8
    ff_dim: int = 1024
    max_seq_len: int = 512
    prenorm: bool = True


@dataclass
class ConvBackboneConfig(BackboneConfig):
    """Configuration for Convolutional backbone."""

    kernel_sizes: tuple[int, ...] = (7, 5, 3, 3)
    channels: tuple[int, ...] = (64, 128, 256, 256)
    conv_type: Literal["1d", "2d"] = "1d"
    pool_type: Literal["max", "avg"] = "max"


@dataclass
class LSTMBackboneConfig(BackboneConfig):
    """Configuration for LSTM backbone."""

    bidirectional: bool = True
    proj_size: int = 0  # Projection size (0 = disabled)


@dataclass
class MambaBackboneConfig(BackboneConfig):
    """Configuration for Mamba backbone."""

    d_state: int = 16
    d_conv: int = 4
    expand: int = 2
