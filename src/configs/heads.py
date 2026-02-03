"""
Head configurations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass
class HeadConfig:
    """Configuration for head modules."""

    input_dim: int = 256  # Must match backbone.output_dim
    output_dim: int = 10  # Task-specific (num_classes, action_dim, etc.)
    hidden_dims: tuple[int, ...] = ()  # Optional MLP layers
    dropout: float = 0.1
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class ClassificationHeadConfig(HeadConfig):
    """Configuration for classification head."""

    num_classes: int = 10
    pool_type: Literal["cls", "mean", "max", "none"] = "mean"
    multi_label: bool = False


@dataclass
class RegressionHeadConfig(HeadConfig):
    """Configuration for regression head."""

    output_dim: int = 1
    pool_type: Literal["mean", "last", "none"] = "last"
    output_activation: Literal["none", "sigmoid", "tanh", "softplus"] = "none"


@dataclass
class SequenceHeadConfig(HeadConfig):
    """Configuration for sequence head."""

    vocab_size: int = 50000
    tie_weights: bool = True  # Tie with encoder embeddings


@dataclass
class RLPolicyHeadConfig(HeadConfig):
    """Configuration for RL policy head."""

    action_dim: int = 4
    continuous: bool = False  # Discrete vs continuous actions
    log_std_min: float = -20.0
    log_std_max: float = 2.0
    separate_value_head: bool = True


@dataclass
class MultiAgentPolicyHeadConfig(HeadConfig):
    """Configuration for multi-agent RL policy head."""

    action_dim: int = 5
    num_agents: int = 4
    continuous: bool = False
    log_std_min: float = -20.0
    log_std_max: float = 2.0
    communication_dim: int = 0  # 0 disables communication
    shared_parameters: bool = True  # Whether agents share network weights
