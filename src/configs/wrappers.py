"""
Wrapper configurations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class NeuroSymbolicConfig:
    """Configuration for neuro-symbolic models."""

    input_dim: int = 256
    hidden_dim: int = 128
    output_dim: int = 10
    num_rules: int = 64
    rule_dim: int = 64
    num_predicates: int = 32
    integration_mode: str = "gated"  # "gated", "residual", or "attention"
    symbolic_depth: int = 3
    dropout: float = 0.1
    temperature: float = 1.0
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class ComposedModelConfig:
    """Configuration for composed models."""

    backbone: str  # Name in BACKBONE_REGISTRY
    head: str  # Name in HEAD_REGISTRY
    backbone_config: dict[str, Any]
    head_config: dict[str, Any]
    freeze_backbone: bool = False
