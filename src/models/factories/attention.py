from __future__ import annotations

from typing import Any

from torch import nn

from ..attention.attention_net import AttentionNetwork
from ..attention.nstransformer import NSTransformer
from ..factories.base import NeuralComponentFactory


class AttentionFactory(NeuralComponentFactory):
    """Factory for attention mechanisms."""

    @staticmethod
    def get_component(name: str, **kwargs: Any) -> nn.Module:
        """Get attention model by name."""
        name = name.lower()
        # NSTransformer uses 'nstransformer' or 'transformer'
        if "nstransformer" in name:
            return NSTransformer(**kwargs)
        elif "attention" in name:
            return AttentionNetwork(**kwargs)
        else:
            raise ValueError(
                f"Unknown attention model: {name}. Available: nstransformer, attention"
            )
