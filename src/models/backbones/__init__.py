"""
Backbones Package.

Task-agnostic feature extractors that can be composed with task-specific heads.
"""

from .base import BACKBONE_REGISTRY, Backbone, BackboneConfig, register_backbone
from .conv import ConvBackbone, ConvBackboneConfig
from .hf_backbone import HuggingFaceBackbone
from .lstm import LSTMBackbone, LSTMBackboneConfig
from .mamba import MambaBackbone, MambaBackboneConfig
from .multimodal import MultiModalBackbone
from .transformer import TransformerBackbone, TransformerBackboneConfig
from .vision import HuggingFaceVisionBackbone

__all__ = [
    # Base
    "Backbone",
    "BackboneConfig",
    "BACKBONE_REGISTRY",
    "register_backbone",
    # Implementations
    "TransformerBackbone",
    "TransformerBackboneConfig",
    "LSTMBackbone",
    "LSTMBackboneConfig",
    "MambaBackbone",
    "MambaBackboneConfig",
    "ConvBackbone",
    "ConvBackboneConfig",
    "HuggingFaceBackbone",
    "HuggingFaceVisionBackbone",
    "MultiModalBackbone",
]
