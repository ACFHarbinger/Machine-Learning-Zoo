"""
Vision Backbones implementation.

Wraps HuggingFace vision models for feature extraction.
"""

from __future__ import annotations

import logging
from typing import Any

import torch
from torch import nn

try:
    from transformers import AutoModel, AutoProcessor
except ImportError:
    AutoModel = None
    AutoProcessor = None

from .base import Backbone, BackboneConfig, register_backbone

logger = logging.getLogger(__name__)


@register_backbone("hf_vision")
class HuggingFaceVisionBackbone(Backbone):
    """
    Backbone implementation using HuggingFace Vision models (CLIP, SigLIP, ViT).
    """

    def __init__(self, config: BackboneConfig) -> None:
        super().__init__(config)
        if AutoModel is None:
            raise ImportError(
                "transformers not installed. Please install it to use HuggingFaceVisionBackbone."
            )

        hf_repo = config.extra.get("hf_repo")
        if not hf_repo:
            raise ValueError(
                "hf_repo must be specified in config.extra for HuggingFaceVisionBackbone"
            )

        device_map = config.extra.get("device_map", "auto")
        dtype_str = config.extra.get("torch_dtype", "float16")
        torch_dtype = (
            getattr(torch, dtype_str) if isinstance(dtype_str, str) else dtype_str
        )

        logger.info("Initializing HuggingFaceVisionBackbone from %s...", hf_repo)

        self.processor = AutoProcessor.from_pretrained(hf_repo, trust_remote_code=True)
        self._model = AutoModel.from_pretrained(
            hf_repo,
            device_map=device_map,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
        )

    @property
    def model(self) -> nn.Module:
        """The underlying HuggingFace model."""
        return self._model

    def forward(
        self, x: torch.Tensor | dict[str, torch.Tensor], **kwargs: Any
    ) -> torch.Tensor:
        """
        Forward pass producing image features.

        Args:
            x: Input tensor or dictionary of tensors from processor
            **kwargs: Additional arguments for the model's forward pass

        Returns:
            Feature tensor (pooled output or last hidden state)
        """
        if isinstance(x, dict):
            outputs = self._model(**x, **kwargs)
        else:
            outputs = self._model(pixel_values=x, **kwargs)

        # For CLIP-like models, we usually want the image_embeds or pooler_output
        if hasattr(outputs, "image_embeds"):
            return outputs.image_embeds
        if hasattr(outputs, "pooler_output"):
            return outputs.pooler_output

        # Fallback to last hidden state mean pooling if no pooler output
        return outputs.last_hidden_state.mean(dim=1)

    @property
    def output_dim(self) -> int:
        """Dimensionality of the backbone's output features."""
        if hasattr(self._model.config, "projection_dim"):
            return self._model.config.projection_dim
        if hasattr(self._model.config, "hidden_size"):
            return self._model.config.hidden_size
        return self.config.hidden_dim

    @property
    def device(self) -> torch.device:
        """The device the model is on."""
        return self._model.device

    def preprocess(
        self, images: Any, return_tensors: str = "pt"
    ) -> dict[str, torch.Tensor]:
        """Preprocess images using the model's processor."""
        return self.processor(images=images, return_tensors=return_tensors).to(
            self.device
        )
