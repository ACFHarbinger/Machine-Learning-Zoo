"""
MultiModal Backbone implementation.

Aligns vision features with language embeddings for multi-modal tasks.
"""

from __future__ import annotations

import logging
from typing import Any

import torch
from torch import nn

from .base import Backbone, BackboneConfig, register_backbone
from .hf_backbone import HuggingFaceBackbone
from .vision import HuggingFaceVisionBackbone

logger = logging.getLogger(__name__)


@register_backbone("multimodal")
class MultiModalBackbone(Backbone):
    """
    Backbone that combines a vision encoder and a language model.
    Typically used for Image-to-Text tasks (Image Captioning, VQA).
    """

    def __init__(self, config: BackboneConfig) -> None:
        super().__init__(config)

        # 1. Initialize Vision Backbone
        vision_config_dict = config.extra.get("vision_config")
        if not vision_config_dict:
            raise ValueError("vision_config must be provided in config.extra")

        from .base import BackboneConfig as BC

        vision_config = BC(**vision_config_dict)
        self.vision_backbone = HuggingFaceVisionBackbone(vision_config)

        # 2. Initialize Language Backbone
        llm_config_dict = config.extra.get("llm_config")
        if not llm_config_dict:
            raise ValueError("llm_config must be provided in config.extra")

        llm_config = BC(**llm_config_dict)
        self.llm_backbone = HuggingFaceBackbone(llm_config)

        # 3. Projection Layer (Adapter)
        # Maps vision output_dim to llm hidden_size (or output_dim if used as input)
        # Usually LLMs take embeddings of size 'hidden_size'
        llm_dim = self.llm_backbone.output_dim
        vision_dim = self.vision_backbone.output_dim

        logger.info(
            "Initializing MultiModal projection layer: Vision(%d) -> LLM(%d)",
            vision_dim,
            llm_dim,
        )

        # Simple linear projection as a starting point.
        # More complex adapters (MLP, ResNet, etc.) can be added here.
        self.projection = nn.Linear(vision_dim, llm_dim)

        # Initialize bias and weights for stability
        nn.init.normal_(self.projection.weight, std=0.02)
        nn.init.zeros_(self.projection.bias)

    @property
    def model(self) -> nn.Module:
        """The composition of multimodal components."""
        return nn.ModuleDict(
            {
                "vision": self.vision_backbone,
                "llm": self.llm_backbone,
                "projection": self.projection,
            }
        )

    def forward(
        self,
        pixel_values: torch.Tensor | None = None,
        input_ids: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """
        Multi-modal forward pass.
        Currently focused on extracting fused features or preparing inputs for generation.

        Args:
            pixel_values: Image tensors
            input_ids: Text token tensors
            **kwargs: additional args for vision/llm backbones

        Returns:
            The output of the LLM given the fused inputs.
        """
        # 1. Process images
        if pixel_values is not None:
            vision_features = self.vision_backbone(pixel_values, **kwargs)
            # Project vision features to LLM space
            projected_vision = self.projection(vision_features)
            # projected_vision shape: [batch, vision_dim] -> [batch, llm_dim]
            # If vision_backbone returned spatial features [batch, seq, dim],
            # projection should handle it if correctly configured.

            # For simplicity, we assume projected_vision is [batch, num_patches, llm_dim]
            # or [batch, llm_dim]. If [batch, llm_dim], we unsqueeze to [batch, 1, llm_dim]
            if projected_vision.dim() == 2:
                projected_vision = projected_vision.unsqueeze(1)
        else:
            projected_vision = None

        # 2. Process text
        if input_ids is not None:
            # We don't call llm_backbone.forward because it usually returns pooled state.
            # For multimodal generative models, we often need the full embeddings.
            # Using self.llm_backbone.model.get_input_embeddings() might be better.
            llm_model = self.llm_backbone.model
            text_embeddings = llm_model.get_input_embeddings()(input_ids)

            if projected_vision is not None:
                # Concatenate [vision, text] or [text_prefix, vision, text_suffix]
                # Here we do [vision, text]
                inputs_embeds = torch.cat([projected_vision, text_embeddings], dim=1)
            else:
                inputs_embeds = text_embeddings
        else:
            inputs_embeds = projected_vision

        # 3. LLM pass
        # Note: most LLMs accept inputs_embeds instead of input_ids
        outputs = self.llm_backbone.model(inputs_embeds=inputs_embeds, **kwargs)

        # Return last hidden state or logits depending on task
        return outputs.last_hidden_state

    @property
    def output_dim(self) -> int:
        return self.llm_backbone.output_dim

    def generate(
        self,
        pixel_values: torch.Tensor | None = None,
        input_ids: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> Any:
        """Generation logic for multimodal model."""
        # For true multimodal generation, we need to pass inputs_embeds to model.generate
        # This is supported by most Transformers LLMs.

        inputs_embeds = None

        if pixel_values is not None:
            vision_features = self.vision_backbone(pixel_values)
            projected_vision = self.projection(vision_features)
            if projected_vision.dim() == 2:
                projected_vision = projected_vision.unsqueeze(1)
            inputs_embeds = projected_vision

        if input_ids is not None:
            text_embeddings = self.llm_backbone.model.get_input_embeddings()(input_ids)
            if inputs_embeds is not None:
                inputs_embeds = torch.cat([inputs_embeds, text_embeddings], dim=1)
            else:
                inputs_embeds = text_embeddings

        return self.llm_backbone.model.generate(inputs_embeds=inputs_embeds, **kwargs)
