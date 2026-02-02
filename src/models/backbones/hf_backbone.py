"""
HuggingFace Backbone Implementation.

Provides a unified wrapper for HuggingFace Transformers models,
allowing them to be used as both standalone LLMs and feature extractors.
"""

from __future__ import annotations

import logging
from typing import Any

import torch
from torch import nn
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)

from .base import Backbone, BackboneConfig, register_backbone

logger = logging.getLogger(__name__)


@register_backbone("huggingface")
class HuggingFaceBackbone(Backbone):
    """
    Backbone wrapper for HuggingFace Transformers models.
    """

    def __init__(self, config: BackboneConfig) -> None:
        """
        Initialize the HuggingFace backbone.

        Args:
            config: Backbone configuration.
                Required extra keys:
                - 'hf_repo': The HuggingFace repository ID or local path.
                - 'device_map': PyTorch device mapping (default: 'auto').
                - 'torch_dtype': PyTorch dtype (default: torch.float16).
                - 'load_in_8bit': Whether to load in 8-bit quantization.
                - 'load_in_4bit': Whether to load in 4-bit quantization.
        """
        super().__init__(config)

        hf_repo = config.extra.get("hf_repo")
        if not hf_repo:
            raise ValueError("BackboneConfig must contain 'hf_repo' in extra dict")

        device_map = config.extra.get("device_map", "auto")
        torch_dtype = config.extra.get("torch_dtype", torch.float16)
        load_in_8bit = config.extra.get("load_in_8bit", False)
        load_in_4bit = config.extra.get("load_in_4bit", False)

        logger.info("Initializing HuggingFaceBackbone from %s...", hf_repo)

        self.tokenizer = AutoTokenizer.from_pretrained(hf_repo, trust_remote_code=True)

        model_kwargs = {
            "device_map": device_map,
            "torch_dtype": torch_dtype,
            "trust_remote_code": True,
        }
        if load_in_8bit:
            model_kwargs["load_in_8bit"] = True
        if load_in_4bit:
            model_kwargs["load_in_4bit"] = True

        self._model = AutoModelForCausalLM.from_pretrained(hf_repo, **model_kwargs)

        # Ensure tokenizer has pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    @property
    def model(self) -> PreTrainedModel:
        """The underlying HuggingFace model."""
        return self._model

    def forward(self, x: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """
        Forward pass producing feature representation (last hidden states).

        Args:
            x: Input tensor (input_ids)
            **kwargs: Additional arguments for the model's forward pass

        Returns:
            Feature tensor (last hidden states)
        """
        outputs = self._model(x, output_hidden_states=True, **kwargs)
        # Return the last hidden state of the last token as the feature representation
        return outputs.hidden_states[-1][:, -1, :]

    @property
    def output_dim(self) -> int:
        """Dimensionality of the backbone's output features."""
        if hasattr(self._model.config, "hidden_size"):
            return self._model.config.hidden_size
        return self.config.hidden_dim

    @property
    def device(self) -> torch.device:
        """The device the model is on."""
        return self._model.device

    def generate(self, *args: Any, **kwargs: Any) -> Any:
        """Proxied generate method for LLM usage."""
        return self._model.generate(*args, **kwargs)
