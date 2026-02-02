"""Module for model explainability and visualization."""

import torch
import torch.nn as nn
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np


class ExplainabilityModule:
    """
    Utilities for interpreting model decisions.
    """

    @staticmethod
    def integrated_gradients(
        model: nn.Module,
        inputs: torch.Tensor,
        target_idx: int,
        baseline: Optional[torch.Tensor] = None,
        steps: int = 50,
    ) -> torch.Tensor:
        """
        Compute Integrated Gradients for a given input and target.

        Args:
            model: The neural network.
            inputs: Input tensor (e.g., embeddings or images).
            target_idx: The index of the target class/token.
            baseline: Baseline input (defaults to zeros).
            steps: Number of interpolation steps.

        Returns:
            torch.Tensor: Attributions with the same shape as inputs.
        """
        if baseline is None:
            baseline = torch.zeros_like(inputs)

        # Generate interpolated inputs
        alphas = torch.linspace(0, 1, steps).to(inputs.device)
        interpolated_inputs = [
            baseline + alpha * (inputs - baseline) for alpha in alphas
        ]

        grads = []
        for x in interpolated_inputs:
            x.requires_grad_(True)
            output = model(x)

            # If output is (batch, seq, vocab), we might need to handle target_idx carefully
            # For simplicity, assume output is (batch, num_classes) or handle logits
            if len(output.shape) == 3:  # (batch, seq, vocab)
                # Take the last token or a specific sequence index?
                # Usually we want the last token for generative models
                score = output[:, -1, target_idx]
            else:
                score = output[:, target_idx]

            model.zero_grad()
            score.backward(retain_graph=True)
            grads.append(x.grad.clone())

        avg_grads = torch.stack(grads).mean(dim=0)
        attributions = (inputs - baseline) * avg_grads
        return attributions

    @staticmethod
    def get_attention_maps(
        model: nn.Module,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> List[torch.Tensor]:
        """
        Extract attention maps from a Transformer model.

        Args:
            model: A HuggingFace model or similar that returns 'attentions'.
            input_ids: Input token IDs.
            attention_mask: Attention mask.

        Returns:
            List[torch.Tensor]: A list of attention maps (one per layer).
        """
        # Ensure model is in eval mode and returns attentions
        was_training = model.training
        model.eval()

        with torch.no_grad():
            outputs = model(
                input_ids, attention_mask=attention_mask, output_attentions=True
            )
            attentions = (
                outputs.attentions
            )  # Tuple of (batch, num_heads, seq_len, seq_len)

        if was_training:
            model.train()

        return list(attentions)

    @staticmethod
    def visualize_attention(
        attention_map: torch.Tensor,
        tokens: List[str],
        layer_idx: int = -1,
        head_idx: Optional[int] = None,
    ) -> Any:
        """
        Generate a visualization (heatmap data) for an attention map.

        Args:
            attention_map: Attention tensor (num_layers, batch, num_heads, seq, seq).
            tokens: String tokens for labeling.
            layer_idx: Which layer to visualize.
            head_idx: Which head to visualize (averages heads if None).
        """
        # This would typically return a structure suitable for a frontend plot
        # or use matplotlib to save an image.
        layer_attn = attention_map[layer_idx][0]  # Assuming batch size 1

        if head_idx is not None:
            attn_data = layer_attn[head_idx]
        else:
            attn_data = layer_attn.mean(dim=0)

        return {
            "attentions": attn_data.cpu().numpy().tolist(),
            "tokens": tokens,
            "layer": layer_idx,
            "head": head_idx if head_idx is not None else "average",
        }
