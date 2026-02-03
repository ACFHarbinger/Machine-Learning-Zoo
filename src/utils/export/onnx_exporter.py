"""Utility for exporting PyTorch models to ONNX."""

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def export_to_onnx(
    model: nn.Module,
    output_path: str,
    input_shape: Tuple[int, ...],
    input_names: Optional[list] = None,
    output_names: Optional[list] = None,
    dynamic_axes: Optional[Dict[str, Dict[int, str]]] = None,
    opset_version: int = 14,
) -> Path:
    """
    Export a PyTorch model to ONNX format.

    Args:
        model: The PyTorch model instance.
        output_path: Destination path for the .onnx file.
        input_shape: Shape of the dummy input tensor (e.g., (1, 100, 1)).
        input_names: Names for input nodes.
        output_names: Names for output nodes.
        dynamic_axes: Dictionary defining variable dimensions.
        opset_version: ONNX opset version.

    Returns:
        Path: The path to the exported ONNX file.
    """
    model.eval()
    dummy_input = torch.randn(*input_shape)

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if input_names is None:
        input_names = ["input"]
    if output_names is None:
        output_names = ["output"]

    logger.info(f"Exporting model to {output_path}...")

    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
    )

    logger.info("Export completed successfully.")
    return output_path


class ONNXExporter:
    """Convenience wrapper for exporting Zoo models."""

    @staticmethod
    def export_time_series(
        model: nn.Module,
        output_path: str,
        seq_len: int = 100,
        num_features: int = 1,
        dynamic_axes: Optional[Dict[str, Dict[int, str]]] = None,
    ):
        """Export a time-series backbone model."""
        input_shape = (1, seq_len, num_features)
        if dynamic_axes is None:
            dynamic_axes = {
                "input": {0: "batch_size", 1: "sequence_length"},
                "output": {0: "batch_size", 1: "sequence_length"},
            }
        return export_to_onnx(
            model, output_path, input_shape, dynamic_axes=dynamic_axes
        )
