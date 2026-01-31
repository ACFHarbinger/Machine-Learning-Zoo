#!/usr/bin/env python3
"""
Main entry point for Machine Learning Zoo.

This script provides a command-line interface to load presets,
build models, and run training or evaluation tasks.
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict

try:
    import torch
    from omegaconf import OmegaConf

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: PyTorch or OmegaConf not available. Some features may not work.")

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

try:
    from models.composed import build_model
    from utils.config import deep_sanitize

    LOCAL_MODULES_AVAILABLE = True
except ImportError as e:
    LOCAL_MODULES_AVAILABLE = False
    print(f"Warning: Could not import local modules: {e}")


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to the YAML configuration file

    Returns:
        Sanitized configuration dictionary
    """
    if not TORCH_AVAILABLE:
        raise ImportError("OmegaConf not available")

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    # Load YAML using OmegaConf
    cfg = OmegaConf.load(config_path)

    # Convert to plain dict
    return deep_sanitize(cfg)


def build_model_from_config(config: Dict[str, Any]) -> Any:
    """
    Build a model from configuration.

    Args:
        config: Configuration dictionary

    Returns:
        Built model
    """
    if not LOCAL_MODULES_AVAILABLE:
        raise ImportError("Local modules not available")

    backbone_name = config.get("backbone")
    head_name = config.get("head", "classification")  # Default to classification if not specified
    backbone_config = config.get("backbone_config", {})
    head_config = config.get("head_config", {})

    if not backbone_name:
        raise ValueError("Configuration must specify a 'backbone'")

    print(f"Building model with backbone: {backbone_name}, head: {head_name}")
    print(f"Backbone config: {backbone_config}")
    print(f"Head config: {head_config}")

    model = build_model(
        backbone_name=backbone_name,
        head_name=head_name,
        backbone_config=backbone_config,
        head_config=head_config,
    )

    return model


def demonstrate_model(model: Any, config: Dict[str, Any]) -> None:
    """
    Demonstrate the built model with dummy data.

    Args:
        model: The built model
        config: Configuration dictionary
    """
    if not TORCH_AVAILABLE:
        print("Cannot demonstrate model: PyTorch not available")
        return

    # Get data dimensions from config or use defaults
    seq_len = config.get("data", {}).get("seq_len", 64)
    batch_size = config.get("data", {}).get("batch_size", 32)

    # Try to infer input dimensions based on backbone type
    backbone = config.get("backbone", "")
    if "transformer" in backbone.lower() or "attention" in backbone.lower():
        # Assume sequence data: (batch_size, seq_len, feature_dim)
        feature_dim = 128  # Default
        dummy_input = torch.randn(batch_size, seq_len, feature_dim)
    elif "lstm" in backbone.lower() or "rnn" in backbone.lower():
        # Sequence data: (batch_size, seq_len, feature_dim)
        feature_dim = 64
        dummy_input = torch.randn(batch_size, seq_len, feature_dim)
    elif "cnn" in backbone.lower() or "conv" in backbone.lower():
        # Image-like data: (batch_size, channels, height, width)
        dummy_input = torch.randn(batch_size, 3, 32, 32)
    else:
        # Generic: assume 2D input
        dummy_input = torch.randn(batch_size, 784)  # Like MNIST flattened

    print(f"Created dummy input with shape: {dummy_input.shape}")

    # Move to device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    dummy_input = dummy_input.to(device)

    # Forward pass
    with torch.no_grad():
        try:
            output = model(dummy_input)
            print(
                f"Model output shape: {output.shape if hasattr(output, 'shape') else type(output)}"
            )
            print(f"Model output type: {type(output)}")
            if isinstance(output, torch.Tensor):
                print(f"Output sample: {output[0] if output.numel() > 0 else 'Empty tensor'}")
        except Exception as e:
            print(f"Error during forward pass: {e}")
            print("This might be expected if the model requires specific input preprocessing.")


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Machine Learning Zoo - Build and demonstrate models from presets"
    )
    parser.add_argument("preset", nargs="?", help="Path to preset YAML configuration file")
    parser.add_argument("--list-presets", action="store_true", help="List available preset files")
    parser.add_argument(
        "--demo",
        action="store_true",
        default=True,
        help="Run model demonstration with dummy data (default: True)",
    )

    args = parser.parse_args()

    # List presets if requested
    if args.list_presets:
        presets_dir = project_root / "presets"
        if presets_dir.exists():
            print("Available presets:")
            for preset_file in presets_dir.glob("*.yaml"):
                print(f"  - {preset_file.name}")
        else:
            print("No presets directory found.")
        return

    if not args.preset:
        print("Error: No preset specified. Use --list-presets to see available presets.")
        sys.exit(1)

    # Load configuration
    try:
        config = load_config(args.preset)
        print(f"Loaded configuration from {args.preset}")
        print(f"Configuration keys: {list(config.keys())}")
    except Exception as e:
        print(f"Error loading configuration: {e}")
        sys.exit(1)

    # Build model
    try:
        model = build_model_from_config(config)
        print("Model built successfully!")
        print(f"Model type: {type(model)}")
        print(f"Number of parameters: {sum(p.numel() for p in model.parameters()):,}")
    except Exception as e:
        print(f"Error building model: {e}")
        sys.exit(1)

    # Demonstrate model
    if args.demo:
        print("\n--- Model Demonstration ---")
        demonstrate_model(model, config)

    print("\nDone!")


if __name__ == "__main__":
    main()
