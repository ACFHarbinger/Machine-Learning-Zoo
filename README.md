# Machine Learning Zoo

A comprehensive machine learning library with modular components for building and training various ML models.

## Features

- **Modular Architecture**: Separate backbones, heads, and pipelines for flexible model composition
- **Multiple Model Types**: Support for transformers, LSTMs, CNNs, attention mechanisms, and more
- **Reinforcement Learning**: Built-in support for RL algorithms like PPO
- **Preset Configurations**: Ready-to-use configurations for common tasks
- **Extensible**: Easy to add new components through registries

## Installation

### From Source

```bash
# Clone the repository
git clone https://github.com/yourusername/machine-learning-zoo.git
cd machine-learning-zoo

# Install with uv (recommended)
uv pip install -e .

# Or install in development mode with all optional dependencies
uv pip install -e ".[dev,docs]"

# For GPU support (if you have CUDA)
uv pip install -e ".[gpu]"
```

> **Note**: We recommend using [uv](https://github.com/astral-sh/uv) for fast, reliable package installation and dependency resolution.

### Adding Dependencies

When developing, you can add new dependencies using uv:

```bash
# Add a runtime dependency
uv add torch numpy

# Add a development dependency
uv add --dev pytest black

# Add to a specific optional dependency group
uv add --optional docs sphinx
```

### Using the Main Script

After installation, you can use the command-line interface:

```bash
# List available presets
mlzoo --list-presets

# Build and demo a model from a preset
mlzoo presets/supervised_classification.yaml
```

### Available Presets

- `rl_ppo_mamba.yaml`: Reinforcement learning with PPO algorithm using Mamba backbone
- `supervised_classification.yaml`: Supervised classification with Transformer backbone

### Building Models Programmatically

```python
from models.composed import build_model

# Build a classification model
model = build_model(
    backbone_name="transformer",
    head_name="classification",
    backbone_config={"hidden_dim": 256, "num_layers": 4},
    head_config={"num_classes": 10}
)
```

## Project Structure

- `models/`: Model components (backbones, heads, factories)
- `pipeline/`: Training and evaluation pipelines
- `data/`: Data loading and preprocessing utilities
- `envs/`: Environment definitions for RL
- `configs/`: Configuration management
- `presets/`: Predefined configuration files
- `utils/`: Utility functions and helpers

## Contributing

This is a modular ML library designed for easy extension. To add new components:

1. Implement your component class
2. Register it in the appropriate registry
3. Add configuration support if needed

## License

See LICENSE file for details.
