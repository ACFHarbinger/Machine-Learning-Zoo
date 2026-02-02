# Machine Learning Zoo

[![License: AGPL v3](https://img.shields.io/badge/License-AGPL_v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![Python](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Checked with mypy](http://www.mypy-lang.org/static/mypy_badge.svg)](http://mypy-lang.org/)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)

A comprehensive machine learning library with modular components for building and training various ML models.

## 📚 Documentation

- [**Contributing**](.github/CONTRIBUTING.md): How to contribute code.
- [**Architecture**](ARCHITECTURE.md): System design overview.
- [**Agents**](AGENTS.md): Understanding the Agent/Sidecar system.
- [**Development**](DEVELOPMENT.md): Setup and development guide.
- [**Roadmap**](ROADMAP.md): Future plans.
- [**Tutorial**](TUTORIAL.md): Build your first model.

## Features

- **Modular Architecture**: Separate backbones, heads, and pipelines for flexible model composition.
- **Multiple Model Types**: Support for transformers, LSTMs, CNNs, attention mechanisms, and more.
- **Reinforcement Learning**: Built-in support for RL algorithms like PPO.
- **Preset Configurations**: Ready-to-use configurations for common tasks.
- **Extensible**: Easy to add new components through registries.

## Installation

We recommend using [uv](https://github.com/astral-sh/uv) for fast, reliable package installation.

```bash
# Clone the repository
git clone https://github.com/yourusername/machine-learning-zoo.git
cd machine-learning-zoo

# Install with uv
uv pip install -e .

# Or install in development mode with all optional dependencies
uv pip install -e ".[dev,docs]"

# For GPU support (if you have CUDA)
uv pip install -e ".[gpu]"
```

See [DEPENDENCIES.md](DEPENDENCIES.md) for a full list of requirements.

## Usage

After installation, you can use the command-line interface:

```bash
# List available presets
mlzoo --list-presets

# Build and demo a model from a preset
mlzoo presets/supervised_classification.yaml
```

### Building Models Programmatically

```python
from models.composed import build_model

# Build a classification model
model = build_model(
    backbone_name="transformer",
    head_name="classification",
    backbone_kwargs={"d_model": 512, "nhead": 8},
    head_kwargs={"num_classes": 10},
)
print(model)
```

## Support

If you encounter issues, please check [TROUBLESHOOTING.md](TROUBLESHOOTING.md) or open an issue on GitHub.
