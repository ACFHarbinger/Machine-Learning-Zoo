# Dependencies

[![Python](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/downloads/)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)

The Machine Learning Zoo relies on a robust set of libraries for computation, configuration, and environment management.

## Core Runtime

- **[PyTorch](https://pytorch.org/)**: tensor computation and deep learning.
- **[NumPy](https://numpy.org/)**: Fundamental package for scientific computing.
- **[Pandas](https://pandas.pydata.org/)**: Data analysis and manipulation.

## Configuration & CLI

- **[Hydra](https://hydra.cc/)**: Framework for elegantly configuring complex applications.
- **[OmegaConf](https://omegaconf.readthedocs.io/)**: Flexible YAML-based configuration merging.
- **[tqdm](https://github.com/tqdm/tqdm)**: Fast, extensible progress bars.

## Reinforcement Learning

- **[Gymnasium](https://gymnasium.farama.org/)**: Standard API for reinforcement learning environments.
- **[Stable Baselines3](https://stable-baselines3.readthedocs.io/)**: Reliable implementations of RL algorithms.

## Development & Testing

- **[pytest](https://docs.pytest.org/)**: The testing framework used.
- **[Black](https://github.com/psf/black)**: The uncompromising code formatter.
- **[isort](https://pycqa.github.io/isort/)**: Import sorter.
- **[mypy](https://mypy-lang.org/)**: Optional static typing for Python.

## Managing Dependencies

We use `uv` for high-performance dependency management.

```bash
# Sync dependencies
uv sync

# Add a new dependency
uv add package_name
```
