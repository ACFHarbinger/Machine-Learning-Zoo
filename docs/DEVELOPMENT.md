# Development Guide

[![Python](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/downloads/)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

This guide covers how to set up your environment for developing on Machine Learning Zoo.

## Prerequisites

- **Operating System**: Linux (Ubuntu/Kubuntu recommended), macOS, or Windows WSL2.
- **Python**: Version 3.8 or higher.
- **Package Manager**: [uv](https://github.com/astral-sh/uv) (Highly recommended for speed).

## Environment Setup

1.  **Install uv** (if not installed):

    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```

2.  **Clone the repository**:

    ```bash
    git clone https://github.com/yourusername/machine-learning-zoo.git
    cd machine-learning-zoo
    ```

3.  **Install Dependencies**:

    ```bash
    # Creates a virtual environment and installs all deps
    uv sync

    # Or editable install
    uv pip install -e ".[dev,docs]"
    ```

4.  **Activate Virtual Environment**:
    ```bash
    source .venv/bin/activate
    ```

## Common Tasks

### Running the Sidecar

The main entry point for the Python backend is `ml_sidecar_main.py`.

```bash
python ml_sidecar_main.py
```

_Note: This usually runs as a subprocess, so it expects input via stdin._

### formatting Code

We use `black` and `isort`:

```bash
# Format code
black .
isort .
```

### Type Checking

Run `mypy` to check for type errors:

```bash
mypy .
```

### Adding New Models

1.  Create a new model file in `models/`.
2.  Register it in `models/__init__.py`.
3.  Add a generic configuration in `configs/`.

## Troubleshooting Environment

If you encounter `Library not loaded` or `CUDA error`:

- Ensure you have the correct PyTorch version for your CUDA driver.
- Run `uv pip install torch --index-url https://download.pytorch.org/whl/cu118` (adjust for your CUDA version).
