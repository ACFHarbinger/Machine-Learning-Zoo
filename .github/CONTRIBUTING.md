# Contributing to Machine Learning Zoo

[![License: AGPL v3](https://img.shields.io/badge/License-AGPL_v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Checked with mypy](http://www.mypy-lang.org/static/mypy_badge.svg)](http://mypy-lang.org/)

Thank you for your interest in contributing to Machine Learning Zoo! We welcome contributions from the community to help improve this library.

## Getting Started

1.  **Fork the repository** on GitHub.
2.  **Clone your fork** locally:
    ```bash
    git clone https://github.com/yourusername/machine-learning-zoo.git
    cd machine-learning-zoo
    ```
3.  **Set up the development environment**. We recommend using `uv`:
    ```bash
    uv pip install -e ".[dev,docs]"
    ```
    See [DEVELOPMENT.md](DEVELOPMENT.md) for more details.

## Development Workflow

1.  **Create a branch** for your feature or fix:
    ```bash
    git checkout -b feature/my-new-feature
    ```
2.  **Make your changes**. Ensure you follow the coding style.
3.  **Run tests** to ensure no regressions:
    ```bash
    pytest
    ```
4.  **Lint your code**:
    ```bash
    black .
    isort .
    mypy .
    ```

## Code Style

We follow strict code quality standards to maintain a clean codebase:

- **Formatting**: We use [Black](https://github.com/psf/black) with a line length of 88.
- **Imports**: Sorted by [isort](https://pycqa.github.io/isort/).
- **Typing**: All code must be fully typed and pass [mypy](https://mypy.readthedocs.io/) strict checks (see `pyproject.toml` for exceptions).

## Pull Requests

1.  Push your branch to your fork.
2.  Open a Pull Request against the `main` branch.
3.  Describe your changes clearly and link any related issues.
4.  Wait for review and address any feedback.

## Reporting Issues

If you find a bug or have a feature request, please open an issue in the issue tracker. Provide as much detail as possible, including reproduction steps for bugs.
