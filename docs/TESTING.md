# Testing Guide

[![Python](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/downloads/)
[![Coverage](https://img.shields.io/badge/coverage-80%25-green)](TESTING.md)

Ensuring code quality and reliability is a priority. We use `pytest` for unit and integration testing.

## Running Tests

To run the full test suite:

```bash
pytest
```

To run with coverage report:

```bash
pytest --cov=machine_learning_zoo coverage_html
```

## Test Structure

Tests are located in the `tests/` directory and mirror the source structure.

- `tests/test_models/`: Unit tests for model architectures.
- `tests/test_pipeline/`: Tests for data pipelines.
- `tests/test_integration/`: End-to-end integration tests.

## Markers

We use pytest markers to categorize tests:

- `@pytest.mark.slow`: Tests that take a long time to run (excluded by default in fast checks).
- `@pytest.mark.gpu`: Tests that require a CUDA device.
- `@pytest.mark.integration`: Tests that involve multiple components working together.

Example running only fast CPU tests:

```bash
pytest -m "not slow and not gpu"
```

## Writing Tests

We encourage writing tests for every new feature.

1.  Use `conftest.py` fixtures for common setup (e.g., creating dummy data).
2.  Mock heavy dependencies (like actual GPU calls) in unit tests.
3.  Ensure your tests pass strict typing if possible.
