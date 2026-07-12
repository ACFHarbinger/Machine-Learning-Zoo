# Troubleshooting

[![Python](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/downloads/)

Common issues and their solutions when working with Machine Learning Zoo.

## Installation Issues

### `ModuleNotFoundError: No module named 'machine_learning_zoo'`

**Cause**: The package is not installed in the current environment or the Python path is incorrect.
**Solution**:

- Ensure you are in the correct virtual environment (`source .venv/bin/activate`).
- Reinstall in editable mode: `uv pip install -e .`

### `CUDA initialization: Torch not compiled with CUDA enabled`

**Cause**: PyTorch was installed without CUDA support.
**Solution**:

- Uninstall torch: `uv pip uninstall torch`
- Reinstall with the correct index:
  ```bash
  uv pip install torch --index-url https://download.pytorch.org/whl/cu118
  ```
  (Check [pytorch.org](https://pytorch.org/) for the command matching your drivers).

## Runtime Issues

### Agent hangs or disconnects

**Cause**: IPC issues between the main process and the sidecar, or the model is taking too long to load.
**Solution**:

- Check the logs for "Sidecar starting" messages.
- Ensure the `NdjsonTransport` is receiving input.
- If using a large local model, increase the timeout configuration in your main app settings.

### `ValueError: Failed to create llama_context`

**Cause**: Usually indicates OOM (Out Of Memory) or corrupted model files.
**Solution**:

- Check GPU VRAM usage.
- Try loading a smaller model.
- Verify the model path in `model_registry`.

## Performance

### Inference is slow

**Solution**:

- Ensure `device_manager` is selecting "cuda" and not "cpu".
- Check if you are running in debug mode (which might disable optimizations).
