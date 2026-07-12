# Rules — Python

- Python 3.10+; `from __future__ import annotations` in modules using new-style unions.
- Formatting: black (line-length 88 for black, ruff line-length 120) + isort (black profile) +
  ruff format. Linting: ruff + mypy + flake8. `just lint` mirrors CI.
- Public functions/classes: full type annotations + Google-style docstrings. Prefer precise types
  over `Any`.
- Construction only via factories/registries (`ModelFactory`, `EnvFactory`, policy factory).
- Configuration via Hydra groups (`src/configs/`); no scattered constants or env lookups in
  business logic.
- Optional/heavy deps (deepspeed, bitsandbytes, llama-cpp, whisper, elevenlabs) import-guarded
  with a message naming the uv extra (`uv sync --extra gpu`). Secrets from env only.
- Errors: custom exceptions from `src/exceptions.py`; no bare `except`.
- Every bug fix ships a regression test.
