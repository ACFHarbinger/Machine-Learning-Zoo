# Skill: Build & Test

Standard verification sequence — run before declaring a task done.

```bash
just sync           # uv sync --all-extras --dev  (first time / after dep changes)
just lint           # black --check + isort --check + ruff + mypy
just typecheck      # mypy .
just test::test     # pytest  (or: just test-run)
just coverage       # pytest --cov=src
```

Selective:

```bash
uv run pytest tests/<area> -q          # one subsystem
uv run pytest -m "not gpu and not slow"   # CI-equivalent subset
```

Docs when changed: `just docs-build` (Sphinx).
