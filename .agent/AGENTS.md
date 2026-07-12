# AGENTS.md — Machine Learning Zoo: Coding-Assistant Handbook

Instructions for AI assistants (and humans) working on this codebase. For the product-level
**Agent / Persona / Sidecar** feature, see [`docs/AGENTS.md`](../docs/AGENTS.md) instead.

## 1. Overview

Machine Learning Zoo (MLZ) is a modular ML library: model families, RL environments/policies,
training pipelines, data/storage, a serving sidecar, speech (STT/TTS), and personas. It is
consumed as a submodule by the `nglab` platform (the Python "strategy brain").

Authoritative design record: [`docs/ARCHITECTURE.md`](../docs/ARCHITECTURE.md). Roadmaps:
[`moon/ROADMAP.md`](../moon/ROADMAP.md) (master) + [`moon/roadmaps/`](../moon/roadmaps/).

## 2. Tech Stack

- **Runtime**: Python 3.10+ managed with **uv** (`uv sync --all-extras --dev`).
- **ML**: PyTorch, PyTorch Lightning, TorchRL/SB3, Transformers, PEFT, DeepSpeed, bitsandbytes.
- **RL**: Gymnasium environments + policy factories.
- **Config**: Hydra + OmegaConf (`src/configs/`).
- **Quality**: ruff (line-length 120), black, isort (black profile), mypy, flake8; pytest + cov.

## 3. Project Structure (`src/`)

```
src/
├── models/      # model families + ModelFactory + registry
├── features/    # feature extractors
├── envs/        # Gymnasium environments + factory + wrappers
├── policies/    # action policies (neural, regular, threshold, black_scholes) + factory
├── pipeline/    # training loops, callbacks, distributed/accelerated, backtesting, continual/active
├── data/        # datasets + loaders (streaming, prefetch, time-series, polymarket)
├── db/ storage/ # database + artifact storage
├── api/ ipc/    # inference server, dashboard, health, A/B; IPC bridge + request handler
├── stt/ tts/    # Whisper STT, ElevenLabs TTS
├── device/ utils/ constants/ enums/ exceptions.py  # infra
└── main.py logic_main.py ml_sidecar_main.py personality.py
```

Repo layout: `moon/` (roadmaps), `docs/` (guides + ADRs), `git/` (CONTRIBUTING + codecov),
`tools/` (justfile sub-modules), `.github/` (CI + templates), `benchmark/`, `examples/`, `tests/`.

## 4. Common Commands (just)

| Action | Command |
| :--- | :--- |
| Sync env | `just sync` |
| Format | `just fmt` |
| Lint (CI-equiv) | `just lint` |
| Type-check | `just typecheck` |
| Tests | `just test::test` (or `just test-run`) |
| Coverage | `just coverage` |
| Run CLI / sidecar / API | `just main` / `just sidecar` / `just serve` |
| Docs | `just docs-build` |
| List everything | `just help` |

## 5. Coding Standards

- `from __future__ import annotations`; full type annotations + Google-style docstrings on public
  APIs (many files still carry broad `Any` — tighten as you touch them).
- **Factories & registries are the only construction path**: `ModelFactory`, `EnvFactory`,
  policy factory, etc. New components register with the central registry + expose a config
  dataclass/Hydra group — never hard-code hyperparameters.
- Optional/heavy deps (deepspeed, bitsandbytes, llama-cpp, elevenlabs, whisper) are import-guarded
  with an actionable message (name the extra: `uv sync --extra gpu`); secrets come from env only.
- Custom exceptions from `src/exceptions.py`; no bare `except`.
- Config lives in Hydra groups (`src/configs/`), not scattered constants.

## 6. Testing

- pytest under `tests/`; keep unmarked tests headless and fast (no network, tiny configs).
- Mark GPU/weight-loading tests so CI can skip them; mock network in serving/speech adapters.
- Every bug fix ships a regression test. Target coverage: see [`git/codecov.yaml`](../git/codecov.yaml).

## 7. Documentation Discipline

- Completed roadmap items move from `moon/` to [`docs/CHANGELOG.md`](../docs/CHANGELOG.md).
- New subsystems get a `moon/roadmaps/<area>.md` section, a docs entry, and factory/registry wiring.
