# Roadmap — Infrastructure

Cross-cutting: device management, Hydra configs, utilities, profiling, and benchmarks.
Implementation in [`src/device/`](../../src/device/), [`src/configs/`](../../src/configs/),
[`src/utils/`](../../src/utils/), and [`benchmark/`](../../benchmark/).

## §1 — Configuration

- [ ] Comment the YAML config groups (currently ~20% documented); add a config README per group.
- [ ] Dataclass-backed config validation; single source of truth via Hydra composition.

## §2 — Devices & utils

- [ ] Document device selection (CPU/CUDA/MPS) and mixed-precision helpers.
- [ ] Registries/profiling utilities documented; remove broad `Any` types.

## §3 — Benchmarks

- [ ] `benchmark/` produces reproducible numbers feeding [`docs/BENCHMARKS.md`](../../docs/BENCHMARKS.md);
      wire into `just bench`.

## §4 — Tooling

- [ ] Ruff/mypy clean; `just lint` + `just test` green in CI (`.github/workflows/`).
