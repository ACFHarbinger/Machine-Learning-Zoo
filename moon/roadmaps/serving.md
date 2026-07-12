# Roadmap — Serving & IPC

The ML sidecar, request handlers, and IPC bridge that expose models to callers (e.g. the nglab
Rust core). Implementation in [`src/api/`](../../src/api/), [`src/ipc/`](../../src/ipc/), and the
`ml_sidecar_main.py` / `main.py` entry points.

## §1 — Sidecar & API

- [ ] Document the sidecar lifecycle and the API surface (`server`, `inference`, `health`,
      `dashboard`, `ab_testing`).
- [ ] Health/readiness endpoints and graceful shutdown documented.

## §2 — IPC bridge

- [ ] Document the IPC transport used by `MlRequestHandler` (commands: `echo`, `set_persona`,
      `chat`) and its streaming vs synchronous modes.
- [ ] Stable message contracts so the Rust/host side can depend on them.

## §3 — Tests

- [ ] `verify_api.py` promoted to a proper integration test; request handlers unit-tested with fakes.
