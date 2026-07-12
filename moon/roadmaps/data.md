# Roadmap — Data & Storage

Datasets, loaders, database, and artifact storage. Implementation in
[`src/data/`](../../src/data/), [`src/db/`](../../src/db/), and [`src/storage/`](../../src/storage/).

## §1 — Datasets & loaders

- [ ] Document each dataset (`polymarket_dataset`, `time_series_dataset`, streaming) and the
      prefetch/streaming loader trade-offs.
- [ ] `download_dataset.py` provenance + caching documented; no secrets in code.

## §2 — Database & storage

- [ ] Document the DB schema/migrations and the artifact-storage abstraction.
- [ ] Idempotent writes; clear separation of raw vs processed data.

## §3 — Tests

- [ ] Loader tests against tiny fixtures (no network); streaming back-pressure covered.
