# Rules — Test Writing

- Framework: pytest under `tests/`. Unmarked tests run headless in CI in seconds — no network, no
  GPU, tiny configs.
- Mark GPU/weight-loading and slow tests so CI can exclude them; mock network in serving/speech
  adapter tests.
- Test the factory/registry contract (construction from config) plus one realistic forward/step
  per subsystem.
- Determinism: fixed seeds; assert on shapes/ranges/structure, not exact float values.
- Fixtures documented (purpose comment); grow the corpus with every field bug.
- Coverage targets in [`git/codecov.yaml`](../../git/codecov.yaml).
