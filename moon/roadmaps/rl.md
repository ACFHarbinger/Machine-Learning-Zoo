# Roadmap — RL Environments & Policies

Gymnasium environments, action policies, and training pipelines. Implementation in
[`src/envs/`](../../src/envs/), [`src/policies/`](../../src/policies/), and
[`src/pipeline/`](../../src/pipeline/).

## §1 — Environments

- [ ] Document the `envs/base.py` + `factory.py` contract; wrappers (`env_wrapper`, `vectorized_env`,
      `multi_agent`) each get a usage docstring.
- [ ] `trading_env.py` config surface documented (observation/action spaces, reward terms).

## §2 — Policies

- [ ] Unify the policy hierarchy (`neural`, `regular`, `threshold`, `black_scholes`) under
      `policies/base.py` + `factory.py`; document when to use each.
- [ ] Analytical baselines (Black-Scholes, threshold) covered by deterministic unit tests.

## §3 — Training pipeline

- [ ] Document `pipeline/base.py`, callbacks, distributed/accelerated training, active/continual
      learning, and backtesting entry points.
- [ ] Reproducibility: seed handling + deterministic-mode toggle documented.

## §4 — Integration

- [ ] Keep the Gymnasium interface stable for the Rust simulation core (nglab consumes it).
