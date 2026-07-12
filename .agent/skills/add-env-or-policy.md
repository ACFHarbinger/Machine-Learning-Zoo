# Skill: Add an RL Environment or Policy

Add to `src/envs/` or `src/policies/`.

**Environment**
1. Subclass `envs/base.py`; implement the Gymnasium API (reset/step/spaces).
2. Register with `EnvFactory`; add a config group for its parameters.
3. Provide wrappers as needed (vectorized/multi-agent) rather than baking variants in.

**Policy**
1. Subclass `policies/base.py` (neural / regular / threshold / analytical).
2. Register with the policy factory; document when to use it vs the alternatives.
3. Analytical baselines (Black-Scholes, threshold) get deterministic unit tests.

Both: keep the Gymnasium interface stable (nglab's Rust core consumes it); update
`moon/roadmaps/rl.md` + CHANGELOG.
