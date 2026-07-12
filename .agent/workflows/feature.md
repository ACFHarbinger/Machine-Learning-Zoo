# Workflow — Feature Implementation

1. **Scope**: identify the subsystem + its factory/registry contract and the relevant
   `moon/roadmaps/` item. Read `docs/ARCHITECTURE.md` if boundaries are involved.
2. **Config first**: add the config dataclass / Hydra group before the code depends on it.
3. **Implement** per the skill (`add-model` / `add-env-or-policy`); register with the factory.
4. **Test**: tiny-config unit test for the contract + one realistic forward/step; mark GPU tests.
5. **Gate**: `just lint && just typecheck && just test::test`.
6. **Document**: roadmap tick, `docs/API_REFERENCE.md`, `docs/CHANGELOG.md`.
