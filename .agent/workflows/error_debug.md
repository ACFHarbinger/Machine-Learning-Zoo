# Workflow — Error Debugging

1. **Reproduce** with the narrowest entry (`just main …` / `just sidecar`) and the resolved Hydra
   config printed; save the failing config/input.
2. **Localize** along: config → factory/registry construction → shapes/device → data loader →
   serving/IPC (per `.agent/prompts/debug.md`).
3. **Fix at the failing layer** (e.g. a registry-key typo is a config fix, not a model change).
4. **Regression test**: add the failing case (tiny config, no network); mark GPU/slow if needed.
5. **Verify**: `just test::test`, rerun the reproduction; update TROUBLESHOOTING if user-facing.
