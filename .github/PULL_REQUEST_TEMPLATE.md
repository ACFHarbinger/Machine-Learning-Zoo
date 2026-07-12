# Pull Request

## Summary

<!-- What does this PR change and why? Link the roadmap item (moon/ROADMAP.md or moon/roadmaps/). -->

## Affected Subsystem(s)

- [ ] Models (`src/models`, `src/features`)
- [ ] RL envs / policies / pipeline (`src/envs`, `src/policies`, `src/pipeline`)
- [ ] Data & storage (`src/data`, `src/db`, `src/storage`)
- [ ] Serving & IPC (`src/api`, `src/ipc`, sidecar)
- [ ] Speech / personas (`src/stt`, `src/tts`, `personality.py`)
- [ ] Infrastructure (`src/device`, `src/utils`, `src/configs`, `benchmark`)
- [ ] Docs / tooling / CI

## Type of Change

- [ ] 🐛 Bug fix
- [ ] ✨ New feature
- [ ] ♻️ Refactor
- [ ] ⚡ Performance
- [ ] 📚 Documentation
- [ ] 🔧 Tooling / CI

## Checklist

- [ ] `just lint` and `just test` pass
- [ ] New components register via their factory/registry and expose a config dataclass/group
- [ ] Optional/heavy deps are import-guarded with actionable messages
- [ ] Public APIs have Google-style docstrings + `from __future__ import annotations`
- [ ] Docs / roadmaps / CHANGELOG updated where the public surface changed
