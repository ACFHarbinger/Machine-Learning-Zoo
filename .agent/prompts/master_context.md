# Master Context Prompt

```
You are working on Machine Learning Zoo (MLZ): a modular Python ML library (models, RL
envs/policies, training pipelines, data/storage, a serving sidecar, speech STT/TTS, personas),
consumed as a submodule by the nglab platform.

Structure: src/ holds the library (models, features, envs, policies, pipeline, data, db, storage,
api, ipc, stt, tts, device, utils). Roadmaps in moon/; guides + ADRs in docs/; CONTRIBUTING +
codecov in git/; tools/ holds the justfile sub-modules.

Rules of engagement:
- Construction goes through factories/registries (ModelFactory, EnvFactory, policy factory);
  config via Hydra groups (src/configs/), never hard-coded hyperparameters.
- from __future__ import annotations; Google-style docstrings on public APIs.
- Guard optional/heavy deps (deepspeed, bitsandbytes, whisper, elevenlabs) with actionable
  install messages; secrets from env only.
- Run `just lint`, `just typecheck`, and `just test::test` before declaring done.
- Read docs/ARCHITECTURE.md before structural changes; update moon/ + docs/CHANGELOG.md after.
```
