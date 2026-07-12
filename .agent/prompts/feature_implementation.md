# Feature Implementation Prompt

```
Implement the following MLZ feature: {FEATURE}.

Before writing code:
1. Locate the target subsystem (models / envs / policies / pipeline / data / api / stt / tts) and
   read its base.py + factory.py contract and the matching moon/roadmaps/ file.
2. Confirm there isn't an existing registry entry or factory hook you should extend instead.

Requirements:
- Register the new component with its central registry; expose a config dataclass / Hydra group
  (src/configs/) — no hard-coded hyperparameters.
- Guard optional/heavy deps with an import guard naming the uv extra.
- from __future__ import annotations; Google-style docstrings; tighten types (avoid Any).
- Add pytest coverage (tiny CPU configs; mark GPU/weight-loading tests; mock network).
- Update: the subsystem roadmap in moon/roadmaps/, docs/API_REFERENCE.md if the public surface
  changed, and docs/CHANGELOG.md.
```
