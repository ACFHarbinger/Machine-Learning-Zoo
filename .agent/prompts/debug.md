# Debugging Prompt

```
Debug the following MLZ failure: {SYMPTOM}.

Triage order:
1. Config — Hydra composition resolved as expected? (print the resolved config; check the group
   overrides and dataclass validation before touching model code).
2. Construction — is the component built via its factory/registry? A KeyError usually means a
   missing/typo'd registry key, not a model bug.
3. Shapes/devices — tensor shape or device (CPU/CUDA/MPS) mismatch; check the device helper and
   dtype/precision settings.
4. Data — loader returning the expected batch structure? (test against a tiny fixture, no network).
5. Serving — only then the sidecar/IPC path (request handler, streaming vs sync, persona state).

Reproduce with the narrowest entry (`just main ...` / `just sidecar`), add a regression test,
then fix at the failing layer.
```
