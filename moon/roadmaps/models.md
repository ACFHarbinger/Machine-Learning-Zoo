# Roadmap — Models

Model library and its factories/registries. Implementation lives in [`src/models/`](../../src/models/)
(attention, autoencoders, backbones, convolutional, competitive, composed, …) and
[`src/features/`](../../src/features/).

## §1 — Registry & factory hygiene

- [ ] Every model registers via the central registry; `ModelFactory` is the only construction path.
- [ ] Document each family's `base.py` contract (forward signature, config dataclass, weight I/O).
- [ ] Google-style docstrings + `from __future__ import annotations` on all public model modules.

## §2 — Fine-tuning stack

- [ ] Consolidate PEFT / DeepSpeed / Lightning fine-tuning entry points behind one config group.
- [ ] Quantization (bitsandbytes) presets documented per backbone with VRAM footprints.

## §3 — Coverage & tests

- [ ] Shape/forward unit tests per family (CPU-only, tiny configs); `gpu`-marked tests for weights.
- [ ] Golden-config smoke tests so factory wiring can't silently break.

## §4 — Future

- [ ] New architectures land here with a registry entry, a config dataclass, a Docs/API reference
      row, and a CHANGELOG note.
