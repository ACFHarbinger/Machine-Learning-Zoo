# Skill: Add a Model

Add a new model to `src/models/`.

1. **Placement**: new module under the right family dir (attention, autoencoders, backbones,
   convolutional, competitive, …); subclass the family `base.py`.
2. **Registry + factory**: register under a string key so `ModelFactory` can build it; no direct
   instantiation elsewhere.
3. **Config**: add a config dataclass / Hydra group (`src/configs/`) with the hyperparameters and
   sensible defaults; document VRAM footprint.
4. **Fine-tuning/quant**: if applicable, wire PEFT/DeepSpeed/bitsandbytes via config, not branches.
5. **Tests**: shape/forward test on a tiny CPU config; `gpu`-marked test if it must load weights.
6. **Docs**: row in `docs/API_REFERENCE.md`; tick `moon/roadmaps/models.md`; CHANGELOG note.
