# Rules — AI / ML

- Reproducibility: seed everything; expose a deterministic-mode toggle; log resolved config with
  every run.
- Device-agnostic: select CPU/CUDA/MPS via the device helper; never hard-code `.cuda()`.
- Memory: document VRAM footprints for backbones; quantization presets (bitsandbytes) and
  fine-tuning (PEFT/DeepSpeed/Lightning) live behind config, not code branches.
- Weights I/O: safe serialization; version checkpoints; no silent shape-mismatch loads.
- Evaluation/benchmarks go under `benchmark/` with fixed seeds and feed `docs/BENCHMARKS.md`;
  never assert on exact float outputs — assert on shapes, ranges, and monotonicity.
- Serving: models load once and stay resident in the sidecar; requests never reload weights.
