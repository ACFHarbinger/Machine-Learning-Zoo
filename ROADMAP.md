# Roadmap

[![License: AGPL v3](https://img.shields.io/badge/License-AGPL_v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)

This document outlines the high-level goals and planned features for the Machine Learning Zoo.

## Phase 1: Foundation (Current)

- [x] **Modular Architecture**: Establish Backbone/Head/Pipeline structure.
- [x] **Basic RL Support**: PPO implementation.
- [x] **Sidecar Interface**: Robust NDJSON IPC for agent integration.
- [/] **Documentation**: Comprehensive guides and API docs.

## Phase 2: Enhanced Capabilities

- [ ] **Advanced LLM Support**: Better integration for deepseek, llama-3, and other open-source models.
- [ ] **Multi-Modal Pipelines**: Support for Image-to-Text and Text-to-Image within the same registry.
- [ ] **Distributed Training**: Support for `DistributedDataParallel` (DDP) in `training/`.

## Phase 3: Ecosystem Integration

- [ ] **Model Hub**: Easy download/upload of pretrained weights.
- [ ] **Visual Dashboard**: A React-based dashboard for monitoring training real-time (integrated with the main app).
- [ ] **AutoML**: Automated hyperparameter tuning support.

## Phase 4: Performance

- [ ] **Quantization**: Native support for 4-bit/8-bit loading via `bitsandbytes`.
- [ ] **Optimized Inference**: Integration with vLLM or similar libraries for high-throughput serving.

_Note: Timelines are subject to change based on community feedback and contribution._
