# Changelog

[![License: AGPL v3](https://img.shields.io/badge/License-AGPL_v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **Distributed Training**:
  - Implementation of `DistributedDataParallel` (DDP) support in `TrainingOrchestrator`.
  - Added rank-aware logging and progress streaming via `ProgressCallback`.
  - Integrated `pytorch-lightning` for scalable training across multiple GPUs.
- **Multi-Modal Pipelines**:
  - Implemented `HuggingFaceVisionBackbone` for vision encoder support.
  - Implemented `MultiModalBackbone` for aligning vision and text hidden states.
  - Enhanced `InferenceEngine` to support multimodal inputs (image + text).
  - Added support for LLaVA-style architectures (CLIP + LLM).
- **Advanced LLM Support**:
  - Implemented `HuggingFaceBackbone` for native Transformers integration.
  - Added support for Llama-3, Llama-3.1, and DeepSeek (R1-Distill) models.
  - Enhanced `InferenceEngine` with `top_p`, `top_k`, and `repetition_penalty` parameters.
  - Implemented 4-bit/8-bit quantization loading via `bitsandbytes`.
- Initial creation of documentation files:
  - CONTRIBUTING.md
  - ARCHITECTURE.md
    ... (rest of the file)

### Changed

- Refactored `ml_sidecar_main.py` entry point (recent optimizations).
- Enhanced test coverage for core components.

### Fixed

- Resolved local model loading issues with `deepseek-r1-32b`.
