# Changelog

[![License: AGPL v3](https://img.shields.io/badge/License-AGPL_v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **Domain Adaptation**:
  - Implemented `MMDLoss` for distribution alignment.
  - Implemented `GradientReversalLayer` and `DomainDiscriminator` for Domain Adversarial Neural Networks (DANN).
  - Integrated domain adaptation losses into `PiLightningModule`.

- **Continual Learning**:
  - Implemented `EWCCallback` for Elastic Weight Consolidation.
  - Implemented `ReplayBuffer` and `ReplayDataset` for experience replay.
  - Integrated continual learning strategies into `TrainingOrchestrator`.
- **Prompt Engineering Toolkit**:
  - Implementation of `PromptTemplate` engine using Jinja2.
  - New `PromptRegistry` and `FewShotRegistry` for persistent storage of templates and examples.
  - Automated `PromptOptimizer` using LLM-in-the-loop refinement.
  - Integrated template rendering and few-shot decoration into `InferenceEngine`.
- **DeepSpeed Integration**:
  - Full support for `DeepSpeedStrategy` in `TrainingOrchestrator`.
  - Automated ZeRO-stage configuration (Stage 1, 2, and 3).
  - Support for CPU offloading of optimizer states and parameters to handle large models.
- **LoRA & QLoRA Support**:
  - Integrated `peft` library for parameter-efficient fine-tuning.
  - Support for 4-bit (QLoRA) training via `bitsandbytes` and `BitsAndBytesConfig`.
  - Automatic adapter management in `ModelHub` (listing and discovery).
  - Adapter-only saving in `TrainingOrchestrator` to optimize storage.
- **Model Hub**:
  - Centralized `ModelHub` for unified model inventory and lifecycle management.
  - New `StorageManager` for disk space monitoring and model size estimation.
  - Consolidated all model configurations into a single source of truth in `hub.py`.
  - Refactored `ModelRegistry` and `InferenceEngine` to delegate downloads and storage tracking to the Hub.
  - Standardized type hints across the entire model management stack for Python 3.9+ compatibility.

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
