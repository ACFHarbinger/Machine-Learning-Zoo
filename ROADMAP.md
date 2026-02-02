# Roadmap

[![License: AGPL v3](https://img.shields.io/badge/License-AGPL_v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)

This document outlines the high-level goals and planned features for the Machine Learning Zoo.

## Phase 1: Foundation (Current)

- [x] **Modular Architecture**: Establish Backbone/Head/Pipeline structure.
- [x] **Basic RL Support**: PPO implementation.
- [x] **Sidecar Interface**: Robust NDJSON IPC for agent integration.
- [/] **Documentation**: Comprehensive guides and API docs.

## Phase 2: Enhanced Capabilities

- [x] **Advanced LLM Support**: Better integration for deepseek, llama-3, and other open-source models.
- [x] **Multi-Modal Pipelines**: Support for Image-to-Text via `MultiModalBackbone` and integrated vision encoders (CLIP).
- [x] **Distributed Training**: Full support for `DistributedDataParallel` (DDP) in `training/` via PyTorch Lightning.

## Phase 3: Ecosystem Integration

- [ ] **Model Hub**: Easy download/upload of pretrained weights.
- [ ] **Visual Dashboard**: A React-based dashboard for monitoring training real-time (integrated with the main app).
- [ ] **AutoML**: Automated hyperparameter tuning support.

## Phase 4: Performance

- [x] **Quantization**: Native support for 4-bit/8-bit loading via `bitsandbytes`.
- [ ] **Optimized Inference**: Integration with vLLM or similar libraries for high-throughput serving.
- [ ] **Model Compression**: Pruning and knowledge distillation pipelines to reduce model size without significant accuracy loss.
- [ ] **DeepSpeed Integration**: ZeRO-stage offloading for training large models on limited hardware.

## Phase 5: Fine-Tuning & Adaptation

- [ ] **LoRA / QLoRA Support**: Parameter-efficient fine-tuning for LLMs and vision models.
- [ ] **Prompt Engineering Toolkit**: Few-shot example management, prompt templates, and automated prompt optimization.
- [ ] **Continual Learning**: Strategies for updating models on new data without catastrophic forgetting.
- [ ] **Domain Adaptation**: Transfer learning utilities for shifting between data distributions.

## Phase 6: Explainability & Evaluation

- [ ] **Model Explainability**: SHAP, LIME, and attention visualization integrated into pipelines.
- [ ] **Comprehensive Evaluation Framework**: Unified metrics for classification, regression, NLP, RL, and generative tasks.
- [ ] **Experiment Tracking**: Deeper MLflow/Weights & Biases integration for run comparison and artifact logging.
- [ ] **Bias & Fairness Auditing**: Tools to detect and measure model bias across protected attributes.

## Phase 7: Production & Deployment

- [ ] **Model Serving**: REST/gRPC inference server with batching, health checks, and autoscaling hooks.
- [ ] **Containerization**: Docker images and Helm charts for Kubernetes-based deployment.
- [ ] **Edge Deployment**: ONNX and TensorRT export paths for on-device inference.
- [ ] **A/B Testing Framework**: Traffic splitting and statistical analysis for comparing model versions in production.

## Phase 8: Advanced Research

- [ ] **Federated Learning**: Privacy-preserving distributed training across decentralized data sources.
- [ ] **Synthetic Data Generation**: GAN and diffusion-based augmentation pipelines for low-data regimes.
- [ ] **Neuro-Symbolic Methods**: Hybrid architectures combining neural networks with symbolic reasoning.
- [ ] **Multi-Agent RL**: Cooperative and competitive multi-agent environments building on the existing RL infrastructure.

## Phase 9: Community & Ecosystem Growth

- [ ] **Plugin System**: Third-party backbone/head registration via entry points.
- [ ] **Interactive Notebooks**: Curated Jupyter notebook gallery for tutorials, benchmarks, and demos.
- [ ] **Leaderboard & Benchmarks**: Standardized benchmark suites with a public leaderboard for community contributions.
- [ ] **Comprehensive API Reference**: Auto-generated Sphinx docs with cross-referenced examples for every public module.

_Note: Timelines are subject to change based on community feedback and contribution._
