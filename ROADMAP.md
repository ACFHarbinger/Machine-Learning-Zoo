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

- [x] **Model Hub**: Easy download/upload of pretrained weights.
- [x] **Visual Dashboard**: A React-based dashboard for monitoring training real-time (integrated with the main app).
- [x] **AutoML**: Automated hyperparameter tuning support via Optuna.

## Phase 4: Performance

- [x] **Quantization**: Native support for 4-bit/8-bit loading via `bitsandbytes`.
- [x] **Optimized Inference**: Integration with vLLM or similar libraries for high-throughput serving.
- [x] **Model Compression**: Pruning and knowledge distillation pipelines to reduce model size without significant accuracy loss.
- [x] **DeepSpeed Integration**: ZeRO-stage offloading for training large models on limited hardware.

## Phase 5: Fine-Tuning & Adaptation

- [x] **LoRA / QLoRA Support**: Parameter-efficient fine-tuning for LLMs and vision models.
- [x] **Prompt Engineering Toolkit**: Few-shot example management, prompt templates, and automated prompt optimization.
- [x] **Continual Learning**: Strategies for updating models on new data without catastrophic forgetting (EWC, Experience Replay).
- [x] **Domain Adaptation**: Utilities for distribution alignment (MMD loss) and adversarial training (GRL, DANN).

## Phase 6: Explainability & Evaluation

- [x] **Model Explainability**: Integrated Gradients and attention visualization integrated into pipelines.
- [x] **Comprehensive Evaluation Framework**: Unified metrics for classification, regression, and generative tasks (evaluator.py).
- [x] **Experiment Tracking**: Deeper MLflow/Weights & Biases integration for run comparison and artifact logging.
- [x] **Bias & Fairness Auditing**: Tools to detect and measure model bias across protected attributes.

## Phase 7: Production & Deployment

- [x] **Model Serving**: REST/gRPC inference server with batching, health checks, and autoscaling hooks.
- [x] **Containerization**: Docker images and Helm charts for Kubernetes-based deployment.
- [x] **Edge Deployment**: ONNX and TensorRT export paths for on-device inference.
- [x] **A/B Testing Framework**: Traffic splitting and statistical analysis for comparing model versions in production.

## Phase 8: Advanced Research

- [x] **Federated Learning**: Privacy-preserving distributed training across decentralized data sources.
- [x] **Synthetic Data Generation**: GAN and diffusion-based augmentation pipelines for low-data regimes.
- [x] **Neuro-Symbolic Methods**: Hybrid architectures combining neural networks with symbolic reasoning (gated/residual/attention integration, differentiable rule encoder, logic program executor).
- [x] **Multi-Agent RL**: Cooperative and competitive multi-agent environments with MAPPO training, inter-agent communication, and centralized-training-decentralized-execution.

## Phase 9: Community & Ecosystem Growth

- [x] **Plugin System**: Third-party backbone/head registration via entry points.
- [x] **Interactive Notebooks**: Curated Jupyter notebook gallery for tutorials, benchmarks, and demos.
- [x] **Leaderboard & Benchmarks**: Standardized benchmark suites with a public leaderboard for community contributions.
- [x] **Comprehensive API Reference**: Auto-generated Sphinx docs with cross-referenced examples for every public module.

_Note: Timelines are subject to change based on community feedback and contribution._
