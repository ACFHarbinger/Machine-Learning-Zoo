# Architecture

[![Python](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/downloads/)
[![License: AGPL v3](https://img.shields.io/badge/License-AGPL_v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)

Machine Learning Zoo is designed with a modular architecture to facilitate rapid experimentation and deployment of machine learning models. The system is composed of several key layers.

## High-Level Overview

The system operates as a hybrid application with a core Python-based Machine Learning Sidecar that communicates with a wider system (likely Rust/React based on integration patterns) via IPC.

### Core Components

1.  **Inference Engine (`InferenceEngine`)**:
    - The central orchestrator for model operations.
    - Manages requests for text completion and streaming.
    - Routes requests to appropriate providers (Local, API-based).

2.  **Model Registry (`ModelRegistry`)**:
    - Manages the lifecycle of ML models.
    - Handles loading, unloading, and initialization of models on specific devices (CPU/GPU).

3.  **Transport Layer (`NdjsonTransport`)**:
    - Implements an asynchronous NDJSON (Newline Delimited JSON) protocol over Stdin/Stdout.
    - Allows the Python sidecar to function as a subprocess of a main application.

4.  **Device Management (`DeviceManager`)**:
    - Probes and manages hardware resources (CUDA, MPS, CPU).
    - Ensures models are allocated to the optimal available hardware.

## Directory Structure

- `models/`: Contains model definitions (Backbones, Heads).
- `pipeline/`: Data processing pipelines.
- `configs/`: Hydra/OmegaConf configurations.
- `api/`: Interface definitions.
- `ml_sidecar_main.py`: Entry point for the sidecar service.

## Data Flow

1.  **Request**: An IPC request (JSON) calls a method (e.g., `chat`).
2.  **Handler**: `MlRequestHandler` parses the request.
3.  **Engine**: `InferenceEngine` prepares the context (Persona, History).
4.  **Model**: The request is processed by the loaded LLM or custom model.
5.  **Response**: Output is streamed back via `NdjsonTransport`.
