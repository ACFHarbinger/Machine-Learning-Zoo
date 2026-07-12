# Agents & Personas

[![Python](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/downloads/)
[![License: AGPL v3](https://img.shields.io/badge/License-AGPL_v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)

Machine Learning Zoo supports a flexible Agent and Persona system, allowing dynamic behavior configuration for the AI models.

## Concept

An **Agent** is an autonomous or semi-autonomous entity capable of executing tasks. In this system, agents are primarily driven by LLMs (Large Language Models) interacting through a `Personality` layer.

## Components

### 1. MlRequestHandler

The entry point for agent commands. It handles:

- `echo`: Health check.
- `set_persona`:Configures the active personality.
- `chat`: Main interaction loop (streaming or synchronous).

### 2. Personality

Defined in `personality.py`, a Personality includes:

- **System Prompt**: The core instructions defining the agent's behavior.
- **History**: Conversion context management.
- **Capabilities**: Enabled tools or modes (e.g., coding, chatting).

### 3. Inference Engine

The `InferenceEngine` executes the agent's logic. It supports:

- **Local Providers**: Running models like `deepseek-r1-32b` or Llama via built-in loaders.
- **Remote Providers**: Connecting to Gemini/Anthropic via API (if configured).

## Defining a New Agent

To define a new agent style/persona, you can construct a JSON object passed to the `set_persona` method:

```json
{
  "system_prompt": "You are a helpful coding assistant specialized in Rust.",
  "parameters": {
    "temperature": 0.7,
    "max_tokens": 1024
  }
}
```

## Interaction

Agents communicate via NDJSON over stdin/stdout:

```json
{
  "method": "chat",
  "id": "1",
  "params": { "message": "Hello, agent!", "history": [] }
}
```
