"""
Model constants and configurations.
"""

from typing import Any, Dict, List

from ..enums.models import DeepModelType, MacModelType

# Configurations for specific hardware (24GB VRAM)
MODEL_CONFIGS: Dict[str, Dict[str, Any]] = {
    "phi-2-dpo-v7": {
        "path": "phi-2-dpo-v7.Q4_K_M.gguf",
        "hf_repo": "TheBloke/phi-2-dpo-v7-GGUF",
        "hf_file": "phi-2-dpo-v7.Q4_K_M.gguf",
        "n_gpu_layers": 20,
    },
    "llama-2-7b-chat": {
        "path": "llama-2-7b-chat.Q4_K_M.gguf",
        "hf_repo": "TheBloke/Llama-2-7B-Chat-GGUF",
        "hf_file": "llama-2-7b-chat.Q4_K_M.gguf",
        "n_gpu_layers": 20,
    },
    "mistral-7b-instruct-v0.2": {
        "path": "mistral-7b-instruct-v0.2.Q4_K_M.gguf",
        "hf_repo": "TheBloke/Mistral-7B-Instruct-v0.2-GGUF",
        "hf_file": "mistral-7b-instruct-v0.2.Q4_K_M.gguf",
        "n_gpu_layers": 20,
    },
    "neural-chat-7b-v3-1": {
        "path": "neural-chat-7b-v3-1.Q4_K_M.gguf",
        "hf_repo": "TheBloke/neural-chat-7B-v3-1-GGUF",
        "hf_file": "neural-chat-7b-v3-1.Q4_K_M.gguf",
        "n_gpu_layers": 20,
    },
    "openhermes-2.5-mistral-7b": {
        "path": "openhermes-2.5-mistral-7b.Q4_K_M.gguf",
        "hf_repo": "TheBloke/OpenHermes-2.5-Mistral-7B-GGUF",
        "hf_file": "openhermes-2.5-mistral-7b.Q4_K_M.gguf",
        "n_gpu_layers": 20,
    },
    "zephyr-7b-beta": {
        "path": "zephyr-7b-beta.Q4_K_M.gguf",
        "hf_repo": "TheBloke/zephyr-7B-beta-GGUF",
        "hf_file": "zephyr-7b-beta.Q4_K_M.gguf",
        "n_gpu_layers": 20,
    },
    "tinyllama-1.1b-chat-v1.0": {
        "path": "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
        "hf_repo": "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF",
        "hf_file": "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
        "n_gpu_layers": 0,  # Small model, CPU is fine
    },
    "deepseek-r1-32b": {
        "loader": "gguf",
        "path": "deepseek-r1-distill-qwen-32b.Q4_K_M.gguf",
        "n_gpu_layers": 40,
        "n_ctx": 2048,
        "hf_repo": "bartowski/DeepSeek-R1-Distill-Qwen-32B-GGUF",
        "hf_file": "DeepSeek-R1-Distill-Qwen-32B-Q4_K_M.gguf",
        "license": "MIT",
        "author": "DeepSeek",
    },
    "llama-3.3-70b": {
        "loader": "gguf",
        "path": "llama-3.3-70b-instruct.Q4_K_M.gguf",
        "n_gpu_layers": 35,
        "n_ctx": 2048,
        "hf_repo": "bartowski/Llama-3.3-70B-Instruct-GGUF",
        "hf_file": "Llama-3.3-70B-Instruct-Q4_K_M.gguf",
        "license": "Llama 3.3",
        "author": "Meta",
    },
    "llama-3-8b-instruct": {
        "loader": "transformers",
        "hf_repo": "meta-llama/Meta-Llama-3-8B-Instruct",
        "device_map": "auto",
        "torch_dtype": "float16",
        "load_in_4bit": True,
        "license": "Llama 3",
        "author": "Meta",
    },
    "llama-3.1-8b-instruct": {
        "loader": "transformers",
        "hf_repo": "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "device_map": "auto",
        "torch_dtype": "bfloat16",
        "load_in_4bit": True,
        "license": "Llama 3.1",
        "author": "Meta",
    },
    "deepseek-v3-small": {
        "loader": "transformers",
        "hf_repo": "deepseek-ai/DeepSeek-V3",
        "device_map": "auto",
        "torch_dtype": "bfloat16",
        "load_in_4bit": True,
        "license": "MIT",
        "author": "DeepSeek",
    },
    "deepseek-r1-distill-llama-8b": {
        "loader": "transformers",
        "hf_repo": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        "device_map": "auto",
        "torch_dtype": "bfloat16",
        "load_in_4bit": True,
        "license": "MIT",
        "author": "DeepSeek",
    },
}


# List of MAC model names
MAC_MODEL_NAMES: List[str] = [m.value for m in MacModelType]

# List of deep model names
DEEP_MODEL_NAMES: List[str] = [m.value for m in DeepModelType]
