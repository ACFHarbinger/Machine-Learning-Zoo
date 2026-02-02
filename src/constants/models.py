# Configurations for specific hardware (24GB VRAM)
MODEL_CONFIGS = {
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
}
