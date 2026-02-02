"""Unified Model Hub for managing models and checkpoints."""

import asyncio
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from .storage import StorageManager

logger = logging.getLogger(__name__)

# Shared configurations moved from sidecar_registry.py
MODEL_CONFIGS: Dict[str, Dict[str, Any]] = {
    "deepseek-r1-32b": {
        "loader": "gguf",
        "path": "deepseek-r1-distill-qwen-32b.Q4_K_M.gguf",
        "n_gpu_layers": 40,
        "n_ctx": 2048,
        "hf_repo": "bartowski/DeepSeek-R1-Distill-Qwen-32B-GGUF",
        "hf_file": "DeepSeek-R1-Distill-Qwen-32B-Q4_K_M.gguf",
    },
    "llama-3.3-70b": {
        "loader": "gguf",
        "path": "llama-3.3-70b-instruct.Q4_K_M.gguf",
        "n_gpu_layers": 35,
        "n_ctx": 2048,
        "hf_repo": "bartowski/Llama-3.3-70B-Instruct-GGUF",
        "hf_file": "Llama-3.3-70B-Instruct-Q4_K_M.gguf",
    },
    "llama-3-8b-instruct": {
        "loader": "transformers",
        "hf_repo": "meta-llama/Meta-Llama-3-8B-Instruct",
        "device_map": "auto",
        "torch_dtype": "float16",
        "load_in_4bit": True,
    },
    "llama-3.1-8b-instruct": {
        "loader": "transformers",
        "hf_repo": "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "device_map": "auto",
        "torch_dtype": "bfloat16",
        "load_in_4bit": True,
    },
    "deepseek-v3-small": {
        "loader": "transformers",
        "hf_repo": "deepseek-ai/DeepSeek-V3",
        "device_map": "auto",
        "torch_dtype": "bfloat16",
        "load_in_4bit": True,
    },
    "deepseek-r1-distill-llama-8b": {
        "loader": "transformers",
        "hf_repo": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        "device_map": "auto",
        "torch_dtype": "bfloat16",
        "load_in_4bit": True,
    },
    "llava-v1.5-7b": {
        "loader": "multimodal",
        "vision_config": {
            "hf_repo": "openai/clip-vit-large-patch14-336",
            "device_map": "auto",
            "torch_dtype": "float16",
        },
        "llm_config": {
            "hf_repo": "lmsys/vicuna-7b-v1.5",
            "device_map": "auto",
            "torch_dtype": "float16",
            "load_in_4bit": True,
        },
    },
    # Legacy configurations
    "phi-2-dpo-v7": {
        "loader": "gguf",
        "path": "phi-2-dpo-v7.Q4_K_M.gguf",
        "hf_repo": "TheBloke/phi-2-dpo-v7-GGUF",
        "hf_file": "phi-2-dpo-v7.Q4_K_M.gguf",
        "n_gpu_layers": 20,
    },
    "llama-2-7b-chat": {
        "loader": "gguf",
        "path": "llama-2-7b-chat.Q4_K_M.gguf",
        "hf_repo": "TheBloke/Llama-2-7B-Chat-GGUF",
        "hf_file": "llama-2-7b-chat.Q4_K_M.gguf",
        "n_gpu_layers": 20,
    },
    "mistral-7b-instruct-v0.2": {
        "loader": "gguf",
        "path": "mistral-7b-instruct-v0.2.Q4_K_M.gguf",
        "hf_repo": "TheBloke/Mistral-7B-Instruct-v0.2-GGUF",
        "hf_file": "mistral-7b-instruct-v0.2.Q4_K_M.gguf",
        "n_gpu_layers": 20,
    },
    "neural-chat-7b-v3-1": {
        "loader": "gguf",
        "path": "neural-chat-7b-v3-1.Q4_K_M.gguf",
        "hf_repo": "TheBloke/neural-chat-7B-v3-1-GGUF",
        "hf_file": "neural-chat-7b-v3-1.Q4_K_M.gguf",
        "n_gpu_layers": 20,
    },
    "openhermes-2.5-mistral-7b": {
        "loader": "gguf",
        "path": "openhermes-2.5-mistral-7b.Q4_K_M.gguf",
        "hf_repo": "TheBloke/OpenHermes-2.5-Mistral-7B-GGUF",
        "hf_file": "openhermes-2.5-mistral-7b.Q4_K_M.gguf",
        "n_gpu_layers": 20,
    },
    "zephyr-7b-beta": {
        "loader": "gguf",
        "path": "zephyr-7b-beta.Q4_K_M.gguf",
        "hf_repo": "TheBloke/zephyr-7B-beta-GGUF",
        "hf_file": "zephyr-7b-beta.Q4_K_M.gguf",
        "n_gpu_layers": 20,
    },
    "tinyllama-1.1b-chat-v1.0": {
        "loader": "gguf",
        "path": "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
        "hf_repo": "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF",
        "hf_file": "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
        "n_gpu_layers": 0,
    },
}


class ModelHub:
    """Central interface for model inventory and storage management."""

    def __init__(self, models_dir: Optional[Path] = None):
        """Initialize the hub."""
        self.models_dir = models_dir or Path.home() / ".pi-assistant" / "models"
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.storage = StorageManager(self.models_dir)

    def list_configs(self) -> Dict[str, Dict[str, Any]]:
        """List all pre-configured models."""
        return MODEL_CONFIGS

    def list_local(self) -> List[Dict[str, Any]]:
        """List models currently stored on disk."""
        local_models = []

        # Check configured GGUF models
        for model_id, cfg in MODEL_CONFIGS.items():
            if cfg.get("loader") == "gguf":
                path = self.models_dir / str(cfg.get("path", ""))
                if path.exists():
                    local_models.append(
                        {
                            "model_id": model_id,
                            "path": str(path),
                            "type": "gguf",
                            "size": path.stat().st_size,
                        }
                    )
            elif cfg.get("loader") == "transformers":
                # For transformers, we check if the repo is cached (simplified)
                # In a real scenario, we'd check the cache directory
                local_models.append(
                    {
                        "model_id": model_id,
                        "type": "transformers",
                        "repo": cfg.get("hf_repo"),
                    }
                )

        # Check adapters directory
        adapters_dir = self.models_dir / "adapters"
        if adapters_dir.exists():
            for adapter_path in adapters_dir.iterdir():
                if (
                    adapter_path.is_dir()
                    and (adapter_path / "adapter_config.json").exists()
                ):
                    local_models.append(
                        {
                            "id": adapter_path.name,
                            "path": str(adapter_path),
                            "type": "adapter",
                            "base_model": self._get_base_model_from_adapter(
                                adapter_path
                            ),
                        }
                    )

        return local_models

    def list_adapters(self) -> List[Dict[str, Any]]:
        """List locally saved LoRA adapters."""
        return [m for m in self.list_local() if m.get("type") == "adapter"]

    def _get_base_model_from_adapter(self, adapter_path: Path) -> Optional[str]:
        """Extract base model name from adapter config."""
        import json

        config_path = adapter_path / "adapter_config.json"
        try:
            with open(config_path, "r") as f:
                data = json.load(f)
                return str(data.get("base_model_name_or_path"))
        except Exception:
            return None

    async def download(
        self,
        model_id: str,
        progress_callback: Optional[Callable[[Dict[str, Any]], Any]] = None,
    ) -> Dict[str, Any]:
        """Download a model to local storage."""
        config = MODEL_CONFIGS.get(model_id)
        if not config:
            raise ValueError(f"Unknown model: {model_id}")

        loader = config.get("loader", "gguf")

        if loader == "gguf":
            return await self._download_gguf(model_id, config, progress_callback)
        else:
            # Transformers usually handles its own caching via from_pretrained
            logger.info(
                "Transformers model %s will be handled by the backbone loader", model_id
            )
            return {"status": "delegated", "model_id": model_id}

    async def _download_gguf(
        self,
        model_id: str,
        config: Dict[str, Any],
        progress_callback: Optional[Callable[[Dict[str, Any]], Any]] = None,
    ) -> Dict[str, Any]:
        """Specific logic for GGUF downloading."""
        hf_repo = config.get("hf_repo")
        hf_file = config.get("hf_file")
        if not hf_repo or not hf_file:
            raise ValueError(f"No download info for {model_id}")

        dest_name = config.get("path")
        dest_path = self.models_dir / str(dest_name)

        if dest_path.exists():
            return {"status": "exists", "path": str(dest_path)}

        if progress_callback:
            await progress_callback({"status": "starting", "model_id": model_id})

        def _run_download():
            from huggingface_hub import hf_hub_download

            path = hf_hub_download(
                repo_id=hf_repo,
                filename=hf_file,
                local_dir=str(self.models_dir),
                local_dir_use_symlinks=False,
            )
            # Rename if needed
            dl_path = Path(path)
            if dl_path.name != dest_name:
                final_path = self.models_dir / str(dest_name)
                dl_path.rename(final_path)
                return str(final_path)
            return path

        loop = asyncio.get_event_loop()
        local_path = await loop.run_in_executor(None, _run_download)

        if progress_callback:
            await progress_callback(
                {"status": "complete", "model_id": model_id, "path": local_path}
            )

        return {"status": "downloaded", "path": local_path}

    def delete(self, model_id: str) -> bool:
        """Delete a local model file."""
        config = MODEL_CONFIGS.get(model_id)
        if not config or "path" not in config:
            # Try to delete by ID if it's a direct file
            path = self.models_dir / model_id
        else:
            path = self.models_dir / str(config["path"])

        if path.exists():
            if path.is_file():
                path.unlink()
            else:
                import shutil

                shutil.rmtree(path)
            logger.info("Deleted model storage: %s", path)
            return True
        return False
