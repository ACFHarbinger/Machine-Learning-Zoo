"""Model registry for loading and managing models."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

try:
    from ..configs.sidecar_model import LoadedModel
except ImportError:
    # Handle direct execution or testing
    from pi_sidecar.configs.sidecar_model import LoadedModel

logger = logging.getLogger(__name__)


# Configurations for specific hardware (24GB VRAM)
MODEL_CONFIGS: dict[str, dict[str, Any]] = {
    "deepseek-r1-32b": {
        "path": "deepseek-r1-distill-qwen-32b.Q4_K_M.gguf",
        "n_gpu_layers": 50,  # Partial offload to leave room for KV cache and overhead
        "n_ctx": 2048,
        "hf_repo": "bartowski/DeepSeek-R1-Distill-Qwen-32B-GGUF",
        "hf_file": "DeepSeek-R1-Distill-Qwen-32B-Q4_K_M.gguf",
    },
    "llama-3.3-70b": {
        "path": "llama-3.3-70b-instruct.Q4_K_M.gguf",
        "n_gpu_layers": 35,  # Partial offload to fit in 24GB VRAM
        "n_ctx": 2048,
        "hf_repo": "bartowski/Llama-3.3-70B-Instruct-GGUF",
        "hf_file": "Llama-3.3-70B-Instruct-Q4_K_M.gguf",
    },
}


class ModelRegistry:
    """Registry for managing loaded models."""

    def __init__(self, models_dir: Path | None = None):
        """
        Initialize the model registry.
        Args:
            models_dir: The directory to store models.
        """
        self.models_dir = models_dir or Path.home() / ".pi-assistant" / "models"
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self._loaded: dict[str, LoadedModel] = {}

    def list_models(self) -> list[dict]:
        """
        List available models.
        Returns:
            A list of dictionaries containing the model information.
        """
        models: list[dict[str, Any]] = []

        # List models in models_dir
        for item in self.models_dir.iterdir():
            if item.is_dir():
                models.append(
                    {
                        "model_id": item.name,
                        "path": str(item),
                        "loaded": item.name in self._loaded,
                        "downloaded": True,
                        "backend": "transformers",
                    }
                )
            elif item.suffix == ".gguf":
                models.append(
                    {
                        "model_id": item.name,
                        "path": str(item),
                        "loaded": item.name in self._loaded,
                        "downloaded": True,
                        "backend": "llama.cpp",
                    }
                )

        # Add loaded models not in directory
        for model_id, model in self._loaded.items():
            if not any(m["model_id"] == model_id for m in models):
                models.append(
                    {
                        "model_id": model_id,
                        "path": None,
                        "loaded": True,
                        "downloaded": True,
                        "backend": getattr(model, "backend", "unknown"),
                    }
                )

        # Add configured models
        for model_id, cfg in MODEL_CONFIGS.items():
            if not any(m["model_id"] == model_id for m in models):
                gguf_path = self.models_dir / str(cfg["path"])
                models.append(
                    {
                        "model_id": model_id,
                        "path": cfg["path"],
                        "loaded": model_id in self._loaded,
                        "downloaded": gguf_path.exists(),
                        "backend": "llama.cpp",
                        "hf_repo": cfg.get("hf_repo"),
                    }
                )

        return models

    async def load_model(self, model_id: str, backend: str | None = None) -> LoadedModel:
        """
        Load a model by ID.
        Args:
            model_id: The model ID to load.
            backend: Optional backend override ("transformers" or "llama.cpp")
        Returns:
            A LoadedModel object containing the model and tokenizer.
        """
        if model_id in self._loaded:
            logger.info("Model already loaded: %s", model_id)
            return self._loaded[model_id]

        logger.info("Loading model: %s (backend: %s)", model_id, backend or "auto")

        # Check path and config
        dir_path = Path(self.models_dir)
        model_path = dir_path / str(model_id)

        # Determine if it's GGUF from ID, path, or config
        is_gguf = (
            model_id.endswith(".gguf")
            or model_path.suffix == ".gguf"
            or (
                model_id in MODEL_CONFIGS
                and str(MODEL_CONFIGS[model_id].get("path", "")).endswith(".gguf")
            )
        )

        # Decide backend
        effective_backend = backend
        if not effective_backend:
            effective_backend = "llama.cpp" if is_gguf else "transformers"

        if effective_backend == "llama.cpp":
            from llama_cpp import Llama  # type: ignore

            n_gpu_layers: int = -1  # Default to GPU if not specified
            n_ctx: int = 2048

            # Check for custom config
            if model_id in MODEL_CONFIGS:
                config: dict[str, Any] = MODEL_CONFIGS[model_id]
                # If path exists relative to models_dir, use full path, otherwise use name
                config_path = str(config.get("path", ""))
                cfg_path = Path(self.models_dir) / config_path
                path = str(cfg_path) if cfg_path.exists() else config_path

                n_gpu_layers = int(config.get("n_gpu_layers", -1))
                n_ctx = int(config.get("n_ctx", 2048))

                logger.info(
                    "Loading configured model %s with n_gpu_layers=%s, n_ctx=%s",
                    model_id,
                    n_gpu_layers,
                    n_ctx,
                )
            else:
                path = str(model_path) if model_path.exists() else model_id
                logger.info("Loading GGUF model from %s", path)

            try:
                model = Llama(
                    model_path=path,
                    n_gpu_layers=n_gpu_layers,
                    n_ctx=n_ctx,
                    verbose=True,
                )
            except ValueError as e:
                if "Failed to create llama_context" in str(e) and n_gpu_layers != 0:
                    logger.warning(
                        "Failed to create Llama context with GPU acceleration. Falling back to CPU."
                    )
                    model = Llama(
                        model_path=path,
                        n_gpu_layers=0,
                        n_ctx=n_ctx,
                        verbose=True,
                    )
                else:
                    raise e

            loaded = LoadedModel(
                model_id=model_id,
                model=model,
                tokenizer=None,
                backend="llama.cpp",
            )
        else:
            from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore

            # Check if it's a local path or HuggingFace ID
            if model_path.exists():
                path = str(model_path)
            else:
                path = model_id  # Treat as HuggingFace model ID

            tokenizer = AutoTokenizer.from_pretrained(path)
            model = AutoModelForCausalLM.from_pretrained(path)

            loaded = LoadedModel(
                model_id=model_id,
                model=model,
                tokenizer=tokenizer,
                backend="transformers",
            )

        self._loaded[model_id] = loaded
        return loaded

    def get_model(self, model_id: str) -> LoadedModel | None:
        """
        Get a loaded model, or None if not loaded.
        Args:
            model_id: The model ID to get.
        Returns:
            A LoadedModel object containing the model and tokenizer.
        """
        return self._loaded.get(model_id)

    def unload_model(self, model_id: str) -> bool:
        """
        Unload a model to free memory.
        Args:
            model_id: The model ID to unload.
        Returns:
            A boolean indicating whether the model was unloaded.
        """
        if model_id in self._loaded:
            del self._loaded[model_id]
            logger.info("Unloaded model: %s", model_id)
            return True
        return False

    async def download_model(
        self, model_id: str, progress_callback: Callable[[dict[str, Any]], Any] | None = None
    ) -> dict[str, Any]:
        """
        Download a model GGUF file from HuggingFace.
        Args:
            model_id: The model ID from MODEL_CONFIGS.
            progress_callback: Optional async callback for progress updates.
        Returns:
            A dictionary with download status and local path.
        """
        config = MODEL_CONFIGS.get(model_id)
        if not config:
            raise ValueError(
                f"Unknown model ID: {model_id}. Available: {list(MODEL_CONFIGS.keys())}"
            )

        hf_repo = config.get("hf_repo")
        hf_file = config.get("hf_file")
        if not hf_repo or not hf_file:
            raise ValueError(f"Model {model_id} has no HuggingFace download info configured")

        dest_path = Path(self.models_dir) / str(config["path"])
        if dest_path.exists():
            logger.info("Model %s already downloaded at %s", model_id, dest_path)
            return {"status": "already_downloaded", "model_id": model_id, "path": str(dest_path)}

        logger.info("Downloading %s from %s/%s", model_id, hf_repo, hf_file)

        if progress_callback:
            await progress_callback({"status": "downloading", "model_id": model_id, "progress": 0})

        def _download() -> str:
            from huggingface_hub import hf_hub_download  # type: ignore

            downloaded_path = hf_hub_download(
                repo_id=hf_repo,
                filename=hf_file,
                local_dir=str(self.models_dir),
                local_dir_use_symlinks=False,
            )
            # Rename to our expected filename if different
            dl = Path(downloaded_path)
            if dl.name != config["path"]:
                target = self.models_dir / str(config["path"])
                dl.rename(target)
                return str(target)
            return downloaded_path

        loop = asyncio.get_event_loop()
        local_path = await loop.run_in_executor(None, _download)

        logger.info("Download complete: %s -> %s", model_id, local_path)

        if progress_callback:
            await progress_callback({"status": "complete", "model_id": model_id, "progress": 100})

        return {"status": "downloaded", "model_id": model_id, "path": local_path}
