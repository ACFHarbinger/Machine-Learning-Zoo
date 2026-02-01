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
        "n_gpu_layers": 40,  # Safer partial offload for 24GB VRAM with system overhead
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

    def __init__(
        self,
        models_dir: Path | None = None,
        device_manager: Any | None = None,
    ):
        """
        Initialize the model registry.
        Args:
            models_dir: The directory to store models.
            device_manager: Optional DeviceManager for device-aware placement.
        """
        self.models_dir = models_dir or Path.home() / ".pi-assistant" / "models"
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self._loaded: dict[str, LoadedModel] = {}
        self.device_manager = device_manager

    def list_models(self) -> list[dict]:
        """
        List available models with device placement info.
        Returns:
            A list of dictionaries containing the model information.
        """
        models: list[dict[str, Any]] = []

        def _device_info(model_id: str) -> dict[str, Any]:
            """Get device and size info for a loaded model."""
            loaded = self._loaded.get(model_id)
            if loaded:
                return {
                    "device": loaded.device,
                    "model_size_mb": loaded.model_size_mb,
                }
            return {"device": None, "model_size_mb": None}

        # List models in models_dir
        for item in self.models_dir.iterdir():
            if item.is_dir():
                info = _device_info(item.name)
                models.append(
                    {
                        "model_id": item.name,
                        "path": str(item),
                        "loaded": item.name in self._loaded,
                        "downloaded": True,
                        "backend": "transformers",
                        **info,
                    }
                )
            elif item.suffix == ".gguf":
                info = _device_info(item.name)
                models.append(
                    {
                        "model_id": item.name,
                        "path": str(item),
                        "loaded": item.name in self._loaded,
                        "downloaded": True,
                        "backend": "llama.cpp",
                        **info,
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
                        "device": model.device,
                        "model_size_mb": model.model_size_mb,
                    }
                )

        # Add configured models
        for model_id, cfg in MODEL_CONFIGS.items():
            if not any(m["model_id"] == model_id for m in models):
                gguf_path = self.models_dir / str(cfg["path"])
                info = _device_info(model_id)
                models.append(
                    {
                        "model_id": model_id,
                        "path": cfg["path"],
                        "loaded": model_id in self._loaded,
                        "downloaded": gguf_path.exists(),
                        "backend": "llama.cpp",
                        "hf_repo": cfg.get("hf_repo"),
                        **info,
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

        if is_gguf:
            n_gpu_layers: int = 0  # Default to CPU for safety
            n_ctx: int = 2048

            # Try to find config by ID or by path/filename
            config = MODEL_CONFIGS.get(model_id)
            if not config:
                # Search configs for matching path
                for cfg_id, cfg in MODEL_CONFIGS.items():
                    if cfg.get("path") == model_id or Path(cfg.get("path", "")).name == model_id:
                        config = cfg
                        model_id = cfg_id
                        break

            # Check for custom config
            if config:
                # If path exists relative to models_dir, use full path, otherwise use name
                config_path = str(config.get("path", ""))
                cfg_path = Path(self.models_dir) / config_path
                path = str(cfg_path) if cfg_path.exists() else config_path
                n_gpu_layers = int(config.get("n_gpu_layers", 0))
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

            device = "cuda:0" if n_gpu_layers > 0 else "cpu"
            loaded = LoadedModel(
                model_id=model_id,
                model=model,
                tokenizer=None,
                backend="llama.cpp",
                device=device,
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

            # Device-aware placement
            size_mb = self._estimate_model_size(model)
            if self.device_manager:
                target_device = self.device_manager.best_device_for("inference", size_mb)
            else:
                import torch

                target_device = "cuda:0" if torch.cuda.is_available() else "cpu"
            model = model.to(target_device)
            logger.info("Placed transformers model %s on %s (%d MB)", model_id, target_device, size_mb)

            loaded = LoadedModel(
                model_id=model_id,
                model=model,
                tokenizer=tokenizer,
                backend="transformers",
                device=target_device,
                model_size_mb=size_mb,
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
            import gc

            gc.collect()
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass
            logger.info("Unloaded model: %s", model_id)
            return True
        return False

    async def migrate_model(self, model_id: str, target_device: str) -> dict[str, Any]:
        """
        Move a loaded model to a different device at runtime.

        Args:
            model_id: The model to migrate.
            target_device: Target device string, e.g. "cpu", "cuda:0".

        Returns:
            Dict with status, model_id, previous_device, new_device.
        """
        loaded = self._loaded.get(model_id)
        if loaded is None:
            raise ValueError(f"Model not loaded: {model_id}")

        prev_device = loaded.device
        if prev_device == target_device:
            return {
                "status": "already_on_device",
                "model_id": model_id,
                "device": target_device,
            }

        import gc

        if loaded.backend == "llama.cpp":
            # llama.cpp doesn't support .to() — must reload with different n_gpu_layers
            from llama_cpp import Llama  # type: ignore

            old_model = loaded.model
            model_path = old_model.model_path
            n_ctx = old_model.n_ctx()
            n_gpu_layers = -1 if "cuda" in target_device else 0

            del old_model
            gc.collect()
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass

            new_model = Llama(
                model_path=model_path,
                n_gpu_layers=n_gpu_layers,
                n_ctx=n_ctx,
                verbose=True,
            )
            loaded.model = new_model
            loaded.device = target_device
        else:
            # transformers: straightforward .to()
            import torch

            loaded.model = loaded.model.to(target_device)
            loaded.device = target_device
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        logger.info("Migrated %s: %s -> %s", model_id, prev_device, target_device)
        return {
            "status": "migrated",
            "model_id": model_id,
            "previous_device": prev_device,
            "new_device": target_device,
        }

    @staticmethod
    def _estimate_model_size(model: Any) -> int:
        """Estimate model size in MB from parameter count."""
        try:
            param_bytes = sum(p.nelement() * p.element_size() for p in model.parameters())
            return param_bytes // (1024 * 1024)
        except Exception:
            return 0

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
