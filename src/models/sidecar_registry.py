"""Model registry for loading and managing models."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from ..configs.sidecar_model import LoadedModel

logger = logging.getLogger(__name__)


from .hub import ModelHub, MODEL_CONFIGS


class ModelRegistry:
    """Registry for managing loaded models."""

    def __init__(
        self,
        models_dir: Optional[Path] = None,
        device_manager: Optional[Any] = None,
    ):
        """
        Initialize the model registry.
        Args:
            models_dir: The directory to store models.
            device_manager: Optional DeviceManager for device-aware placement.
        """
        self.models_dir = models_dir or Path.home() / ".pi-assistant" / "models"
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.hub = ModelHub(self.models_dir)
        self._loaded: Dict[str, LoadedModel] = {}
        self.device_manager = device_manager

    def list_models(self) -> List[Dict[str, Any]]:
        """List available local models."""
        return self.hub.list_local()

    async def load_model(
        self, model_id: str, backend: Optional[str] = None
    ) -> LoadedModel:
        """
        Load a model by ID.
        Args:
            model_id: The model ID to load.
            backend: Optional backend override ("transformers" or "llama.cpp")
        Returns:
            A LoadedModel object containing the model and tokenizer.
        """
        if model_id in self._loaded:
            logger.info("Model %s already loaded", model_id)
            return self._loaded[model_id]

        logger.info("Loading model: %s (backend: %s)", model_id, backend or "auto")

        config = MODEL_CONFIGS.get(model_id)

        # Fallback for direct GGUF files by name
        if not config and (
            model_id.endswith(".gguf") or (Path(self.models_dir) / model_id).exists()
        ):
            # Default config for direct files
            config = {"loader": "gguf", "path": model_id}

        if not config:
            raise ValueError(f"Unknown model identifier: {model_id}")

        loader_type = config.get("loader", "gguf")

        # Override backend if specified
        if backend == "llama.cpp":
            loader_type = "gguf"
        elif backend == "transformers":
            loader_type = "transformers"

        if loader_type == "gguf":
            await self._load_gguf(model_id, config)
        elif loader_type == "transformers":
            await self._load_hf(model_id, config)
        elif loader_type == "multimodal":
            await self._load_multimodal(model_id, config)
        else:
            raise ValueError(f"Unsupported loader type: {loader_type}")

        return self._loaded[model_id]

    async def _load_gguf(
        self,
        model_id: str,
        config: Dict[str, Any],
    ) -> None:
        """Load GGUF model using llama-cpp-python."""
        from llama_cpp import Llama  # type: ignore

        path = config["path"]
        model_path = self.models_dir / path

        # Check if we need to download
        if not model_path.exists():
            # If we have repo info, download it
            if "hf_repo" in config:
                await self.download_model(model_id)

        str_path = str(model_path)
        logger.info("Loading GGUF model from %s...", str_path)

        n_gpu_layers = int(config.get("n_gpu_layers", 0))
        n_ctx = int(config.get("n_ctx", 2048))

        def _load():
            try:
                return Llama(
                    model_path=str_path,
                    n_gpu_layers=n_gpu_layers,
                    n_ctx=n_ctx,
                    verbose=True,
                )
            except ValueError as e:
                # Fallback to CPU if GPU failed
                if "Failed to create llama_context" in str(e) and n_gpu_layers != 0:
                    logger.warning("GPU init failed, falling back to CPU")
                    return Llama(
                        model_path=str_path, n_gpu_layers=0, n_ctx=n_ctx, verbose=True
                    )
                raise e

        loop = asyncio.get_event_loop()
        llm = await loop.run_in_executor(None, _load)

        device = "cuda:0" if n_gpu_layers > 0 else "cpu"

        self._loaded[model_id] = LoadedModel(
            model=llm,
            tokenizer=None,
            config=config,
            model_id=model_id,
            loaded_at=__import__("time").time(),
            backend="llama.cpp",
            device=device,
        )

    async def _load_hf(
        self,
        model_id: str,
        config: Dict[str, Any],
    ) -> None:
        """Load Transformers model using huggingface/transformers."""
        from .backbones.hf_backbone import HuggingFaceBackbone  # type: ignore
        from .backbones.base import BackboneConfig
        import torch

        hf_repo = config["hf_repo"]
        device_map = config.get("device_map", "auto")
        # Handle string "float16" etc
        dtype_str = config.get("torch_dtype", "float16")
        torch_dtype = getattr(torch, dtype_str)
        load_in_8bit = config.get("load_in_8bit", False)
        load_in_4bit = config.get("load_in_4bit", False)

        logger.info("Loading Transformers model from %s...", hf_repo)

        def _load_hf_inner():
            backbone_config = BackboneConfig(
                extra={
                    "hf_repo": hf_repo,
                    "device_map": device_map,
                    "torch_dtype": torch_dtype,
                    "load_in_8bit": load_in_8bit,
                    "load_in_4bit": load_in_4bit,
                }
            )
            backbone = HuggingFaceBackbone(backbone_config)
            return backbone, backbone.tokenizer

        loop = asyncio.get_event_loop()
        backbone, tokenizer = await loop.run_in_executor(None, _load_hf_inner)

        # Estimate device
        device = str(backbone.model.device)
        size_mb = self.hub.storage.estimate_model_size_mb(backbone.model)

        self._loaded[model_id] = LoadedModel(
            model_id=model_id,
            model=backbone,
            tokenizer=tokenizer,
            config=config,
            backend="transformers",
            device=device,
            model_size_mb=size_mb,
            loaded_at=asyncio.get_event_loop().time(),
        )

    async def _load_multimodal(
        self,
        model_id: str,
        config: Dict[str, Any],
    ) -> None:
        """Load MultiModal model."""
        from .backbones.multimodal import MultiModalBackbone
        from .backbones.base import BackboneConfig
        import torch

        vision_config = config["vision_config"]
        llm_config = config["llm_config"]

        logger.info("Loading MultiModal model %s...", model_id)

        def _load_mm_inner():
            backbone_config = BackboneConfig(
                extra={
                    "vision_config": vision_config,
                    "llm_config": llm_config,
                }
            )
            backbone = MultiModalBackbone(backbone_config)
            return backbone, backbone.llm_backbone.tokenizer

        loop = asyncio.get_event_loop()
        backbone, tokenizer = await loop.run_in_executor(None, _load_mm_inner)

        # Estimate device
        device = str(backbone.llm_backbone.model.device)
        size_mb = self._estimate_multimodal_size(backbone)

        self._loaded[model_id] = LoadedModel(
            model_id=model_id,
            model=backbone,
            tokenizer=tokenizer,
            config=config,
            backend="multimodal",
            device=device,
            model_size_mb=size_mb,
            loaded_at=asyncio.get_event_loop().time(),
        )

    def _estimate_multimodal_size(self, model: Any) -> float:
        """Estimate multimodal model size in MB."""
        import torch

        size_mb = self._estimate_model_size(model.vision_backbone.model)
        size_mb += self._estimate_model_size(model.llm_backbone.model)
        # Add projection layer size
        proj_params = sum(p.numel() for p in model.projection.parameters())
        # Use element size of weights
        element_size = model.projection.weight.element_size()
        size_mb += (proj_params * element_size) / (1024 * 1024)
        return round(float(size_mb), 2)

    def get_model(self, model_id: str) -> Optional[LoadedModel]:
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

    async def migrate_model(self, model_id: str, target_device: str) -> Dict[str, Any]:
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

    def _estimate_model_size(self, model: Any) -> float:
        """Estimate model size in MB from parameter count."""
        return self.hub.storage.estimate_model_size_mb(model)

    async def download_model(
        self,
        model_id: str,
        progress_callback: Callable[[dict[str, Any]], Any] | None = None,
    ) -> dict[str, Any]:
        """Download a model to local storage."""
        return await self.hub.download(model_id, progress_callback)
