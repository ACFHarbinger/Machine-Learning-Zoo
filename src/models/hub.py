"""Unified Model Hub for managing models and checkpoints."""

import asyncio
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any, Dict, List, Optional

from .storage import StorageManager
from ..constants.models import MODEL_CONFIGS

logger = logging.getLogger(__name__)


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
                if adapter_path.is_dir() and (adapter_path / "adapter_config.json").exists():
                    local_models.append(
                        {
                            "id": adapter_path.name,
                            "path": str(adapter_path),
                            "type": "adapter",
                            "base_model": self._get_base_model_from_adapter(adapter_path),
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
            logger.info("Transformers model %s will be handled by the backbone loader", model_id)
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
            await progress_callback({"status": "complete", "model_id": model_id, "path": local_path})

        return {"status": "downloaded", "path": local_path}

    def get_storage_stats(self) -> Dict[str, Any]:
        """
        Get storage utilization statistics for the Model Hub.
        """
        usage = self.storage.get_disk_usage()
        hub_size = self.storage.get_models_size()

        return {
            "hub_size_bytes": hub_size,
            "hub_size_gb": round(hub_size / (1024**3), 2),
            "disk_total_gb": round(usage["total"] / (1024**3), 2),
            "disk_used_gb": round(usage["used"] / (1024**3), 2),
            "disk_free_gb": round(usage["free"] / (1024**3), 2),
            "percent_used": round((usage["used"] / usage["total"]) * 100, 1),
        }

    def prune_models(self, keep_n: int = 5) -> List[str]:
        """
        Remove least recently used model files to free up space.
        Args:
            keep_n: Number of models to keep.
        Returns:
            List of deleted model IDs.
        """
        local_models = self.list_local()
        # Sort by access time if possible, otherwise by modification time
        # For simplicity, we'll sort based on file modification time

        def _get_mtime(m):
            p = Path(m.get("path", ""))
            return p.stat().st_mtime if p.exists() else 0

        models_with_time = [m for m in local_models if m.get("path")]
        models_with_time.sort(key=_get_mtime, reverse=True)

        to_delete = models_with_time[keep_n:]
        deleted = []

        for m in to_delete:
            model_id = m.get("model_id") or m.get("id")
            if model_id and self.delete(model_id):
                deleted.append(model_id)

        return deleted

    def get_model_info(self, model_id: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed metadata and status for a specific model.
        """
        config = MODEL_CONFIGS.get(model_id)
        if not config:
            return None

        # Check local status
        local_info = next((m for m in self.list_local() if m.get("model_id") == model_id), None)

        info = {
            "model_id": model_id,
            "is_local": local_info is not None,
            "metadata": {
                "author": config.get("author", "Unknown"),
                "license": config.get("license", "Unknown"),
                "loader": config.get("loader"),
            },
            "configuration": config,
        }

        if local_info:
            info["local_details"] = local_info

        return info
