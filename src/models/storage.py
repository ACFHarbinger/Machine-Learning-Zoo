"""Storage utilities for the Model Hub."""

import shutil
from pathlib import Path
from typing import Any

import torch


class StorageManager:
    """Manages storage resources for the Model Hub."""

    def __init__(self, models_dir: Path):
        """
        Initialize the storage manager.
        Args:
            models_dir: Directory where models are stored.
        """
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)

    def get_disk_usage(self) -> dict[str, int]:
        """
        Get disk usage of the models directory.
        Returns:
            Dict with total, used, and free bytes.
        """
        usage = shutil.disk_usage(self.models_dir)
        return {
            "total": usage.total,
            "used": usage.used,
            "free": usage.free,
        }

    def get_models_size(self) -> int:
        """
        Get total size of all files in the models directory in bytes.
        Returns:
            Total size in bytes.
        """
        return sum(f.stat().st_size for f in self.models_dir.rglob("*") if f.is_file())

    def can_fit(self, size_bytes: int) -> bool:
        """
        Check if a file of given size can fit in the free disk space.
        Args:
            size_bytes: Size of the model in bytes.
        Returns:
            Boolean indicating if it fits.
        """
        usage = self.get_disk_usage()
        # Leave a 1GB buffer
        return (usage["free"] - size_bytes) > (1024 * 1024 * 1024)

    @staticmethod
    def estimate_model_size_mb(model: Any) -> float:
        """
        Estimate model size in MB from parameter count.
        Args:
            model: PyTorch model or any object with .parameters().
        Returns:
            Estimated size in MB.
        """
        try:
            if hasattr(model, "parameters"):
                param_bytes = sum(
                    p.nelement() * p.element_size() for p in model.parameters()
                )
                return round(float(param_bytes / (1024 * 1024)), 2)
        except Exception:
            pass
        return 0.0
