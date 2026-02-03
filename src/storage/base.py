"""
Base classes for model storage backends.

Defines the abstract interface that all storage backends must implement.
"""

import hashlib
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, cast

from ..configs.metadata import ModelMetadata
from ..configs.storage import StorageConfig


class ModelStorage(ABC):
    """Abstract base class for model storage backends."""

    def __init__(self, config: StorageConfig) -> None:
        """Initialize the storage backend."""
        self.config = config
        self._setup_cache()

    def _setup_cache(self) -> None:
        """Setup local cache directory."""
        if self.config.cache_enabled:
            cache_path = Path(self.config.cache_dir)
            cache_path.mkdir(parents=True, exist_ok=True)

    @abstractmethod
    def save(
        self,
        model_data: bytes,
        name: str,
        version: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """
        Save model data to storage.

        Args:
            model_data: Serialized model bytes
            name: Model name
            version: Optional version string (auto-generated if not provided)
            metadata: Optional metadata dictionary

        Returns:
            Full path/URI to saved model
        """
        pass

    @abstractmethod
    def load(
        self,
        name: str,
        version: str | None = None,
    ) -> bytes:
        """
        Load model data from storage.

        Args:
            name: Model name
            version: Optional version (latest if not provided)

        Returns:
            Serialized model bytes
        """
        pass

    @abstractmethod
    def exists(self, name: str, version: str | None = None) -> bool:
        """Check if a model exists in storage."""
        pass

    @abstractmethod
    def delete(self, name: str, version: str | None = None) -> bool:
        """Delete a model from storage."""
        pass

    @abstractmethod
    def list_models(self) -> list[str]:
        """List all model names in storage."""
        pass

    @abstractmethod
    def list_versions(self, name: str) -> list[str]:
        """List all versions of a model."""
        pass

    @abstractmethod
    def get_metadata(self, name: str, version: str | None = None) -> ModelMetadata:
        """Get metadata for a model."""
        pass

    def save_file(
        self,
        file_path: Path,
        name: str,
        version: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Save model from file path."""
        with open(file_path, "rb") as f:
            return self.save(f.read(), name, version, metadata)

    def load_to_file(
        self,
        name: str,
        output_path: Path,
        version: str | None = None,
    ) -> Path:
        """Load model to a file path."""
        data = self.load(name, version)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "wb") as f:
            f.write(data)
        return output_path

    @staticmethod
    def compute_checksum(data: bytes) -> str:
        """Compute SHA256 checksum of data."""
        return hashlib.sha256(data).hexdigest()

    def _get_cache_path(self, name: str, version: str) -> Path:
        """Get local cache path for a model."""
        return Path(self.config.cache_dir) / name / f"{version}.pt"

    def _save_to_cache(self, data: bytes, name: str, version: str) -> Path:
        """Save model data to local cache."""
        cache_path = self._get_cache_path(name, version)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "wb") as f:
            f.write(data)
        return cache_path

    def _load_from_cache(self, name: str, version: str) -> bytes | None:
        """Load model data from local cache if available."""
        cache_path = self._get_cache_path(name, version)
        if cache_path.exists():
            with open(cache_path, "rb") as f:
                return f.read()
        return None

    def _compress(self, data: bytes) -> bytes:
        """Compress data based on config."""
        if self.config.compression == "none":
            return data
        elif self.config.compression == "gzip":
            import gzip

            return gzip.compress(data, compresslevel=6)
        elif self.config.compression == "zstd":
            try:
                import zstandard as zstd

                cctx = zstd.ZstdCompressor(level=3)
                return cast(bytes, cctx.compress(data))
            except ImportError:
                # Fallback to gzip if zstd not available
                import gzip

                return gzip.compress(data, compresslevel=6)
        return data

    def _decompress(self, data: bytes) -> bytes:
        """Decompress data based on config."""
        if self.config.compression == "none":
            return data
        elif self.config.compression == "gzip":
            import gzip

            return gzip.decompress(data)
        elif self.config.compression == "zstd":
            try:
                import zstandard as zstd

                dctx = zstd.ZstdDecompressor()
                return cast(bytes, dctx.decompress(data))
            except ImportError:
                import gzip

                return gzip.decompress(data)

        return data
