"""
Storage Configurations.

Contains configurations for cloud storage backends and model retention policies.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

__all__ = ["CloudStorageConfig", "RetentionConfig"]


@dataclass
class CloudStorageConfig:
    """Configuration for cloud storage backends.

    Attributes:
        bucket: S3 bucket or GCS bucket name.
        prefix: Prefix/folder for models within bucket.
        compression_level: Zstd compression level (1-22, default 3).
        enable_versioning: Enable object versioning.
        fallback_local_path: Local path to use if cloud fails.
    """

    bucket: str
    prefix: str = "models"
    compression_level: int = 3
    enable_versioning: bool = True
    fallback_local_path: str | None = None

    # AWS specific
    aws_region: str = "us-east-1"
    aws_access_key_id: str | None = None
    aws_secret_access_key: str | None = None

    # GCS specific
    gcs_project: str | None = None
    gcs_credentials_path: str | None = None


@dataclass
class RetentionConfig:
    """Configuration for model retention."""

    keep_latest_n: int = 5
    keep_best_metric: str | None = "val_loss"
    keep_best_n: int = 1
    max_age_days: int | None = None
    dry_run: bool = False


@dataclass
class StorageConfig:
    """Configuration for model storage backends."""

    # Storage type: local, s3, gcs
    storage_type: str = "local"

    # Local storage settings
    local_path: str = "./model_weights"

    # S3 settings
    s3_bucket: str = ""
    s3_prefix: str = "models/"
    s3_region: str = "us-east-1"
    aws_access_key_id: str | None = None
    aws_secret_access_key: str | None = None

    # GCS settings
    gcs_bucket: str = ""
    gcs_prefix: str = "models/"
    gcs_credentials_path: str | None = None

    # Common settings
    compression: str = "zstd"  # none, gzip, zstd
    versioning: bool = True
    max_versions: int = 5

    # Cache settings
    cache_enabled: bool = True
    cache_dir: str = ".cache/models"
    cache_max_size_gb: float = 10.0

    @classmethod
    def from_env(cls) -> "StorageConfig":
        """Create config from environment variables."""
        return cls(
            storage_type=os.getenv("NGLAB_MODEL_STORAGE", "local"),
            local_path=os.getenv("MODEL_WEIGHTS_DIR", "./model_weights"),
            s3_bucket=os.getenv("S3_BUCKET_NAME", ""),
            s3_prefix=os.getenv("S3_PREFIX", "models/"),
            s3_region=os.getenv("AWS_DEFAULT_REGION", "us-east-1"),
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
            gcs_bucket=os.getenv("GCS_BUCKET_NAME", ""),
            gcs_prefix=os.getenv("GCS_PREFIX", "models/"),
            gcs_credentials_path=os.getenv("GOOGLE_APPLICATION_CREDENTIALS"),
        )
