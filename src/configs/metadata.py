"""
Metadata Configurations.

Defines the structure for model metadata.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ModelMetadata:
    """Metadata for stored models."""

    name: str
    version: str
    checksum: str
    size_bytes: int
    created_at: str
    framework: str = "pytorch"
    architecture: str = ""
    metrics: dict[str, Any] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)
