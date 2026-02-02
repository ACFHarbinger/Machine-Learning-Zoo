from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from ..enums import RunStatus


@dataclass
class RunInfo:
    """Information about a training run."""

    run_id: str
    status: RunStatus
    config: dict[str, Any]
    started_at: datetime | None = None
    completed_at: datetime | None = None
    metrics: dict[str, float] = field(default_factory=dict)
    error: str | None = None
    model_path: str | None = None
    tool_name: str | None = None  # Name when deployed as a tool
    deployed: bool = False  # Whether this run is deployed for inference
    deploy_device: str | None = None  # Device the deployed model is on
    task_type: str | None = None  # "classification", "regression", "generation", etc.
