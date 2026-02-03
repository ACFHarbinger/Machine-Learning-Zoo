"""
Accelerated Trainer Configuration.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class AcceleratedTrainerConfig:
    """Configuration for accelerated training."""

    # Training params
    max_epochs: int = 10
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    batch_size: int = 32
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0

    # Accelerate params
    mixed_precision: str = "no"  # "no", "fp16", "bf16"
    gradient_checkpointing: bool = False

    # Logging/Checkpointing
    log_every_n_steps: int = 50
    save_every_n_epochs: int = 1
    output_dir: str = "~/.pi-assistant/models"
    run_name: str = "run"

    # Distributed
    deepspeed_config: str | None = None  # Path to DeepSpeed config

    # Extra
    seed: int = 42
    extra: dict[str, Any] = field(default_factory=dict)
