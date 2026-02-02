from dataclasses import dataclass, field
from typing import Any


@dataclass
class LoadedModel:
    """A loaded model with its tokenizer or Llama instance."""

    model_id: str
    model: Any
    tokenizer: Any | None = None
    backend: str = "transformers"  # "transformers" or "llama.cpp"
    metadata: dict = field(default_factory=dict)
    device: str = "cpu"  # "cpu", "cuda:0", "cuda:1", "mps"
    model_size_mb: float = 0.0  # estimated memory footprint
    config: dict = field(default_factory=dict)
    loaded_at: float = 0.0
