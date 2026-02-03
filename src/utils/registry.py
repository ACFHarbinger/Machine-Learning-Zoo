from __future__ import annotations

import importlib.metadata
import logging
from collections.abc import Callable
from typing import Generic, Optional, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


class Registry(Generic[T]):
    """
    A central registry for managing components.
    """

    def __init__(self, name: str, entry_point_group: Optional[str] = None) -> None:
        self._name = name
        self._registry: dict[str, type[T]] = {}
        self._entry_point_group = entry_point_group
        self._plugins_loaded = False

    def register(self, name: str | None = None) -> Callable[[type[T]], type[T]]:
        """
        Decorator to register a class.
        """

        def wrapper(cls: type[T]) -> type[T]:
            reg_name = name or cls.__name__
            if reg_name in self._registry:
                # We could log a warning here if overwriting is intended
                pass
            self._registry[reg_name] = cls
            return cls

        return wrapper

    def get(self, name: str) -> type[T]:
        """
        Retrieve a class from the registry.
        """
        if not self._plugins_loaded and self._entry_point_group:
            self._load_plugins()

        if name not in self._registry:
            available = ", ".join(sorted(self._registry.keys()))
            raise ValueError(
                f"Unknown component '{name}' in registry '{self._name}'. "
                f"Available components: {available}"
            )
        return self._registry[name]

    def list_available(self) -> list[str]:
        """
        List all registered component names.
        """
        if not self._plugins_loaded and self._entry_point_group:
            self._load_plugins()
        return sorted(list(self._registry.keys()))

    def _load_plugins(self):
        """Load plugins from entry points."""
        if self._entry_point_group:
            eps = importlib.metadata.entry_points().select(
                group=self._entry_point_group
            )
            for ep in eps:
                try:
                    cls = ep.load()
                    reg_name = ep.name
                    if reg_name not in self._registry:
                        self._registry[reg_name] = cls
                        logger.info(
                            f"Loaded plugin '{reg_name}' for registry '{self._name}'"
                        )
                except Exception as e:
                    logger.error(f"Failed to load plugin '{ep.name}': {e}")
        self._plugins_loaded = True

    @property
    def registry(self) -> dict[str, type[T]]:
        """
        Get the internal registry dictionary.
        """
        return self._registry


# Global Registries
MODEL_REGISTRY = Registry("Model", entry_point_group="ml_zoo.models")
POLICY_REGISTRY = Registry("Policy", entry_point_group="ml_zoo.policies")
ENV_REGISTRY = Registry("Environment", entry_point_group="ml_zoo.envs")
PIPELINE_REGISTRY = Registry("Pipeline", entry_point_group="ml_zoo.pipelines")

# Helper Decorators
register_model = MODEL_REGISTRY.register
register_policy = POLICY_REGISTRY.register
register_env = ENV_REGISTRY.register
register_pipeline = PIPELINE_REGISTRY.register
