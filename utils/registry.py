from __future__ import annotations

from collections.abc import Callable
from typing import Generic, TypeVar

T = TypeVar("T")


class Registry(Generic[T]):
    """
    A central registry for managing components.
    """

    def __init__(self, name: str) -> None:
        self._name = name
        self._registry: dict[str, type[T]] = {}

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
        return sorted(list(self._registry.keys()))

    @property
    def registry(self) -> dict[str, type[T]]:
        """
        Get the internal registry dictionary.
        """
        return self._registry


# Global Registries
MODEL_REGISTRY = Registry("Model")
POLICY_REGISTRY = Registry("Policy")
ENV_REGISTRY = Registry("Environment")
PIPELINE_REGISTRY = Registry("Pipeline")

# Helper Decorators
register_model = MODEL_REGISTRY.register
register_policy = POLICY_REGISTRY.register
register_env = ENV_REGISTRY.register
register_pipeline = PIPELINE_REGISTRY.register
