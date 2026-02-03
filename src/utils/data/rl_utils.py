"""
Reinforcement Learning utilities for WSmart-Route.
"""

from typing import Any

import torch
from tensordict import TensorDict


def safe_td_copy(td: Any) -> Any:
    """
    Perform a deep copy of a TensorDict that is safe against circular references
    and complex objects (like Shiboken/Qt proxy objects).
    Only clones Tensors and recurse into nested TensorDicts or dicts.
    """
    if not hasattr(td, "items") or not hasattr(td, "batch_size"):
        return td

    visited = set()

    def _internal_safe_copy(obj):
        obj_id = id(obj)
        if obj_id in visited:
            print(
                f"Warning: Circular reference detected in TensorDict for object {obj_id}. Breaking cycle."
            )
            return None
        visited.add(obj_id)

        try:
            if isinstance(obj, torch.Tensor):
                return obj.clone()

            if hasattr(obj, "items") and hasattr(obj, "batch_size"):  # TensorDict-like
                # Use a plain dict to accumulate clean items
                clean_dict = {}
                for k, v in obj.items():
                    # Only keep tensors or nested collections
                    if isinstance(v, (torch.Tensor, dict)) or (
                        hasattr(v, "items") and hasattr(v, "batch_size")
                    ):
                        copied_v = _internal_safe_copy(v)
                        if copied_v is not None:
                            clean_dict[k] = copied_v

                # Create a new TensorDict with standard batch/device
                return TensorDict(
                    clean_dict, batch_size=obj.batch_size, device=obj.device
                )

            if isinstance(obj, dict):
                res_dict = {}
                for k, v in obj.items():
                    copied_v = _internal_safe_copy(v)
                    if copied_v is not None:
                        res_dict[k] = copied_v
                return res_dict

            if isinstance(obj, (list, tuple)):
                res_list = []
                for x in obj:
                    copied_x = _internal_safe_copy(x)
                    if copied_x is not None:
                        res_list.append(copied_x)
                return type(obj)(res_list)

            # Primitives are safe
            if isinstance(obj, (int, float, str, bool)) or obj is None:
                return obj
        except Exception:
            return None

        return None

    res = _internal_safe_copy(td)
    # Ensure we return at least a zero tensor or original if result is None (to satisfy type hints)
    if res is None:
        if hasattr(td, "batch_size"):
            return TensorDict({}, batch_size=td.batch_size, device=td.device)
        return td
    return res
