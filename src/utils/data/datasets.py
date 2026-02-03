"""
Dataset implementations for Machine Learning Zoo.
"""

from typing import Any, Optional
import torch
from torch.utils.data import Dataset
from tensordict import TensorDict


class BaselineDataset(Dataset):
    """
    Dataset wrapper that adds baseline values to data.
    """

    def __init__(self, dataset: Dataset, baseline_values: torch.Tensor):
        self.dataset = dataset
        self.baseline_values = baseline_values

    def __getitem__(self, index: int) -> dict:
        data = self.dataset[index]
        return {"data": data, "baseline": self.baseline_values[index]}

    def __len__(self) -> int:
        return len(self.dataset)


def tensordict_collate_fn(batch: list) -> TensorDict:
    """
    Collate a list of objects into a TensorDict.
    """
    from .rl_utils import ensure_tensordict
    from torch.utils.data.dataloader import default_collate

    collated = default_collate(batch)
    return ensure_tensordict(collated)
