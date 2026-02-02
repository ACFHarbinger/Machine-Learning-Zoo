"""Dataset wrapper for experience replay."""

import torch
from torch.utils.data import Dataset
from typing import Any, List, Optional
from .continual import ReplayBuffer


class ReplayDataset(Dataset):
    """
    Dataset wrapper that combines a base dataset with samples from a replay buffer.
    """

    def __init__(
        self,
        base_dataset: Dataset,
        replay_buffer: ReplayBuffer,
        replay_ratio: float = 0.2,
    ):
        """
        Initialize ReplayDataset.
        Args:
            base_dataset: The new dataset for the current task.
            replay_buffer: The buffer containing samples from previous tasks.
            replay_ratio: Proportion of replayed samples in each epoch.
        """
        self.base_dataset = base_dataset
        self.replay_buffer = replay_buffer
        self.replay_ratio = replay_ratio

        # Pre-sample from buffer for this epoch if needed,
        # or sample dynamically. Dynamically is better for freshness.
        self.replay_samples = self.replay_buffer.sample(
            int(len(self.base_dataset) * self.replay_ratio)
        )

    def __len__(self) -> int:
        return len(self.base_dataset) + len(self.replay_samples)

    def __getitem__(self, idx: int) -> Any:
        if idx < len(self.base_dataset):
            return self.base_dataset[idx]
        else:
            return self.replay_samples[idx - len(self.base_dataset)]
