"""
Federated Learning Module.
Includes FedAvg implementation for decentralized training.
"""

import copy
import torch
import logging
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)


class FederatedAggregator:
    """
    Central server for Federated Learning.
    Handles weight aggregation using Federated Averaging (FedAvg).
    """

    def __init__(self, global_model: torch.nn.Module):
        self.global_model = global_model
        self.client_weights: List[Dict[str, torch.Tensor]] = []
        self.client_sizes: List[int] = []

    def register_client_update(
        self, state_dict: Dict[str, torch.Tensor], num_samples: int
    ):
        """Register a weight update from a client."""
        self.client_weights.append(state_dict)
        self.client_sizes.append(num_samples)

    def aggregate(self) -> Dict[str, torch.Tensor]:
        """
        Perform FedAvg.
        W_global = sum(n_i / n_total * W_i)
        """
        if not self.client_weights:
            logger.warning("No client updates to aggregate.")
            return self.global_model.state_dict()

        total_samples = sum(self.client_sizes)
        global_state = self.global_model.state_dict()

        # Initialize new global state with zeros
        new_global_state = {k: torch.zeros_like(v) for k, v in global_state.items()}

        for state, size in zip(self.client_weights, self.client_sizes):
            weight = size / total_samples
            for k in new_global_state.keys():
                new_global_state[k] += state[k] * weight

        # Update global model
        self.global_model.load_state_dict(new_global_state)

        # Clear buffer
        self.client_weights = []
        self.client_sizes = []

        return new_global_state


class FederatedClient:
    """
    Client for Federated Learning.
    Handles local training sub-tasks.
    """

    def __init__(self, model: torch.nn.Module, client_id: str):
        self.model = model
        self.client_id = client_id

    def pull_global_weights(self, global_state_dict: Dict[str, torch.Tensor]):
        """Synchronize with the global model."""
        self.model.load_state_dict(global_state_dict)

    def get_local_update(self) -> Dict[str, torch.Tensor]:
        """Return the local model state dict."""
        return copy.deepcopy(self.model.state_dict())
