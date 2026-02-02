"""A/B Testing Framework for Machine Learning Zoo."""

import hashlib
import logging
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class ABTestingManager:
    """
    Manages A/B tests by splitting traffic between different model versions.
    """

    def __init__(self):
        # Dictionary mapping test_id to experiment configuration
        self.experiments: Dict[str, Dict[str, Any]] = {}

    def create_experiment(
        self,
        test_id: str,
        variants: List[Dict[str, Any]],
        traffic_split: Optional[List[float]] = None,
    ) -> None:
        """
        Create a new A/B test experiment.
        Args:
            test_id: Unique identifier for the experiment.
            variants: List of variants, each being a dict with 'model_path' and 'engine'.
            traffic_split: List of floats summing to 1.0. Defaults to equal split.
        """
        if traffic_split is None:
            traffic_split = [1.0 / len(variants)] * len(variants)

        if len(variants) != len(traffic_split):
            raise ValueError("Number of variants and traffic_split must match.")

        if abs(sum(traffic_split) - 1.0) > 1e-6:
            raise ValueError("traffic_split must sum to 1.0")

        self.experiments[test_id] = {
            "variants": variants,
            "traffic_split": traffic_split,
            "cumulative_weights": self._get_cumulative_weights(traffic_split),
        }
        logger.info("Created A/B experiment: %s", test_id)

    def _get_cumulative_weights(self, weights: List[float]) -> List[float]:
        cumulative = []
        current = 0.0
        for w in weights:
            current += w
            cumulative.append(current)
        return cumulative

    def get_variant(
        self, test_id: str, session_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Assign a variant for a given request.
        Uses deterministic hashing if session_id is provided for sticky sessions.
        """
        if test_id not in self.experiments:
            raise ValueError(f"Experiment {test_id} not found.")

        exp = self.experiments[test_id]

        if session_id:
            # Sticky session using hash
            hash_val = int(
                hashlib.md5(f"{test_id}:{session_id}".encode()).hexdigest(), 16
            )
            point = (hash_val % 1000) / 1000.0
        else:
            # Random assignment
            import random

            point = random.random()

        for i, weight in enumerate(exp["cumulative_weights"]):
            if point < weight:
                return exp["variants"][i]

        return exp["variants"][-1]

    def list_experiments(self) -> Dict[str, Any]:
        return self.experiments
