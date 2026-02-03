"""
REINFORCE algorithm implementation.

Reference:
    Williams, R. J. (1992).
    Simple statistical gradient-following algorithms for connectionist reinforcement learning.
    Machine learning, 8(3-4), 229-256.
"""

from typing import Any, Optional, cast

import torch

from ..common.base import RLBaseModule


class REINFORCE(RLBaseModule):
    """
    REINFORCE with baseline.

    Standard policy gradient algorithm with configurable baselines.
    """

    def __init__(
        self,
        entropy_weight: float = 0.0,
        max_grad_norm: float = 1.0,
        **kwargs: Any,
    ) -> None:
        """
        Initialize REINFORCE module.

        Args:
            entropy_weight: Weight for entropy bonus in loss.
            max_grad_norm: Maximum gradient norm for clipping.
            **kwargs: Arguments passed to RLBaseModule.
        """
        super().__init__(**kwargs)
        self.entropy_weight = entropy_weight
        self.max_grad_norm = max_grad_norm

    def calculate_loss(
        self,
        td: Any,
        out: dict,
        batch_idx: int,
        env: Optional[Any] = None,
    ) -> torch.Tensor:
        """
        Compute REINFORCE loss.

        Loss = -E[(R - b) * log π(a|s)]

        Args:
            td: TensorDict with environment state.
            out: Policy output dictionary with 'reward' and 'log_likelihood'.
            batch_idx: Current batch index.
            env: Environment (optional).

        Returns:
            Loss tensor.
        """
        reward = out["reward"]
        log_likelihood = out["log_likelihood"]

        # Get baseline value
        if hasattr(self, "_current_baseline_val") and self._current_baseline_val is not None:
            baseline_val = self._current_baseline_val
        else:
            baseline_val = self.baseline.eval(td, reward, env=env)

        # Compute advantage
        advantage = reward - baseline_val

        # Normalize advantage for stability
        if advantage.numel() > 1:
            advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

        # Policy gradient loss
        loss = -(advantage.detach() * log_likelihood).mean()

        # Entropy bonus (if applicable)
        if self.entropy_weight > 0 and "entropy" in out:
            loss = loss - self.entropy_weight * out["entropy"].mean()

        # Log components
        self.log("train/advantage", advantage.mean(), sync_dist=True)
        self.log("train/baseline", baseline_val.mean(), sync_dist=True)

        return cast(torch.Tensor, loss)

    def on_before_optimizer_step(self, optimizer: torch.optim.Optimizer) -> None:
        """Gradient clipping before optimizer step."""
        if self.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                self.policy.parameters(),
                self.max_grad_norm,
            )
