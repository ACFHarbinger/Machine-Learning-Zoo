"""
Baseline implementations for policy gradient methods.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional, Tuple

import torch
import torch.nn as nn
from logic.src.utils.logging.pylogger import get_pylogger
from tensordict import TensorDict

from ...utils.data.rl_utils import safe_td_copy

logger = get_pylogger(__name__)


class Baseline(nn.Module, ABC):
    """Base class for baselines."""

    def __init__(self):
        """Initialize the baseline."""
        super().__init__()

    @abstractmethod
    def eval(
        self, td: TensorDict, reward: torch.Tensor, env: Optional[Any] = None
    ) -> torch.Tensor:  # type: ignore[override]
        """Compute baseline value."""
        raise NotImplementedError

    def unwrap_batch(self, batch: Any) -> Tuple[Any, Optional[torch.Tensor]]:
        """Unwrap the batch if it's wrapped with baseline values.

        Args:
            batch: Wrapped batch.

        Returns:
            Tuple: Unwrapped batch data and optional baseline.
        """
        if isinstance(batch, (dict, TensorDict)):
            if "data" in list(batch.keys()) and "baseline" in list(batch.keys()):
                return batch["data"], batch["baseline"]
        return batch, None

    def unwrap_dataset(self, dataset: Any) -> Any:
        """Unwrap the dataset if it's wrapped."""
        from logic.src.data.datasets import BaselineDataset

        if isinstance(dataset, BaselineDataset):
            return dataset.dataset
        return dataset

    def epoch_callback(
        self,
        policy: nn.Module,
        epoch: int,
        val_dataset: Optional[Any] = None,
        env: Optional[Any] = None,
    ):
        """Optional callback at epoch end."""
        pass

    def wrap_dataset(
        self,
        dataset: Any,
        policy: Optional[nn.Module] = None,
        env: Optional[Any] = None,
    ) -> Any:
        """Optional wrap dataset."""
        return dataset

    def setup(self, policy: nn.Module):
        """Optional setup with policy reference."""
        pass

    def get_learnable_parameters(self) -> list:
        """Get learnable parameters for the optimizer."""
        return []


class NoBaseline(Baseline):
    """No baseline (vanilla REINFORCE)."""

    def __init__(self, **kwargs):
        """Initialize NoBaseline."""
        super().__init__()

    def eval(
        self, td: TensorDict, reward: torch.Tensor, env: Optional[Any] = None
    ) -> torch.Tensor:  # type: ignore[override]
        """
        Return zero baseline (no variance reduction).

        Args:
            td: TensorDict with environment state (unused).
            reward: Current batch rewards.
            env: Environment (unused).

        Returns:
            torch.Tensor: Zeros matching the reward shape.
        """
        return torch.zeros_like(reward)


class ExponentialBaseline(Baseline):
    """
    Moving average (exponential) baseline.
    """

    def __init__(self, beta: float = 0.8, exp_beta: Optional[float] = None, **kwargs):
        """
        Initialize ExponentialBaseline.

        Args:
            beta: Decay factor for exponential moving average.
            exp_beta: Alternative name for beta (matching RLConfig).
            **kwargs: Additional arguments.
        """
        super().__init__()
        self.beta = exp_beta if exp_beta is not None else beta
        self.running_mean: Optional[torch.Tensor] = None

    def eval(
        self, td: TensorDict, reward: torch.Tensor, env: Optional[Any] = None
    ) -> torch.Tensor:  # type: ignore[override]
        """
        Compute baseline value using exponential moving average.

        Args:
            td: TensorDict with environment state.
            reward: Current batch rewards.
            env: Environment (unused).

        Returns:
            torch.Tensor: Baseline value expanded to match reward shape.
        """
        if self.running_mean is None:
            self.running_mean = reward.mean().detach()
        else:
            self.running_mean = (
                self.beta * self.running_mean + (1 - self.beta) * reward.mean().detach()
            )
        return self.running_mean.expand_as(reward)


class RolloutBaseline(Baseline):
    """
    Greedy rollout baseline with significance-based updates.

    Uses greedy decoding with a frozen policy copy as baseline.
    The baseline policy is updated only if the current policy outperforms it
    significantly according to a T-test.
    """

    def __init__(
        self,
        policy: Optional[nn.Module] = None,
        update_every: int = 1,
        bl_alpha: float = 0.05,
        **kwargs,
    ):
        """
        Initialize RolloutBaseline.

        Args:
            policy: Policy to use as initial baseline (will be copied).
            update_every: Update baseline every N epochs.
            bl_alpha: Significance level for T-test to decide on updates.
            **kwargs: Additional keyword arguments.
        """
        super().__init__()

        # Handle old calling convention: RolloutBaseline(model, problem, opts)
        # where problem is a class and opts is a dict
        if isinstance(update_every, type) or (hasattr(update_every, "NAME")):
            # Old style: update_every is actually the problem class
            # bl_alpha is actually the opts dict
            opts = bl_alpha if isinstance(bl_alpha, dict) else kwargs.get("opts", {})
            self.update_every = (
                opts.get("bl_update_every", 1) if isinstance(opts, dict) else 1
            )
            self.bl_alpha = (
                opts.get("bl_alpha", 0.05) if isinstance(opts, dict) else 0.05
            )
        else:
            self.update_every = update_every
            self.bl_alpha = bl_alpha

        self.baseline_policy = None
        if policy is not None:
            self.setup(policy)

    def setup(self, policy: nn.Module):
        """Copy policy for baseline."""
        import copy

        self.baseline_policy = copy.deepcopy(policy)  # type: ignore[assignment]
        if self.baseline_policy is not None:
            self.baseline_policy.eval()  # type: ignore[misc]
            for param in self.baseline_policy.parameters():
                param.requires_grad = False

    def _rollout(
        self, policy: nn.Module, td_or_dataset: Any, env: Optional[Any] = None
    ) -> torch.Tensor:
        """Run greedy rollout on a batch or dataset."""
        from torch.utils.data import Dataset

        if isinstance(td_or_dataset, Dataset):
            return self._rollout_dataset(policy, td_or_dataset, env)
        return self._rollout_batch(policy, td_or_dataset, env)

    def _rollout_dataset(
        self, policy: nn.Module, dataset: Any, env: Optional[Any] = None
    ) -> torch.Tensor:
        """Run greedy rollout on a complete dataset."""
        if env is None:
            raise ValueError(
                "Environment (env) is required for RolloutBaseline evaluation"
            )

        from logic.src.data.datasets import tensordict_collate_fn
        from logic.src.utils.functions.rl import ensure_tensordict
        from torch.utils.data import DataLoader

        # Determine strict batch size from environment
        batch_size = 64  # Default
        if hasattr(env, "batch_size") and len(env.batch_size) > 0:
            batch_size = int(env.batch_size[0])

        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            collate_fn=tensordict_collate_fn,
            num_workers=0,
        )
        rewards = []
        policy.eval()
        with torch.no_grad():
            for batch in loader:
                # Get device from policy
                device = next(policy.parameters()).device
                td_data = ensure_tensordict(batch, device)

                try:
                    if hasattr(env, "reset"):
                        td_data = env.reset(td_data)
                except (OSError, ValueError, KeyError) as e:
                    logger.warning(f"Environment reset failed: {e}")
                    raise e

                # Padding logic for last batch
                real_size = td_data.batch_size[0]
                if real_size != batch_size:
                    pad_size = batch_size - real_size
                    # Repeat the first element to pad safely
                    padding = td_data[0].expand(pad_size)
                    padding_safe = safe_td_copy(padding)
                    td_data = torch.cat([td_data, padding_safe], 0)

                if hasattr(policy, "set_decode_type"):
                    policy.set_decode_type("greedy")
                    res = policy(td_data)
                    out = {"reward": res[0]} if isinstance(res, tuple) else res
                else:
                    out = policy(td_data, env, decode_type="greedy")

                # Unpad rewards
                if real_size != batch_size:
                    out["reward"] = out["reward"][:real_size]

                rewards.append(out["reward"].cpu())

        if len(rewards) == 0:
            return torch.tensor([], device="cpu")
        return torch.cat(rewards)

    def _rollout_batch(
        self, policy: nn.Module, td: Any, env: Optional[Any] = None
    ) -> torch.Tensor:
        """Run greedy rollout on a single batch."""
        import copy

        from logic.src.utils.functions.rl import ensure_tensordict

        # Note: deepcopy can be expensive, but ensure_tensordict helps standardize
        device = next(policy.parameters()).device
        td_data = ensure_tensordict(td, device)
        td_copy = copy.deepcopy(td_data)

        policy.eval()
        with torch.no_grad():
            if hasattr(policy, "set_decode_type"):
                policy.set_decode_type("greedy")
                res = policy(td_copy)
                out = {"reward": res[0]} if isinstance(res, tuple) else res
            else:
                if env is None:
                    raise ValueError(
                        "Environment (env) is required for RolloutBaseline evaluation"
                    )
                out = policy(td_copy, env, decode_type="greedy")

        return out["reward"]

    def wrap_dataset(
        self,
        dataset: Any,
        policy: Optional[nn.Module] = None,
        env: Optional[Any] = None,
    ) -> Any:
        """Wrap the dataset with rollout baseline values."""
        from logic.src.data.datasets import BaselineDataset

        # Compatibility: handle positional arguments if called as (policy, dataset, env)
        if isinstance(dataset, nn.Module):
            # Probably called as wrap_dataset(policy, dataset, env)
            policy_arg = dataset
            dataset_arg = policy  # Actually the 2nd arg
            env_arg = env
            dataset = dataset_arg
            policy = policy_arg
            env = env_arg

        if env is None:
            # We can't actually rollout without env, so return original
            return dataset

        # Use provided policy or fallback to baseline_policy
        p = policy if policy is not None else self.baseline_policy
        if p is None:
            return dataset

        print("Evaluating baseline on dataset...")
        bl_vals = self._rollout(p, dataset, env)
        return BaselineDataset(dataset, bl_vals.view(-1, 1))

    def eval(
        self, td: TensorDict, reward: torch.Tensor, env: Optional[Any] = None
    ) -> torch.Tensor:  # type: ignore[override]
        """
        Compute baseline value.
        """
        # If we have a baseline policy, run it to get the baseline value
        if self.baseline_policy is not None:
            # Note: This is computationally expensive if done every step
            # Ideally use wrap_dataset/unwrap_batch flow
            with torch.no_grad():
                from logic.src.utils.functions.rl import ensure_tensordict

                td = ensure_tensordict(
                    td, next(iter(self.baseline_policy.parameters())).device
                )
                return self._rollout(self.baseline_policy, td, env)

        return torch.zeros_like(reward)

    def epoch_callback(
        self,
        policy: nn.Module,
        epoch: int,
        val_dataset: Optional[Any] = None,
        env: Optional[Any] = None,
    ):
        """Update baseline policy if current policy improves significantly."""
        if (epoch + 1) % self.update_every == 0:
            if (
                val_dataset is not None
                and len(val_dataset) > 0
                and self.baseline_policy is not None
                and env is not None
            ):
                from scipy import stats

                # Evaluate candidate
                candidate_vals = self._rollout(policy, val_dataset, env)
                candidate_mean = candidate_vals.mean().item()

                # Evaluate baseline
                baseline_vals = self._rollout(self.baseline_policy, val_dataset, env)
                baseline_mean = baseline_vals.mean().item()

                # T-test for significance
                t_stat, p_val = stats.ttest_rel(
                    candidate_vals.cpu().numpy(), baseline_vals.cpu().numpy()
                )

                if candidate_mean > baseline_mean and p_val / 2 < self.bl_alpha:
                    print(
                        f"Update baseline: {baseline_mean:.4f} -> {candidate_mean:.4f} (p={p_val / 2:.4f})"
                    )
                    self.setup(policy)
            else:
                self.setup(policy)


class WarmupBaseline(Baseline):
    """Gradual transition from ExponentialBaseline to target baseline."""

    def __init__(
        self,
        baseline: Baseline,
        warmup_epochs: int = 1,
        bl_warmup_epochs: Optional[int] = None,
        beta: float = 0.8,
        exp_beta: Optional[float] = None,
        **kwargs,
    ):
        """
        Initialize WarmupBaseline.

        Args:
            baseline: Target baseline to transition to.
            warmup_epochs: Number of epochs for warmup transition.
            bl_warmup_epochs: Alternative name for warmup_epochs.
            beta: Beta parameter for the warmup exponential baseline.
            exp_beta: Alternative name for beta.
        """
        super().__init__()
        self.baseline = baseline
        self.warmup_epochs = (
            bl_warmup_epochs if bl_warmup_epochs is not None else warmup_epochs
        )
        self.warmup_baseline = ExponentialBaseline(beta=beta, exp_beta=exp_beta)
        self.alpha = 0.0

    def eval(
        self, td: TensorDict, reward: torch.Tensor, env: Optional[Any] = None
    ) -> torch.Tensor:  # type: ignore[override]
        """
        Compute blended baseline value based on warmup progress.

        Args:
            td: TensorDict with environment state.
            reward: Current batch rewards.
            env: Environment.

        Returns:
            torch.Tensor: Blended baseline value.
        """
        if self.alpha >= 1.0:
            return self.baseline.eval(td, reward, env)
        if self.alpha <= 0.0:
            return self.warmup_baseline.eval(td, reward, env)

        v_target = self.baseline.eval(td, reward, env)
        v_warmup = self.warmup_baseline.eval(td, reward, env)
        return self.alpha * v_target + (1 - self.alpha) * v_warmup

    def unwrap_batch(self, batch: Any) -> Tuple[Any, Optional[torch.Tensor]]:
        """Unwrap the batch using the inner baseline.

        Args:
            batch: Wrapped batch.

        Returns:
            Tuple: Unwrapped batch data and optional baseline.
        """
        return self.baseline.unwrap_batch(batch)

    def epoch_callback(
        self,
        policy: nn.Module,
        epoch: int,
        val_dataset: Optional[Any] = None,
        env: Optional[Any] = None,
    ):
        """
        Update warmup alpha and call inner baseline callback.

        Args:
            policy: Current policy.
            epoch: Current epoch number.
            val_dataset: Validation dataset.
            env: Environment.
        """
        self.baseline.epoch_callback(policy, epoch, val_dataset, env)
        if epoch < self.warmup_epochs:
            self.alpha = (epoch + 1) / float(self.warmup_epochs)
        else:
            self.alpha = 1.0

    def get_learnable_parameters(self) -> list:
        """Get learnable parameters of the inner baseline."""
        return self.baseline.get_learnable_parameters()


class CriticBaseline(Baseline):
    """Learned critic baseline."""

    def __init__(self, critic: Optional[nn.Module] = None, **kwargs):
        """
        Initialize CriticBaseline.

        Args:
            critic: Critic neural network module.
        """
        super().__init__()
        self.critic = critic

    def eval(
        self, td: TensorDict, reward: torch.Tensor, env: Optional[Any] = None
    ) -> torch.Tensor:  # type: ignore[override]
        """
        Compute baseline value using learned critic.

        Args:
            td: TensorDict with environment state.
            reward: Current batch rewards (used for shape if critic is None).
            env: Environment (unused).

        Returns:
            torch.Tensor: Critic value predictions.
        """
        if self.critic is None:
            return torch.zeros_like(reward)

        from logic.src.utils.functions.rl import ensure_tensordict

        td = ensure_tensordict(td, next(iter(self.critic.parameters())).device)
        return self.critic(td).squeeze(-1)

    def get_learnable_parameters(self) -> list:
        """Get learnable parameters for the critic network."""
        return list(self.critic.parameters()) if self.critic is not None else []

    def state_dict(self, *args, **kwargs):
        """Compatibility state_dict."""
        sd = super().state_dict(*args, **kwargs)
        if self.critic is not None:
            sd["critic"] = self.critic.state_dict()
        return sd


class POMOBaseline(Baseline):
    """
    POMO baseline: mean reward across starts of the SAME instance.
    """

    def __init__(self, **kwargs):
        """Initialize POMOBaseline."""
        super().__init__()

    def eval(
        self, td: TensorDict, reward: torch.Tensor, env: Optional[Any] = None
    ) -> torch.Tensor:  # type: ignore[override]
        """
        Compute POMO baseline as mean reward across starting points.

        Args:
            td: TensorDict with environment state.
            reward: Reward tensor with shape [batch, num_starts].
            env: Environment (unused).

        Returns:
            torch.Tensor: Mean reward expanded to match input shape.
        """
        # Reward shape: [batch, num_starts]
        if reward.dim() > 1:
            return reward.mean(dim=1, keepdim=True).expand_as(reward)
        return reward.mean()


# Baseline registry
BASELINE_REGISTRY = {
    "none": NoBaseline,
    "exponential": ExponentialBaseline,
    "rollout": RolloutBaseline,
    "critic": CriticBaseline,
    "warmup": WarmupBaseline,
    "pomo": POMOBaseline,
}


def get_baseline(name: str, policy: Optional[nn.Module] = None, **kwargs) -> Baseline:
    """Get baseline by name."""
    if name not in BASELINE_REGISTRY:
        raise ValueError(f"Unknown baseline: {name}")

    baseline = BASELINE_REGISTRY[name](**kwargs)
    if policy is not None:
        baseline.setup(policy)
    return baseline
