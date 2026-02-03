"""
Multi-Agent Reinforcement Learning Training.

Provides MAPPO (Multi-Agent Proximal Policy Optimization) and
independent learner training for cooperative and competitive
multi-agent environments.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Categorical, Normal

logger = logging.getLogger(__name__)

__all__ = [
    "MARLConfig",
    "MultiAgentRolloutBuffer",
    "MAPPOTrainer",
]


@dataclass
class MARLConfig:
    """Configuration for multi-agent RL training."""

    num_agents: int = 4
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    num_epochs: int = 4
    num_minibatches: int = 4
    rollout_length: int = 128
    shared_parameters: bool = True
    continuous: bool = False
    extra: dict[str, Any] = field(default_factory=dict)


class MultiAgentRolloutBuffer:
    """
    Stores rollout data for multiple agents.

    Each buffer entry contains per-agent observations, actions,
    rewards, values, and log-probabilities collected during environment
    interaction.
    """

    def __init__(self, num_agents: int, rollout_length: int) -> None:
        self.num_agents = num_agents
        self.rollout_length = rollout_length
        self.reset()

    def reset(self) -> None:
        """Clear all stored data."""
        self.observations: list[dict[str, np.ndarray]] = []
        self.actions: list[dict[str, Any]] = []
        self.rewards: list[dict[str, float]] = []
        self.values: list[torch.Tensor] = []
        self.log_probs: list[torch.Tensor] = []
        self.dones: list[bool] = []
        self._ptr = 0

    def add(
        self,
        observations: dict[str, np.ndarray],
        actions: dict[str, Any],
        rewards: dict[str, float],
        values: torch.Tensor,
        log_probs: torch.Tensor,
        done: bool,
    ) -> None:
        """Add a single timestep of multi-agent experience."""
        self.observations.append(observations)
        self.actions.append(actions)
        self.rewards.append(rewards)
        self.values.append(values.detach())
        self.log_probs.append(log_probs.detach())
        self.dones.append(done)
        self._ptr += 1

    @property
    def size(self) -> int:
        return self._ptr

    @property
    def is_full(self) -> bool:
        return self._ptr >= self.rollout_length

    def compute_returns_and_advantages(
        self,
        last_values: torch.Tensor,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute GAE advantages and discounted returns for all agents.

        Args:
            last_values: Value estimates at the final step, shape (num_agents,).
            gamma: Discount factor.
            gae_lambda: GAE lambda parameter.

        Returns:
            Tuple of (returns, advantages) each of shape
            (rollout_length, num_agents).
        """
        T = self.size
        advantages = torch.zeros(T, self.num_agents)
        returns = torch.zeros(T, self.num_agents)

        # Values tensor: (T, num_agents)
        values = torch.stack(self.values, dim=0)  # (T, num_agents)

        # Rewards tensor: (T, num_agents)
        agent_ids = [f"agent_{i}" for i in range(self.num_agents)]
        rewards = torch.tensor([
            [step_rewards[aid] for aid in agent_ids]
            for step_rewards in self.rewards
        ])

        # Dones tensor
        dones = torch.tensor(self.dones, dtype=torch.float32)

        last_gae = torch.zeros(self.num_agents)

        for t in reversed(range(T)):
            if t == T - 1:
                next_values = last_values
                next_non_terminal = 1.0 - float(self.dones[-1])
            else:
                next_values = values[t + 1]
                next_non_terminal = 1.0 - dones[t + 1].item()

            delta = rewards[t] + gamma * next_values * next_non_terminal - values[t]
            last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
            advantages[t] = last_gae
            returns[t] = advantages[t] + values[t]

        return returns, advantages

    def get_minibatches(
        self,
        returns: torch.Tensor,
        advantages: torch.Tensor,
        num_minibatches: int,
    ) -> list[dict[str, torch.Tensor]]:
        """
        Split rollout data into shuffled minibatches.

        Args:
            returns: Computed returns of shape (T, num_agents).
            advantages: Computed advantages of shape (T, num_agents).
            num_minibatches: Number of minibatches to split into.

        Returns:
            List of minibatch dicts with keys: observations, actions,
            old_log_probs, returns, advantages, old_values.
        """
        T = self.size
        indices = np.random.permutation(T)
        batch_size = max(T // num_minibatches, 1)

        old_log_probs = torch.stack(self.log_probs, dim=0)  # (T, num_agents)
        old_values = torch.stack(self.values, dim=0)  # (T, num_agents)

        minibatches = []
        for start in range(0, T, batch_size):
            end = min(start + batch_size, T)
            idx = indices[start:end]

            minibatches.append({
                "indices": torch.tensor(idx, dtype=torch.long),
                "old_log_probs": old_log_probs[idx],
                "returns": returns[idx],
                "advantages": advantages[idx],
                "old_values": old_values[idx],
            })

        return minibatches


class MAPPOTrainer:
    """
    Multi-Agent Proximal Policy Optimization (MAPPO) trainer.

    Implements the centralized-training-decentralized-execution (CTDE)
    paradigm where agents share a centralized value function during
    training but act independently during execution.
    """

    def __init__(
        self,
        policy: nn.Module,
        config: MARLConfig,
        device: torch.device | None = None,
    ) -> None:
        self.policy = policy
        self.config = config
        self.device = device or torch.device("cpu")

        self.optimizer = torch.optim.Adam(
            policy.parameters(), lr=config.learning_rate
        )

        self.buffer = MultiAgentRolloutBuffer(
            num_agents=config.num_agents,
            rollout_length=config.rollout_length,
        )

        self._total_steps = 0
        self._total_episodes = 0

    def collect_rollout(
        self,
        env: Any,
        obs: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Collect a full rollout of experience from the environment.

        Args:
            env: Multi-agent environment with ``step()`` and ``reset()`` methods.
            obs: Current observations dict keyed by agent id.

        Returns:
            Dict with the final observation and episode statistics.
        """
        self.buffer.reset()
        episode_rewards: list[float] = []
        current_episode_reward = 0.0

        for _ in range(self.config.rollout_length):
            features = self._obs_to_features(obs)

            with torch.no_grad():
                actions, log_probs, values = self.policy.get_actions(features)

            # Convert tensor actions to env-compatible dict
            action_dict = self._actions_to_dict(actions)

            next_obs, rewards, terminated, truncated, info = env.step(action_dict)
            done = terminated or truncated

            # values and log_probs: (num_agents,) for single batch
            self.buffer.add(
                observations=obs,
                actions=action_dict,
                rewards=rewards,
                values=values.squeeze(-1) if values.dim() > 1 else values,
                log_probs=log_probs.squeeze(-1) if log_probs.dim() > 1 else log_probs,
                done=done,
            )

            current_episode_reward += sum(rewards.values())
            self._total_steps += 1

            if done:
                next_obs, _ = env.reset()
                self._total_episodes += 1
                episode_rewards.append(current_episode_reward)
                current_episode_reward = 0.0

            obs = next_obs

        return {
            "final_obs": obs,
            "episode_rewards": episode_rewards,
            "total_steps": self._total_steps,
            "total_episodes": self._total_episodes,
        }

    def update(self) -> dict[str, float]:
        """
        Perform MAPPO policy update using collected rollout data.

        Returns:
            Dict of training metrics (policy_loss, value_loss, entropy, etc.).
        """
        # Compute last values for GAE
        last_obs = self.buffer.observations[-1]
        last_features = self._obs_to_features(last_obs)
        with torch.no_grad():
            _, _, last_values = self.policy.get_actions(last_features)
            last_values = last_values.squeeze(-1) if last_values.dim() > 1 else last_values

        returns, advantages = self.buffer.compute_returns_and_advantages(
            last_values=last_values,
            gamma=self.config.gamma,
            gae_lambda=self.config.gae_lambda,
        )

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        num_updates = 0

        for _ in range(self.config.num_epochs):
            minibatches = self.buffer.get_minibatches(
                returns, advantages, self.config.num_minibatches
            )

            for mb in minibatches:
                idx = mb["indices"]
                mb_advantages = mb["advantages"].to(self.device)
                mb_returns = mb["returns"].to(self.device)
                mb_old_log_probs = mb["old_log_probs"].to(self.device)
                mb_old_values = mb["old_values"].to(self.device)

                # Re-evaluate actions at sampled indices
                policy_loss, value_loss, entropy = self._compute_losses(
                    idx, mb_old_log_probs, mb_returns, mb_advantages, mb_old_values
                )

                loss = (
                    policy_loss
                    + self.config.value_coef * value_loss
                    - self.config.entropy_coef * entropy
                )

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self.policy.parameters(), self.config.max_grad_norm
                )
                self.optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.item()
                num_updates += 1

        n = max(num_updates, 1)
        return {
            "policy_loss": total_policy_loss / n,
            "value_loss": total_value_loss / n,
            "entropy": total_entropy / n,
        }

    def _compute_losses(
        self,
        indices: torch.Tensor,
        old_log_probs: torch.Tensor,
        returns: torch.Tensor,
        advantages: torch.Tensor,
        old_values: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute PPO clipped policy loss, value loss, and entropy."""
        # Re-evaluate all timesteps in the minibatch
        new_log_probs_list = []
        new_values_list = []
        entropies_list = []

        for t_idx in indices:
            t = t_idx.item()
            obs = self.buffer.observations[t]
            features = self._obs_to_features(obs)

            output = self.policy(features)

            # Compute per-agent log probs for the actions that were taken
            actions_t = self.buffer.actions[t]
            agent_log_probs = []
            agent_entropies = []

            for i in range(self.config.num_agents):
                agent_id = f"agent_{i}"
                logits_i = output.action_logits[i]  # (1, action_dim) or (action_dim,)
                if logits_i.dim() == 2:
                    logits_i = logits_i.squeeze(0)

                if self.config.continuous:
                    std_i = output.action_std[i]
                    if std_i.dim() == 2:
                        std_i = std_i.squeeze(0)
                    dist = Normal(logits_i, std_i)
                    action_t = torch.tensor(
                        actions_t[agent_id], dtype=torch.float32, device=self.device
                    )
                    lp = dist.log_prob(action_t).sum()
                    ent = dist.entropy().sum()
                else:
                    dist = Categorical(logits=logits_i)
                    action_t = torch.tensor(
                        actions_t[agent_id], dtype=torch.long, device=self.device
                    )
                    lp = dist.log_prob(action_t)
                    ent = dist.entropy()

                agent_log_probs.append(lp)
                agent_entropies.append(ent)

            new_log_probs_list.append(torch.stack(agent_log_probs))
            new_values_list.append(
                output.values.squeeze(-1) if output.values.dim() > 1 else output.values
            )
            entropies_list.append(torch.stack(agent_entropies).mean())

        new_log_probs = torch.stack(new_log_probs_list)  # (mb_size, num_agents)
        new_values = torch.stack(new_values_list)  # (mb_size, num_agents)
        entropy = torch.stack(entropies_list).mean()

        # PPO clipped policy loss
        ratio = torch.exp(new_log_probs - old_log_probs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(
            ratio, 1.0 - self.config.clip_eps, 1.0 + self.config.clip_eps
        ) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()

        # Value loss with clipping
        value_pred_clipped = old_values + torch.clamp(
            new_values - old_values, -self.config.clip_eps, self.config.clip_eps
        )
        value_loss_unclipped = F.mse_loss(new_values, returns, reduction="none")
        value_loss_clipped = F.mse_loss(value_pred_clipped, returns, reduction="none")
        value_loss = torch.max(value_loss_unclipped, value_loss_clipped).mean()

        return policy_loss, value_loss, entropy

    def _obs_to_features(self, obs: dict[str, Any]) -> torch.Tensor:
        """
        Convert observation dict to feature tensor.

        Args:
            obs: Dict mapping agent_id to numpy observation.

        Returns:
            Tensor of shape (num_agents, 1, obs_dim).
        """
        agent_obs = []
        for i in range(self.config.num_agents):
            agent_id = f"agent_{i}"
            o = obs[agent_id]
            if not isinstance(o, torch.Tensor):
                o = torch.tensor(o, dtype=torch.float32, device=self.device)
            agent_obs.append(o.unsqueeze(0))  # (1, obs_dim)

        return torch.stack(agent_obs, dim=0)  # (num_agents, 1, obs_dim)

    def _actions_to_dict(self, actions: torch.Tensor) -> dict[str, Any]:
        """
        Convert action tensor to environment-compatible dict.

        Args:
            actions: Tensor of shape (num_agents, 1) or (num_agents,).

        Returns:
            Dict mapping agent_id to action value.
        """
        action_dict: dict[str, Any] = {}
        for i in range(self.config.num_agents):
            agent_id = f"agent_{i}"
            a = actions[i]
            if a.dim() > 0:
                a = a.squeeze()
            if self.config.continuous:
                action_dict[agent_id] = a.cpu().numpy()
            else:
                action_dict[agent_id] = int(a.item())
        return action_dict
