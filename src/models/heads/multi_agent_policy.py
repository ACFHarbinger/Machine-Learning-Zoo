"""
Multi-Agent RL Policy Head.

Actor-Critic head for multi-agent reinforcement learning with
optional inter-agent communication via message passing.
"""

from __future__ import annotations

from typing import Any, NamedTuple

import torch
from torch import nn
from torch.distributions import Categorical, Normal

from .base import Head, register_head
from ...configs.heads import MultiAgentPolicyHeadConfig

__all__ = [
    "MultiAgentPolicyOutput",
    "MultiAgentPolicyHead",
    "MultiAgentPolicyHeadConfig",
]


class MultiAgentPolicyOutput(NamedTuple):
    """Output from multi-agent RL policy head."""

    action_logits: torch.Tensor  # (num_agents, batch, action_dim)
    action_std: torch.Tensor | None  # For continuous actions
    values: torch.Tensor  # (num_agents, batch)
    messages: torch.Tensor | None  # (num_agents, batch, message_dim)


@register_head("multi_agent_policy")
class MultiAgentPolicyHead(Head):
    """
    Multi-agent actor-critic head.

    Supports:
    - Shared or independent parameters across agents.
    - Discrete or continuous action spaces.
    - Optional communication channels between agents via learned messages.
    - Centralized value function that observes all agents' features.
    """

    def __init__(self, config: MultiAgentPolicyHeadConfig) -> None:
        super().__init__(config)
        self.cfg = config

        if config.shared_parameters:
            self._build_shared(config)
        else:
            self._build_independent(config)

        # Centralized critic: sees concatenated features from all agents
        critic_input_dim = config.input_dim * config.num_agents
        self.centralized_critic = nn.Sequential(
            nn.Linear(critic_input_dim, config.input_dim),
            nn.Tanh(),
            nn.Linear(config.input_dim, config.num_agents),
        )

        # Communication encoder/decoder
        if config.communication_dim > 0:
            self.message_encoder = nn.Sequential(
                nn.Linear(config.input_dim, config.communication_dim),
                nn.Tanh(),
            )
            self.message_decoder = nn.Sequential(
                nn.Linear(
                    config.communication_dim * (config.num_agents - 1),
                    config.input_dim,
                ),
                nn.ReLU(),
            )

    def _build_shared(self, config: MultiAgentPolicyHeadConfig) -> None:
        """Build a single policy network shared by all agents."""
        layers: list[nn.Module] = []
        current_dim = config.input_dim
        for hidden_dim in config.hidden_dims:
            layers.extend([nn.Linear(current_dim, hidden_dim), nn.Tanh()])
            current_dim = hidden_dim

        layers.append(nn.Linear(current_dim, config.action_dim))
        self.shared_actor = nn.Sequential(*layers)

        if config.continuous:
            self.log_std = nn.Parameter(torch.zeros(config.action_dim))

    def _build_independent(self, config: MultiAgentPolicyHeadConfig) -> None:
        """Build independent policy networks for each agent."""
        self.actors = nn.ModuleList()
        for _ in range(config.num_agents):
            layers: list[nn.Module] = []
            current_dim = config.input_dim
            for hidden_dim in config.hidden_dims:
                layers.extend([nn.Linear(current_dim, hidden_dim), nn.Tanh()])
                current_dim = hidden_dim
            layers.append(nn.Linear(current_dim, config.action_dim))
            self.actors.append(nn.Sequential(*layers))

        if config.continuous:
            self.log_stds = nn.ParameterList(
                [nn.Parameter(torch.zeros(config.action_dim)) for _ in range(config.num_agents)]
            )

    def forward(
        self,
        features: torch.Tensor,
        **kwargs: Any,
    ) -> MultiAgentPolicyOutput:
        """
        Forward pass for all agents.

        Args:
            features: Tensor of shape (num_agents, batch, input_dim) containing
                per-agent features, or (batch, input_dim) if features are
                shared (will be expanded for all agents).

        Returns:
            MultiAgentPolicyOutput with per-agent action distributions and values.
        """
        if features.dim() == 2:
            # Shared features: expand for all agents
            features = features.unsqueeze(0).expand(self.cfg.num_agents, -1, -1)

        # Pool sequences if needed
        if features.dim() == 4:
            features = features.mean(dim=2)

        num_agents, batch_size, _ = features.shape

        # Communication phase
        if self.cfg.communication_dim > 0:
            features = self._communicate(features)

        # Actor outputs
        all_logits = []
        for i in range(num_agents):
            agent_features = features[i]  # (batch, input_dim)
            if self.cfg.shared_parameters:
                logits = self.shared_actor(agent_features)
            else:
                logits = self.actors[i](agent_features)
            all_logits.append(logits)

        action_logits = torch.stack(all_logits, dim=0)  # (num_agents, batch, action_dim)

        # Action std for continuous
        action_std = None
        if self.cfg.continuous:
            if self.cfg.shared_parameters:
                log_std = self.log_std.clamp(self.cfg.log_std_min, self.cfg.log_std_max)
                action_std = log_std.exp().unsqueeze(0).unsqueeze(0).expand_as(action_logits)
            else:
                stds = []
                for i in range(num_agents):
                    log_std = self.log_stds[i].clamp(self.cfg.log_std_min, self.cfg.log_std_max)
                    stds.append(log_std.exp().unsqueeze(0).expand(batch_size, -1))
                action_std = torch.stack(stds, dim=0)

        # Centralized critic
        all_features = features.permute(1, 0, 2).reshape(batch_size, -1)  # (batch, num_agents * input_dim)
        values = self.centralized_critic(all_features)  # (batch, num_agents)
        values = values.t()  # (num_agents, batch)

        # Messages
        messages = None
        if self.cfg.communication_dim > 0:
            messages = torch.stack([self.message_encoder(features[i]) for i in range(num_agents)], dim=0)

        return MultiAgentPolicyOutput(
            action_logits=action_logits,
            action_std=action_std,
            values=values,
            messages=messages,
        )

    def _communicate(self, features: torch.Tensor) -> torch.Tensor:
        """
        Inter-agent communication via message passing.

        Each agent encodes a message, then each agent decodes
        messages from all other agents to augment its features.

        Args:
            features: (num_agents, batch, input_dim)

        Returns:
            Updated features of same shape.
        """
        num_agents = features.shape[0]

        # Encode messages
        messages = torch.stack(
            [self.message_encoder(features[i]) for i in range(num_agents)], dim=0
        )  # (num_agents, batch, comm_dim)

        # Each agent receives messages from all others
        updated = []
        for i in range(num_agents):
            received = [messages[j] for j in range(num_agents) if j != i]
            received_cat = torch.cat(received, dim=-1)  # (batch, (N-1)*comm_dim)
            decoded = self.message_decoder(received_cat)  # (batch, input_dim)
            updated.append(features[i] + decoded)

        return torch.stack(updated, dim=0)

    def get_actions(
        self,
        features: torch.Tensor,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample actions for all agents.

        Args:
            features: (num_agents, batch, input_dim) or (batch, input_dim).
            deterministic: Whether to use greedy actions.

        Returns:
            Tuple of (actions, log_probs, values), each of shape
            (num_agents, batch).
        """
        output = self.forward(features)
        num_agents = output.action_logits.shape[0]

        all_actions = []
        all_log_probs = []

        for i in range(num_agents):
            logits_i = output.action_logits[i]

            if self.cfg.continuous:
                assert output.action_std is not None
                dist = Normal(logits_i, output.action_std[i])
                if deterministic:
                    action = logits_i
                else:
                    action = dist.rsample()
                log_prob = dist.log_prob(action).sum(dim=-1)
            else:
                dist = Categorical(logits=logits_i)
                if deterministic:
                    action = logits_i.argmax(dim=-1)
                else:
                    action = dist.sample()
                log_prob = dist.log_prob(action)

            all_actions.append(action)
            all_log_probs.append(log_prob)

        actions = torch.stack(all_actions, dim=0)
        log_probs = torch.stack(all_log_probs, dim=0)

        return actions, log_probs, output.values
