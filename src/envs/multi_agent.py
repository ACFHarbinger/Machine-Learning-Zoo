"""
Multi-Agent Reinforcement Learning Environments.

Provides cooperative and competitive multi-agent environments that
follow the Gymnasium API pattern, supporting variable agent counts
and both shared and independent reward structures.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

__all__ = [
    "MultiAgentEnvConfig",
    "MultiAgentEnvBase",
    "CooperativeGatheringEnv",
    "CompetitiveArenaEnv",
]


@dataclass
class MultiAgentEnvConfig:
    """Configuration for multi-agent environments."""

    num_agents: int = 4
    grid_size: int = 10
    max_steps: int = 200
    reward_type: str = "shared"  # "shared", "individual", or "mixed"
    communication_dim: int = 0  # 0 disables communication
    render_mode: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


class MultiAgentEnvBase(gym.Env[dict[str, NDArray[Any]], dict[str, Any]], ABC):
    """
    Abstract base class for multi-agent environments.

    Observations and actions are dictionaries keyed by agent id.
    Each agent receives its own observation and submits its own action.
    """

    metadata: dict[str, Any] = {"render_modes": ["human"]}  # noqa: RUF012

    def __init__(self, config: MultiAgentEnvConfig) -> None:
        super().__init__()
        self.config = config
        self.num_agents = config.num_agents
        self.agents = [f"agent_{i}" for i in range(config.num_agents)]
        self.current_step = 0

        # Subclasses must set these
        self._agent_obs_space: spaces.Space[Any] | None = None
        self._agent_act_space: spaces.Space[Any] | None = None

    @property
    def observation_space(self) -> spaces.Dict:  # type: ignore[override]
        assert self._agent_obs_space is not None
        return spaces.Dict(
            {agent: self._agent_obs_space for agent in self.agents}
        )

    @observation_space.setter
    def observation_space(self, value: Any) -> None:
        pass

    @property
    def action_space(self) -> spaces.Dict:  # type: ignore[override]
        assert self._agent_act_space is not None
        return spaces.Dict(
            {agent: self._agent_act_space for agent in self.agents}
        )

    @action_space.setter
    def action_space(self, value: Any) -> None:
        pass

    @abstractmethod
    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, NDArray[Any]], dict[str, Any]]:
        pass

    @abstractmethod
    def step(
        self,
        actions: dict[str, Any],
    ) -> tuple[
        dict[str, NDArray[Any]],
        dict[str, float],
        bool,
        bool,
        dict[str, Any],
    ]:
        pass

    def _compute_rewards(
        self,
        individual_rewards: dict[str, float],
    ) -> dict[str, float]:
        """Apply reward sharing strategy."""
        if self.config.reward_type == "shared":
            total = sum(individual_rewards.values())
            shared = total / max(len(individual_rewards), 1)
            return {agent: shared for agent in self.agents}
        elif self.config.reward_type == "mixed":
            total = sum(individual_rewards.values())
            shared = total / max(len(individual_rewards), 1)
            return {
                agent: 0.5 * individual_rewards[agent] + 0.5 * shared
                for agent in self.agents
            }
        else:
            return individual_rewards


class CooperativeGatheringEnv(MultiAgentEnvBase):
    """
    Cooperative multi-agent environment where agents must
    collect resources scattered on a grid.

    Agents share a common goal and receive shared rewards.
    Optional communication channels allow agents to broadcast
    messages to each other.

    Observation per agent:
        - Agent position (2,)
        - Other agent positions (num_agents - 1, 2)
        - Resource positions (num_resources, 2)
        - Optional received messages (num_agents - 1, communication_dim)

    Action per agent:
        Discrete(5): 0=Stay, 1=Up, 2=Down, 3=Left, 4=Right
        If communication is enabled, action is a dict with "move" and "message".
    """

    def __init__(
        self,
        config: MultiAgentEnvConfig | None = None,
        num_resources: int = 8,
    ) -> None:
        config = config or MultiAgentEnvConfig()
        super().__init__(config)
        self.num_resources = num_resources
        self.grid_size = config.grid_size

        # Per-agent observation: own pos + other positions + resource positions
        obs_dim = 2 + (config.num_agents - 1) * 2 + num_resources * 2
        if config.communication_dim > 0:
            obs_dim += (config.num_agents - 1) * config.communication_dim

        self._agent_obs_space = spaces.Box(
            low=0.0,
            high=float(config.grid_size),
            shape=(obs_dim,),
            dtype=np.float64,
        )

        if config.communication_dim > 0:
            self._agent_act_space = spaces.Dict({
                "move": spaces.Discrete(5),
                "message": spaces.Box(
                    low=-1.0,
                    high=1.0,
                    shape=(config.communication_dim,),
                    dtype=np.float64,
                ),
            })
        else:
            self._agent_act_space = spaces.Discrete(5)

        # State
        self._agent_positions: dict[str, NDArray[Any]] = {}
        self._resource_positions: NDArray[Any] = np.zeros((num_resources, 2))
        self._collected: NDArray[np.bool_] = np.zeros(num_resources, dtype=np.bool_)
        self._messages: dict[str, NDArray[Any]] = {}

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, NDArray[Any]], dict[str, Any]]:
        super().reset(seed=seed)
        rng = np.random.default_rng(seed)

        self.current_step = 0

        # Random initial positions
        for agent in self.agents:
            self._agent_positions[agent] = rng.uniform(
                0, self.grid_size, size=(2,)
            )

        # Random resource positions
        self._resource_positions = rng.uniform(
            0, self.grid_size, size=(self.num_resources, 2)
        )
        self._collected = np.zeros(self.num_resources, dtype=np.bool_)

        # Clear messages
        if self.config.communication_dim > 0:
            for agent in self.agents:
                self._messages[agent] = np.zeros(self.config.communication_dim)

        return self._get_observations(), {}

    def step(
        self,
        actions: dict[str, Any],
    ) -> tuple[
        dict[str, NDArray[Any]],
        dict[str, float],
        bool,
        bool,
        dict[str, Any],
    ]:
        self.current_step += 1
        individual_rewards: dict[str, float] = {agent: 0.0 for agent in self.agents}

        # Process actions
        move_delta = {
            0: np.array([0.0, 0.0]),
            1: np.array([0.0, 1.0]),
            2: np.array([0.0, -1.0]),
            3: np.array([-1.0, 0.0]),
            4: np.array([1.0, 0.0]),
        }

        for agent in self.agents:
            action = actions.get(agent, 0)

            if self.config.communication_dim > 0 and isinstance(action, dict):
                move = int(action["move"])
                self._messages[agent] = np.array(action["message"], dtype=np.float64)
            else:
                move = int(action)

            # Move agent
            delta = move_delta.get(move, np.array([0.0, 0.0]))
            self._agent_positions[agent] = np.clip(
                self._agent_positions[agent] + delta,
                0.0,
                float(self.grid_size),
            )

            # Check resource collection
            pos = self._agent_positions[agent]
            for r_idx in range(self.num_resources):
                if self._collected[r_idx]:
                    continue
                dist = np.linalg.norm(pos - self._resource_positions[r_idx])
                if dist < 1.0:
                    self._collected[r_idx] = True
                    individual_rewards[agent] += 1.0

        rewards = self._compute_rewards(individual_rewards)

        terminated = bool(np.all(self._collected))
        truncated = self.current_step >= self.config.max_steps

        info = {
            "collected": int(np.sum(self._collected)),
            "total_resources": self.num_resources,
        }

        return self._get_observations(), rewards, terminated, truncated, info

    def _get_observations(self) -> dict[str, NDArray[Any]]:
        observations: dict[str, NDArray[Any]] = {}

        for agent in self.agents:
            parts: list[NDArray[Any]] = []
            # Own position
            parts.append(self._agent_positions[agent])
            # Other agent positions
            for other in self.agents:
                if other != agent:
                    parts.append(self._agent_positions[other])
            # Resource positions (mark collected as zeros)
            for r_idx in range(self.num_resources):
                if self._collected[r_idx]:
                    parts.append(np.zeros(2))
                else:
                    parts.append(self._resource_positions[r_idx])
            # Messages from other agents
            if self.config.communication_dim > 0:
                for other in self.agents:
                    if other != agent:
                        parts.append(self._messages[other])

            observations[agent] = np.concatenate(parts).astype(np.float64)

        return observations

    def render(self) -> None:
        if self.config.render_mode == "human":
            collected = int(np.sum(self._collected))
            print(
                f"Step {self.current_step}: "
                f"Collected {collected}/{self.num_resources} resources"
            )


class CompetitiveArenaEnv(MultiAgentEnvBase):
    """
    Competitive multi-agent environment where agents compete
    for territory on a grid.

    Each cell on the grid can be claimed by an agent. Agents
    receive individual rewards proportional to the territory
    they control. Agents can capture cells occupied by others.

    Observation per agent:
        - Agent position (2,)
        - Territory ownership grid (grid_size * grid_size,)
        - Other agent positions (num_agents - 1, 2)

    Action per agent:
        Discrete(5): 0=Stay, 1=Up, 2=Down, 3=Left, 4=Right
    """

    def __init__(
        self,
        config: MultiAgentEnvConfig | None = None,
    ) -> None:
        config = config or MultiAgentEnvConfig(reward_type="individual")
        super().__init__(config)
        self.grid_size = config.grid_size

        obs_dim = 2 + config.grid_size * config.grid_size + (config.num_agents - 1) * 2
        self._agent_obs_space = spaces.Box(
            low=-1.0,
            high=float(config.num_agents),
            shape=(obs_dim,),
            dtype=np.float64,
        )
        self._agent_act_space = spaces.Discrete(5)

        # Territory grid: -1 = unclaimed, 0..N-1 = agent index
        self._territory: NDArray[Any] = np.full(
            (config.grid_size, config.grid_size), -1, dtype=np.int32
        )
        self._agent_positions: dict[str, NDArray[Any]] = {}

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, NDArray[Any]], dict[str, Any]]:
        super().reset(seed=seed)
        rng = np.random.default_rng(seed)

        self.current_step = 0
        self._territory = np.full(
            (self.grid_size, self.grid_size), -1, dtype=np.int32
        )

        # Place agents at random grid positions
        for i, agent in enumerate(self.agents):
            pos = rng.integers(0, self.grid_size, size=(2,)).astype(np.float64)
            self._agent_positions[agent] = pos
            # Claim starting cell
            self._territory[int(pos[0]), int(pos[1])] = i

        return self._get_observations(), {}

    def step(
        self,
        actions: dict[str, Any],
    ) -> tuple[
        dict[str, NDArray[Any]],
        dict[str, float],
        bool,
        bool,
        dict[str, Any],
    ]:
        self.current_step += 1

        move_delta = {
            0: np.array([0.0, 0.0]),
            1: np.array([0.0, 1.0]),
            2: np.array([0.0, -1.0]),
            3: np.array([-1.0, 0.0]),
            4: np.array([1.0, 0.0]),
        }

        individual_rewards: dict[str, float] = {}

        for i, agent in enumerate(self.agents):
            action = int(actions.get(agent, 0))
            delta = move_delta.get(action, np.array([0.0, 0.0]))
            new_pos = np.clip(
                self._agent_positions[agent] + delta,
                0.0,
                float(self.grid_size - 1),
            )
            self._agent_positions[agent] = new_pos

            # Claim the cell
            row, col = int(new_pos[0]), int(new_pos[1])
            prev_owner = self._territory[row, col]
            self._territory[row, col] = i

            # Reward: +1 for claiming unclaimed, +0.5 for capturing opponent's cell
            if prev_owner == -1:
                individual_rewards[agent] = 1.0
            elif prev_owner != i:
                individual_rewards[agent] = 0.5
            else:
                individual_rewards[agent] = 0.0

        # Add territory bonus: fraction of grid owned
        total_cells = self.grid_size * self.grid_size
        for i, agent in enumerate(self.agents):
            owned = int(np.sum(self._territory == i))
            individual_rewards[agent] += owned / total_cells

        rewards = self._compute_rewards(individual_rewards)

        # Terminate when all cells are claimed or max steps reached
        unclaimed = int(np.sum(self._territory == -1))
        terminated = unclaimed == 0
        truncated = self.current_step >= self.config.max_steps

        territory_counts = {
            agent: int(np.sum(self._territory == i))
            for i, agent in enumerate(self.agents)
        }
        info = {
            "territory": territory_counts,
            "unclaimed": unclaimed,
        }

        return self._get_observations(), rewards, terminated, truncated, info

    def _get_observations(self) -> dict[str, NDArray[Any]]:
        observations: dict[str, NDArray[Any]] = {}
        flat_territory = self._territory.flatten().astype(np.float64)

        for agent in self.agents:
            parts: list[NDArray[Any]] = []
            parts.append(self._agent_positions[agent])
            parts.append(flat_territory)
            for other in self.agents:
                if other != agent:
                    parts.append(self._agent_positions[other])
            observations[agent] = np.concatenate(parts).astype(np.float64)

        return observations

    def render(self) -> None:
        if self.config.render_mode == "human":
            territory_counts = {
                agent: int(np.sum(self._territory == i))
                for i, agent in enumerate(self.agents)
            }
            print(
                f"Step {self.current_step}: "
                f"Territory: {territory_counts}"
            )
