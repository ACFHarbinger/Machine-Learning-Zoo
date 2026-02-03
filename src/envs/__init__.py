"""
nglab Arena - Python wrapper for Rust RL trading environment

This module provides Gymnasium-compatible environments for:
- CLOB (Central Limit Order Book) trading
- Polymarket prediction markets
- General trading simulation
- Multi-agent cooperative and competitive scenarios
"""

from __future__ import annotations

from .envs import ClobEnv, PolymarketEnv, TradingEnv
from .factory import EnvFactory
from .multi_agent import (
    CompetitiveArenaEnv,
    CooperativeGatheringEnv,
    MultiAgentEnvBase,
    MultiAgentEnvConfig,
)

get_env = EnvFactory.get_env

__all__ = [
    "ClobEnv",
    "PolymarketEnv",
    "TradingEnv",
    "get_env",
    "MultiAgentEnvBase",
    "MultiAgentEnvConfig",
    "CooperativeGatheringEnv",
    "CompetitiveArenaEnv",
]
