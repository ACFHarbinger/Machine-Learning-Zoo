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
from .multi_agent import (
    CooperativeGatheringEnv,
    CompetitiveArenaEnv,
    MultiAgentEnvBase,
    MultiAgentEnvConfig,
)

__all__ = [
    "ClobEnv",
    "PolymarketEnv",
    "TradingEnv",
    "MultiAgentEnvBase",
    "MultiAgentEnvConfig",
    "CooperativeGatheringEnv",
    "CompetitiveArenaEnv",
]
