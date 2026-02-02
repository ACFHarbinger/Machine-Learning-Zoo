"""
Heads Package.

Task-specific output layers that attach to backbone features.
"""

from .base import HEAD_REGISTRY, Head, HeadConfig, register_head
from .classification import ClassificationHead, ClassificationHeadConfig
from .regression import RegressionHead, RegressionHeadConfig
from .rl_policy import PolicyOutput, RLPolicyHead, RLPolicyHeadConfig
from .sequence import SequenceHead, SequenceHeadConfig
from .multi_agent_policy import (
    MultiAgentPolicyHead,
    MultiAgentPolicyHeadConfig,
    MultiAgentPolicyOutput,
)

__all__ = [
    # Base
    "Head",
    "HeadConfig",
    "HEAD_REGISTRY",
    "register_head",
    # Implementations
    "ClassificationHead",
    "ClassificationHeadConfig",
    "RegressionHead",
    "RegressionHeadConfig",
    "RLPolicyHead",
    "RLPolicyHeadConfig",
    "PolicyOutput",
    "SequenceHead",
    "SequenceHeadConfig",
    "MultiAgentPolicyHead",
    "MultiAgentPolicyHeadConfig",
    "MultiAgentPolicyOutput",
]
