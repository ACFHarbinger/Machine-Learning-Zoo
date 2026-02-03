"""
Training Pipeline for Machine Learning Zoo.

This package provides a modular training pipeline with support for multiple
paradigms: reinforcement learning, supervised learning, unsupervised learning,
semi-supervised learning, and self-supervised learning.
"""

from __future__ import annotations

from .train import create_model, get_mode_registry
from .factory import PipelineFactory

__all__ = ["create_model", "get_mode_registry", "PipelineFactory"]
