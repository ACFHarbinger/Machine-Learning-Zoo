"""Automated prompt optimization utilities."""

import logging
from typing import Any, Dict, Optional

from ...inference.engine import InferenceEngine
from .prompts import PromptTemplate

logger = logging.getLogger(__name__)


class PromptOptimizer:
    """Helper for automated prompt optimization using an LLM."""

    def __init__(self, engine: InferenceEngine):
        """
        Initialize the optimizer.
        Args:
            engine: The inference engine to use for optimization.
        """
        self.engine = engine

    async def optimize(
        self,
        candidate_prompt: str,
        objective: str,
        feedback: Optional[str] = None,
        provider: str = "local",
    ) -> str:
        """
        Optimize a prompt using an LLM.
        Args:
            candidate_prompt: The prompt to optimize.
            objective: The goal or metric for optimization.
            feedback: Optional feedback on the current candidate.
            provider: The provider to use for optimization.
        Returns:
            str: The optimized prompt.
        """
        optimization_prompt = f"""# Prompt Optimizer

Your task is to improve a candidate prompt based on an objective and optional feedback.

## Candidate Prompt:
{candidate_prompt}

## Objective:
{objective}

## Feedback (if any):
{feedback or "None"}

## Instructions:
1. Analyze the candidate prompt.
2. Identify areas for improvement (clarity, specificity, context).
3. Generate a NEW, IMPROVED version of the prompt.
4. Respond ONLY with the improved prompt text. No explanations.

Improved Prompt:"""

        result = await self.engine.complete(
            prompt=optimization_prompt,
            provider=provider,
            temperature=0.3,  # Lower temperature for stable optimization
        )

        return result.get("text", candidate_prompt).strip()
