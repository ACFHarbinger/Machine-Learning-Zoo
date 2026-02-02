"""Registry for managing and retrieving few-shot examples."""

import json
import logging
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

from .prompts import FewShotExample

logger = logging.getLogger(__name__)


class FewShotRegistry:
    """Registry for managing libraries of few-shot examples."""

    def __init__(self, examples_dir: Optional[Path] = None):
        """
        Initialize the registry.
        Args:
            examples_dir: Directory containing example JSON files.
        """
        self.examples_dir = examples_dir or Path.home() / ".pi-assistant" / "examples"
        self.examples_dir.mkdir(parents=True, exist_ok=True)
        self._examples: Dict[str, List[FewShotExample]] = {}

    def load_all(self) -> None:
        """Load all examples from the examples directory."""
        if not self.examples_dir.exists():
            return

        for example_file in self.examples_dir.glob("*.json"):
            try:
                with open(example_file, "r") as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        name = example_file.stem
                        self._examples[name] = [
                            FewShotExample(**ex) if isinstance(ex, dict) else ex
                            for ex in data
                        ]
                        logger.info(
                            "Loaded %d examples for: %s",
                            len(self._examples[name]),
                            name,
                        )
            except Exception as e:
                logger.error("Failed to load examples from %s: %s", example_file, e)

    def get_examples(
        self, category: str, count: int = 3, strategy: str = "random"
    ) -> List[FewShotExample]:
        """
        Get examples from a specific category.
        Args:
            category: The category (library) of examples.
            count: Number of examples to retrieve.
            strategy: Selection strategy ('random', 'first').
        Returns:
            List[FewShotExample]: The retrieved examples.
        """
        examples = self._examples.get(category, [])
        if not examples:
            return []

        if strategy == "random":
            return random.sample(examples, min(count, len(examples)))
        else:
            return examples[:count]

    def add_example(self, category: str, example: FewShotExample) -> None:
        """
        Add an example to a category and save to disk.
        Args:
            category: The category/library name.
            example: The few-shot example to add.
        """
        if category not in self._examples:
            self._examples[category] = []

        self._examples[category].append(example)
        self._save_category(category)

    def _save_category(self, category: str) -> None:
        """Save a category's examples to disk."""
        examples = self._examples.get(category, [])
        example_file = self.examples_dir / f"{category}.json"

        try:
            with open(example_file, "w") as f:
                json.dump(
                    [
                        {
                            "input": ex.input,
                            "output": ex.output,
                            "metadata": ex.metadata,
                        }
                        for ex in examples
                    ],
                    f,
                    indent=2,
                )
        except Exception as e:
            logger.error("Failed to save examples for category %s: %s", category, e)
