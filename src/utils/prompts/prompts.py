"""Prompt engineering toolkit for template management and few-shot examples."""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from jinja2 import Template

logger = logging.getLogger(__name__)


@dataclass
class FewShotExample:
    """A single few-shot example."""

    input: str
    output: str
    metadata: Optional[Dict[str, Any]] = None


class PromptTemplate:
    """A Jinja2-based prompt template."""

    def __init__(self, template_str: str, name: Optional[str] = None):
        """
        Initialize the template.
        Args:
            template_str: The Jinja2 template string.
            name: Optional name for the template.
        """
        self.template = Template(template_str)
        self.name = name

    def render(self, **kwargs: Any) -> str:
        """
        Render the template with given variables.
        Args:
            **kwargs: Variables to inject into the template.
        Returns:
            str: The rendered prompt.
        """
        return self.template.render(**kwargs)


class PromptRegistry:
    """Registry for managing prompt templates on disk."""

    def __init__(self, prompts_dir: Optional[Path] = None):
        """
        Initialize the registry.
        Args:
            prompts_dir: Directory containing prompt YAML files.
        """
        self.prompts_dir = prompts_dir or Path.home() / ".pi-assistant" / "prompts"
        self.prompts_dir.mkdir(parents=True, exist_ok=True)
        self._templates: Dict[str, PromptTemplate] = {}

    def load_all(self) -> None:
        """Load all prompt templates from the prompts directory."""
        if not self.prompts_dir.exists():
            return

        for prompt_file in self.prompts_dir.glob("*.yaml"):
            try:
                with open(prompt_file, "r") as f:
                    data = yaml.safe_load(f)
                    if isinstance(data, dict) and "template" in data:
                        name = prompt_file.stem
                        self._templates[name] = PromptTemplate(
                            template_str=data["template"], name=name
                        )
                        logger.info("Loaded prompt template: %s", name)
            except Exception as e:
                logger.error("Failed to load prompt template %s: %s", prompt_file, e)

    def get_template(self, name: str) -> Optional[PromptTemplate]:
        """
        Get a template by name.
        Args:
            name: The name of the template.
        Returns:
            Optional[PromptTemplate]: The template if found.
        """
        return self._templates.get(name)

    def register(self, name: str, template_str: str) -> None:
        """
        Register a new template and save it to disk.
        Args:
            name: The name of the template.
            template_str: The template string.
        """
        self._templates[name] = PromptTemplate(template_str, name=name)
        prompt_file = self.prompts_dir / f"{name}.yaml"
        try:
            with open(prompt_file, "w") as f:
                yaml.dump({"template": template_str}, f)
        except Exception as e:
            logger.error("Failed to save prompt template %s: %s", name, e)
