"""Verification script for Prompt Engineering Toolkit."""

from pathlib import Path
from src.utils.prompts.prompts import PromptRegistry, PromptTemplate
from src.utils.prompts.few_shot import FewShotRegistry, FewShotExample


def verify_prompts():
    print("Verifying Prompt Engineering Toolkit...")

    # 1. Test PromptTemplate
    print("\nTesting PromptTemplate...")
    template_str = "Hello {{ name }}! Welcome to {{ place }}."
    template = PromptTemplate(template_str)
    rendered = template.render(name="World", place="Machine Learning Zoo")
    print(f"Rendered: {rendered}")
    assert rendered == "Hello World! Welcome to Machine Learning Zoo."

    # 2. Test PromptRegistry
    print("\nTesting PromptRegistry...")
    registry = PromptRegistry(prompts_dir=Path("./test_prompts"))
    registry.register("welcome", "Welcome to the Zoo, {{ user }}!")
    registry.load_all()
    loaded_template = registry.get_template("welcome")
    assert loaded_template is not None
    print(f"Loaded template rendered: {loaded_template.render(user='Alice')}")

    # 3. Test FewShotRegistry
    print("\nTesting FewShotRegistry...")
    fs_registry = FewShotRegistry(examples_dir=Path("./test_examples"))
    example = FewShotExample(input="Capital of France", output="Paris")
    fs_registry.add_example("geography", example)
    fs_registry.load_all()
    examples = fs_registry.get_examples("geography", count=1)
    assert len(examples) == 1
    print(f"Retrieved example: {examples[0].input} -> {examples[0].output}")

    print("\nVerification complete!")


if __name__ == "__main__":
    verify_prompts()
