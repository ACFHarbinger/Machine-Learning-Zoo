"""Verification script for Phase 6: Explainability & Evaluation."""

import torch
import torch.nn as nn
from src.training.explainability import ExplainabilityModule
from src.training.evaluation import Evaluator


def test_evaluator():
    print("Testing Evaluator...")

    # Classification
    y_true = torch.tensor([0, 1, 2, 0])
    y_pred = torch.tensor([0, 2, 2, 1])  # 2/4 correct
    metrics = Evaluator.classification_metrics(y_true, y_pred)
    print(f"Classification Metrics: {metrics}")
    assert metrics["accuracy"] == 0.5

    # Regression
    y_true_reg = torch.tensor([1.0, 2.0])
    y_pred_reg = torch.tensor([1.5, 2.5])
    metrics_reg = Evaluator.regression_metrics(y_true_reg, y_pred_reg)
    print(f"Regression Metrics: {metrics_reg}")
    assert metrics_reg["mse"] == 0.25

    # Generation (Perplexity)
    logits = torch.randn(2, 5, 100)  # (batch, seq, vocab)
    labels = torch.randint(0, 100, (2, 5))
    metrics_gen = Evaluator.generation_metrics(logits, labels)
    print(f"Generation Metrics: {metrics_gen}")
    assert metrics_gen["perplexity"] > 0

    print("Evaluator tests passed!")


def test_explainability():
    print("\nTesting ExplainabilityModule...")

    class MockModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(10, 2)

        def forward(self, x):
            return self.linear(x)

    model = MockModel()
    inputs = torch.randn(1, 10, requires_grad=True)

    # Test Integrated Gradients
    attributions = ExplainabilityModule.integrated_gradients(
        model, inputs, target_idx=0, steps=10
    )
    print(f"Attributions shape: {attributions.shape}")
    assert attributions.shape == inputs.shape

    # Test Attention extraction (Mocking a HF-like model)
    class MockTransformer(nn.Module):
        def forward(self, input_ids, attention_mask=None, output_attentions=False):
            class Output:
                def __init__(self):
                    self.attentions = (
                        torch.randn(1, 2, 4, 4),
                    )  # (batch, head, seq, seq)

            return Output()

    transformer = MockTransformer()
    input_ids = torch.zeros(1, 4, dtype=torch.long)
    maps = ExplainabilityModule.get_attention_maps(transformer, input_ids)
    print(f"Number of attention maps: {len(maps)}")
    assert len(maps) == 1
    assert maps[0].shape == (1, 2, 4, 4)

    print("Explainability tests passed!")


if __name__ == "__main__":
    try:
        test_evaluator()
        test_explainability()
        print("\nAll Phase 6 tests passed!")
    except Exception as e:
        print(f"\nTests failed: {e}")
        exit(1)
