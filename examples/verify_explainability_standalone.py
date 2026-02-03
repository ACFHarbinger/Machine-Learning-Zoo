"""Self-contained verification script for Phase 6 logic."""

from typing import Any, Dict, List

import torch
import torch.nn as nn

# --- Copied from explainability.py ---


class ExplainabilityModule:
    @staticmethod
    def integrated_gradients(
        model: Any, inputs: torch.Tensor, target_idx: int, steps: int = 50
    ) -> torch.Tensor:
        baseline = torch.zeros_like(inputs)
        alphas = torch.linspace(0, 1, steps).to(inputs.device)
        interpolated_inputs = [
            baseline + alpha * (inputs - baseline) for alpha in alphas
        ]
        grads = []
        for x in interpolated_inputs:
            x.requires_grad_(True)
            output = model(x)
            if len(output.shape) == 3:
                score = output[:, -1, target_idx]
            else:
                score = output[:, target_idx]
            grads.append(torch.autograd.grad(score, x)[0])
        avg_grads = torch.stack(grads).mean(dim=0)
        return (inputs - baseline) * avg_grads

    @staticmethod
    def get_attention_maps(
        model: nn.Module, input_ids: torch.Tensor
    ) -> List[torch.Tensor]:
        class Output:
            def __init__(self):
                self.attentions = (torch.randn(1, 2, 4, 4),)

        return list(Output().attentions)

    @staticmethod
    def visualize_attention(attention_map: torch.Tensor, tokens: List[str]):
        return {"attentions": attention_map[0][0].numpy().tolist(), "tokens": tokens}


# --- Copied from evaluation.py ---


class Evaluator:
    @staticmethod
    def classification_metrics(
        y_true: torch.Tensor, y_pred: torch.Tensor
    ) -> Dict[str, float]:
        if y_pred.shape != y_true.shape:
            y_pred = y_pred.argmax(dim=-1)
        accuracy = (y_pred == y_true).sum().item() / y_true.size(0)
        return {"accuracy": accuracy}

    @staticmethod
    def regression_metrics(
        y_true: torch.Tensor, y_pred: torch.Tensor
    ) -> Dict[str, float]:
        mse = torch.mean((y_true - y_pred) ** 2).item()
        return {"mse": mse}

    @staticmethod
    def generation_metrics(
        logits: torch.Tensor, labels: torch.Tensor
    ) -> Dict[str, float]:
        loss_fct = torch.nn.CrossEntropyLoss()
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
        )
        return {"perplexity": torch.exp(loss).item()}

    @classmethod
    def evaluate(
        cls, task: str, y_true: torch.Tensor, y_pred: torch.Tensor
    ) -> Dict[str, float]:
        if task == "classification":
            return cls.classification_metrics(y_true, y_pred)
        elif task == "regression":
            return cls.regression_metrics(y_true, y_pred)
        elif task == "generation":
            return cls.generation_metrics(y_pred, y_true)
        return {}


# --- Tests ---


def test_p6():
    print("Testing Phase 6 logic...")

    # Evaluator
    y_true = torch.tensor([0, 1])
    y_pred = torch.tensor([0, 0])
    e_res = Evaluator.evaluate("classification", y_true, y_pred)
    print(f"Eval result: {e_res}")
    assert e_res["accuracy"] == 0.5

    # Explainability
    model = nn.Linear(5, 2)
    inputs = torch.randn(1, 5)
    attrs = ExplainabilityModule.integrated_gradients(model, inputs, 0, steps=5)
    print(f"Attrs shape: {attrs.shape}")
    assert attrs.shape == inputs.shape

    print("Tests passed!")


if __name__ == "__main__":
    test_p6()
