"""Unified evaluation metrics for multiple tasks."""

import torch
from typing import Any, Dict, List, Optional, Union
import numpy as np


class Evaluator:
    """
    Unified evaluator for common ML tasks.
    """

    @staticmethod
    def classification_metrics(
        y_true: torch.Tensor, y_pred: torch.Tensor
    ) -> Dict[str, float]:
        """
        Compute accuracy, precision, recall, and F1 for classification.

        Args:
            y_true: Ground truth labels.
            y_pred: Predicted labels/logits.
        """
        if y_pred.shape != y_true.shape:
            y_pred = y_pred.argmax(dim=-1)

        correct = (y_pred == y_true).sum().item()
        total = y_true.size(0)
        accuracy = correct / total if total > 0 else 0.0

        # Simple macro-averaging logic for F1/Precision/Recall
        # (Could use sklearn or torchmetrics if available)
        classes = torch.unique(y_true)
        f1_scores = []
        for c in classes:
            tp = ((y_pred == c) & (y_true == c)).sum().item()
            fp = ((y_pred == c) & (y_true != c)).sum().item()
            fn = ((y_pred != c) & (y_true == c)).sum().item()

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (
                2 * precision * recall / (precision + recall)
                if (precision + recall) > 0
                else 0.0
            )
            f1_scores.append(f1)

        return {
            "accuracy": accuracy,
            "f1_macro": float(np.mean(f1_scores)) if f1_scores else 0.0,
        }

    @staticmethod
    def regression_metrics(
        y_true: torch.Tensor, y_pred: torch.Tensor
    ) -> Dict[str, float]:
        """
        Compute MSE and MAE for regression.
        """
        mse = torch.mean((y_true - y_pred) ** 2).item()
        mae = torch.mean(torch.abs(y_true - y_pred)).item()
        return {"mse": mse, "mae": mae}

    @staticmethod
    def generation_metrics(
        logits: torch.Tensor, labels: torch.Tensor
    ) -> Dict[str, float]:
        """
        Compute perplexity for sequence generation.

        Args:
            logits: Predicted logits (batch, seq, vocab).
            labels: Target token IDs (batch, seq).
        """
        # Cross entropy loss
        loss_fct = torch.nn.CrossEntropyLoss()
        # Shift so that tokens < n predict n
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
        )
        perplexity = torch.exp(loss).item()

        return {"perplexity": perplexity}

    @classmethod
    def evaluate(
        cls, task: str, y_true: torch.Tensor, y_pred: torch.Tensor
    ) -> Dict[str, float]:
        """
        Generic evaluation entry point.
        """
        if task == "classification":
            return cls.classification_metrics(y_true, y_pred)
        elif task == "regression":
            return cls.regression_metrics(y_true, y_pred)
        elif task == "generation":
            return cls.generation_metrics(y_pred, y_true)
        else:
            raise ValueError(f"Unknown task type: {task}")
