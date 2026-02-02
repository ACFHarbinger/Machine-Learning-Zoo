"""Fairness auditing toolkit for measuring bias in models."""

import logging
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch

logger = logging.getLogger(__name__)


class FairnessAuditor:
    """
    Toolkit for auditing model fairness across different demographic groups.
    """

    @staticmethod
    def demographic_parity_difference(
        y_pred: torch.Tensor, sensitive_attr: torch.Tensor
    ) -> float:
        """
        Compute the maximum difference in selection rate (P(y_pred=1 | sensitive_attr))
        between any two groups.
        """
        groups = torch.unique(sensitive_attr)
        selection_rates = []

        for group in groups:
            mask = sensitive_attr == group
            if torch.sum(mask) == 0:
                continue
            rate = torch.mean(y_pred[mask].float()).item()
            selection_rates.append(rate)

        if not selection_rates:
            return 0.0

        return max(selection_rates) - min(selection_rates)

    @staticmethod
    def disparate_impact_ratio(
        y_pred: torch.Tensor, sensitive_attr: torch.Tensor
    ) -> float:
        """
        Compute the ratio of selection rates between the least and most advantaged groups.
        Ideally should be > 0.8 (80% rule).
        """
        groups = torch.unique(sensitive_attr)
        selection_rates = []

        for group in groups:
            mask = sensitive_attr == group
            if torch.sum(mask) == 0:
                continue
            rate = torch.mean(y_pred[mask].float()).item()
            selection_rates.append(rate)

        if not selection_rates or min(selection_rates) == 0:
            return 0.0 if not selection_rates else float("inf")

        return min(selection_rates) / max(selection_rates)

    @staticmethod
    def equalized_odds_difference(
        y_true: torch.Tensor, y_pred: torch.Tensor, sensitive_attr: torch.Tensor
    ) -> float:
        """
        Compute the maximum difference in TPR or FPR between any two groups.
        Greater of |TPR_a - TPR_b| and |FPR_a - FPR_b|.
        """
        groups = torch.unique(sensitive_attr)
        tprs = []
        fprs = []

        for group in groups:
            mask = sensitive_attr == group
            if torch.sum(mask) == 0:
                continue

            y_t = y_true[mask]
            y_p = y_pred[mask]

            tp = torch.sum((y_t == 1) & (y_p == 1)).float()
            fn = torch.sum((y_t == 1) & (y_p == 0)).float()
            fp = torch.sum((y_t == 0) & (y_p == 1)).float()
            tn = torch.sum((y_t == 0) & (y_p == 0)).float()

            tpr = (tp / (tp + fn)).item() if (tp + fn) > 0 else 0.0
            fpr = (fp / (fp + tn)).item() if (fp + tn) > 0 else 0.0

            tprs.append(tpr)
            fprs.append(fpr)

        if not tprs:
            return 0.0

        tpr_diff = max(tprs) - min(tprs)
        fpr_diff = max(fprs) - min(fprs)

        return max(tpr_diff, fpr_diff)

    def audit(
        self,
        y_true: torch.Tensor,
        y_pred: torch.Tensor,
        sensitive_attr: torch.Tensor,
        metrics: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        """
        Perform a comprehensive fairness audit.
        """
        if metrics is None:
            metrics = ["demographic_parity", "disparate_impact", "equalized_odds"]

        results = {}

        if "demographic_parity" in metrics:
            results["demographic_parity_diff"] = self.demographic_parity_difference(
                y_pred, sensitive_attr
            )

        if "disparate_impact" in metrics:
            results["disparate_impact_ratio"] = self.disparate_impact_ratio(
                y_pred, sensitive_attr
            )

        if "equalized_odds" in metrics:
            results["equalized_odds_diff"] = self.equalized_odds_difference(
                y_true, y_pred, sensitive_attr
            )

        return results
