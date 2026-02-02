"""Domain adaptation utilities for transfer learning and distribution shifting."""

import torch
import torch.nn as nn
from torch.autograd import Function
from typing import Any, Dict, Optional


class GradientReversalFunction(Function):
    """
    Gradient Reversal Layer (GRL) function.
    Leaves data unchanged during forward pass, but flips the sign of
    gradients during backward pass.
    """

    @staticmethod
    def forward(ctx: Any, x: torch.Tensor, alpha: float) -> torch.Tensor:
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> Any:
        output = grad_output.neg() * ctx.alpha
        return output, None


class GradientReversalLayer(nn.Module):
    """
    Gradient Reversal Layer (GRL) module.
    """

    def __init__(self, alpha: float = 1.0):
        super().__init__()
        self.alpha = alpha

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return GradientReversalFunction.apply(x, self.alpha)


class MMDLoss(nn.Module):
    """
    Maximum Mean Discrepancy (MMD) loss.
    Measures the distance between two probability distributions by comparing
    their means in a Reproducing Kernel Hilbert Space (RKHS).
    """

    def __init__(self, kernel_type: str = "rbf"):
        super().__init__()
        self.kernel_type = kernel_type

    def forward(self, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.kernel_type == "rbf":
            return self._rbf_mmd(source, target)
        else:
            # Linear MMD
            delta = source.mean(0) - target.mean(0)
            return delta.dot(delta)

    def _rbf_mmd(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        xx, yy, xy = torch.mm(x, x.t()), torch.mm(y, y.t()), torch.mm(x, y.t())
        rx = xx.diag().unsqueeze(0).expand_as(xx)
        ry = yy.diag().unsqueeze(0).expand_as(yy)

        dxx = rx.t() + rx - 2 * xx
        dyy = ry.t() + ry - 2 * yy
        dxy = rx.t() + ry - 2 * xy

        XX, YY, XY = (
            torch.zeros_like(dxx),
            torch.zeros_like(dyy),
            torch.zeros_like(dxy),
        )

        bandwidth_range = [1, 2, 5, 8, 10]
        for a in bandwidth_range:
            XX += torch.exp(-0.5 * dxx / a)
            YY += torch.exp(-0.5 * dyy / a)
            XY += torch.exp(-0.5 * dxy / a)

        return XX.mean() + YY.mean() - 2 * XY.mean()


class DomainDiscriminator(nn.Module):
    """
    Module that distinguishes between source and target domains.
    Used in Domain Adversarial Neural Networks (DANN).
    """

    def __init__(self, input_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),  # 2 classes: source vs target
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
