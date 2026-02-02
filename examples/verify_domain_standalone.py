"""Self-contained verification script for Domain Adaptation logic."""

import torch
import torch.nn as nn
from torch.autograd import Function
from typing import Any

# --- Copied from domain_adaptation.py to be self-contained ---


class GradientReversalFunction(Function):
    @staticmethod
    def forward(ctx: Any, x: torch.Tensor, alpha: float) -> torch.Tensor:
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> Any:
        output = grad_output.neg() * ctx.alpha
        return output, None


class GradientReversalLayer(nn.Module):
    def __init__(self, alpha: float = 1.0):
        super().__init__()
        self.alpha = alpha

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return GradientReversalFunction.apply(x, self.alpha)


class MMDLoss(nn.Module):
    def __init__(self, kernel_type: str = "rbf"):
        super().__init__()
        self.kernel_type = kernel_type

    def forward(self, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.kernel_type == "rbf":
            return self._rbf_mmd(source, target)
        else:
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
    def __init__(self, input_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


# --- Verification Tests ---


def test_mmd_loss():
    print("Testing MMDLoss...")
    mmd = MMDLoss(kernel_type="rbf")
    source = torch.randn(10, 5)
    loss_same = mmd(source, source)
    print(f"Loss (same): {loss_same.item():.6f}")
    assert loss_same.item() < 1e-5
    target_shifted = torch.randn(10, 5) + 2.0
    loss_diff = mmd(source, target_shifted)
    print(f"Loss (diff): {loss_diff.item():.6f}")
    assert loss_diff.item() > loss_same.item()
    print("MMDLoss test passed!")


def test_grl():
    print("\nTesting GradientReversalLayer...")
    grl = GradientReversalLayer(alpha=1.0)
    x = torch.randn(5, requires_grad=True)
    y = grl(x)
    assert torch.allclose(x, y)
    loss = y.sum()
    loss.backward()
    print(f"Reversed grad sum: {x.grad.sum().item()}")
    assert torch.allclose(x.grad, -torch.ones(5))
    print("GradientReversalLayer test passed!")


def test_discriminator():
    print("\nTesting DomainDiscriminator...")
    disc = DomainDiscriminator(input_dim=10, hidden_dim=20)
    x = torch.randn(5, 10)
    out = disc(x)
    assert out.shape == (5, 2)
    print("DomainDiscriminator test passed!")


if __name__ == "__main__":
    try:
        test_mmd_loss()
        test_grl()
        test_discriminator()
        print("\nAll Domain Adaptation tests passed (Self-Contained)!")
    except Exception as e:
        print(f"\nTests failed: {e}")
        exit(1)
