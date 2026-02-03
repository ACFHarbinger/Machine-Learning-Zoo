"""Verification script for Domain Adaptation utilities."""

import torch
from src.pipeline.training.domain_adaptation import (
    DomainDiscriminator,
    GradientReversalLayer,
    MMDLoss,
)


def test_mmd_loss():
    print("Testing MMDLoss...")
    mmd = MMDLoss(kernel_type="rbf")

    # Same distribution
    source = torch.randn(10, 5)
    target = torch.randn(10, 5)
    loss_same = mmd(source, source)
    print(f"Loss (same): {loss_same.item():.6f}")
    assert loss_same.item() < 1e-5

    # Different distribution
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

    # Forward pass should be identity
    assert torch.allclose(x, y)

    # Backward pass should reverse gradient
    loss = y.sum()
    loss.backward()
    print(f"Original grad sum: {torch.ones(5).sum().item()}")
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
        print("\nAll Domain Adaptation tests passed!")
    except Exception as e:
        print(f"\nTests failed: {e}")
        exit(1)
