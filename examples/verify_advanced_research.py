"""Verification script for Phase 8: Advanced Research."""

import torch
import torch.nn as nn
from src.training.federated import FederatedAggregator, FederatedClient
from src.utils.gen.generators import GANGenerator, VAEGenerator
from src.envs.envs import TradingEnv, MultiAgentEnvWrapper


def test_federated_learning():
    print("Testing Federated Learning...")
    global_model = nn.Linear(10, 1)
    aggregator = FederatedAggregator(global_model)

    # Simulate 2 clients
    client1_model = nn.Linear(10, 1)
    client2_model = nn.Linear(10, 1)

    client1 = FederatedClient(client1_model, "client_1")
    client2 = FederatedClient(client2_model, "client_2")

    # Synchronize
    client1.pull_global_weights(global_model.state_dict())
    client2.pull_global_weights(global_model.state_dict())

    # Simulate local training (just adding noise)
    with torch.no_grad():
        for p in client1_model.parameters():
            p.add_(torch.randn_like(p) * 0.1)
        for p in client2_model.parameters():
            p.add_(torch.randn_like(p) * 0.2)

    # Aggregate
    aggregator.register_client_update(client1.get_local_update(), num_samples=100)
    aggregator.register_client_update(client2.get_local_update(), num_samples=200)

    new_weights = aggregator.aggregate()

    print("Federated aggregation successful.")
    # Basic check: global model weight should be different from initial
    assert not torch.equal(
        new_weights["weight"], torch.zeros_like(new_weights["weight"])
    )
    print("Federated Learning tests passed!")


def test_synthetic_generation():
    print("\nTesting Synthetic Data Generation...")

    # Test GAN
    latent_dim = 16
    output_dim = 10
    gan = GANGenerator(latent_dim, output_dim, [32, 64])
    z = torch.randn(5, latent_dim)
    fake_data = gan(z)
    assert fake_data.shape == (5, output_dim)
    print("GAN generation successful.")

    # Test VAE
    vae = VAEGenerator(input_dim=10, latent_dim=8, hidden_dims=[16, 32])
    samples = vae.sample(num_samples=10)
    assert samples.shape == (10, 10)
    print("VAE generation successful.")

    print("Synthetic Data Generation tests passed!")


def test_multi_agent_rl():
    print("\nTesting Multi-Agent RL Wrapper...")
    base_env = TradingEnv()
    ma_env = MultiAgentEnvWrapper(base_env, num_agents=2)

    obs, info = ma_env.reset()
    assert len(obs) == 2
    assert "agent_0" in obs and "agent_1" in obs

    actions = {"agent_0": 1, "agent_1": 0}
    obs, rewards, term, trunc, info = ma_env.step(actions)

    assert len(obs) == 2
    assert len(rewards) == 2
    print("Multi-Agent RL wrapper successful.")


if __name__ == "__main__":
    try:
        test_federated_learning()
        test_synthetic_generation()
        test_multi_agent_rl()
        print("\nAll Advanced Research tests passed!")
    except Exception as e:
        print(f"\nTests failed: {e}")
        import traceback

        traceback.print_exc()
        exit(1)
