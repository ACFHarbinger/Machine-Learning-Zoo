"""
Verification script for the Trading Arena environment.
"""

import os
import sys

try:
    from ..envs import TradingEnv

    print("Successfully imported TradingEnv")

    env = TradingEnv(lookback=30, max_steps=100)
    print("Created TradingEnv")

    obs, info = env.reset()
    print(f"Reset environment. Observation shape: {obs.shape}")

    obs, info = env.reset()
    print(f"Reset environment. Observation shape: {obs.shape}")

    print("Running simulation loop...")
    for i in range(100):
        action = 1  # Buy
        obs, reward, terminated, truncated, info = env.step(action)
        if i % 10 == 0:
            print(
                f"Step {i}: Reward={reward:.4f}, Portfolio={info.get('portfolio_value'):.2f}"
            )
        if terminated or truncated:
            print("Episode finished")
            break

    print("Verification successful!")

except ImportError as e:
    print(f"ImportError: {e}")
    sys.exit(1)
except Exception as e:
    print(f"Error: {e}")
    sys.exit(1)
