"""Verification script for Continual Learning strategies."""

import torch
from pathlib import Path
from src.training.trainer import TrainingOrchestrator


def verify_continual_learning():
    print("Verifying Continual Learning Integration...")

    orchestrator = TrainingOrchestrator(model_name="sshleifer/tiny-gpt2")

    texts_task_a = ["The capital of France is Paris.", "France is a country in Europe."]
    texts_task_b = [
        "The capital of Germany is Berlin.",
        "Germany is a country in Europe.",
    ]

    # 1. Verify EWC
    print("\nTesting EWC strategy...")
    results_ewc = orchestrator.train(
        train_texts=texts_task_a,
        epochs=1,
        batch_size=1,
        continual_mode="ewc",
        ewc_lambda=0.5,
    )
    assert orchestrator.ewc_callback is not None
    assert len(orchestrator.ewc_callback.fisher_matrices) > 0
    print("EWC task A completed and Fisher computed.")

    # Fine-tune on Task B with EWC
    orchestrator.train(
        train_texts=texts_task_b, epochs=1, batch_size=1, continual_mode="ewc"
    )
    print("EWC task B completed with penalty.")

    # 2. Verify Replay
    print("\nTesting Experience Replay strategy...")
    orchestrator_replay = TrainingOrchestrator(model_name="sshleifer/tiny-gpt2")
    orchestrator_replay.train(
        train_texts=texts_task_a, epochs=1, batch_size=1, continual_mode="replay"
    )
    assert orchestrator_replay.replay_buffer is not None
    assert len(orchestrator_replay.replay_buffer.buffer) > 0
    print(f"Replay buffer size: {len(orchestrator_replay.replay_buffer.buffer)}")

    # Fine-tune on Task B with Replay
    orchestrator_replay.train(
        train_texts=texts_task_b,
        epochs=1,
        batch_size=1,
        continual_mode="replay",
        replay_ratio=0.5,
    )
    print("Experience Replay task B completed.")

    print("\nContinual Learning verification complete!")


if __name__ == "__main__":
    verify_continual_learning()
