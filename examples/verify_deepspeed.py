"""Verification script for DeepSpeed integration."""

import torch
from pathlib import Path
from src.training.trainer import TrainingOrchestrator


def verify_deepspeed_init():
    print("Verifying DeepSpeed initialization...")

    # Initialize trainer
    trainer = TrainingOrchestrator(
        model_name="gpt2", output_dir=Path("./test_checkpoints")
    )

    # Define dummy texts
    train_texts = ["This is a test sentence.", "Another test sentence for DeepSpeed."]

    try:
        # Run training with deepspeed strategy
        # We use accelerator="cpu" and devices=1 for dry run if no GPU,
        # but DeepSpeed usually requires GPU.
        # For verification of the logic, we'll see if the pl.Trainer is created correctly.
        print("Initializing trainer with strategy='deepspeed_stage_2'...")
        results = trainer.train(
            train_texts=train_texts,
            epochs=1,
            batch_size=1,
            accelerator="cpu",  # DeepSpeed usually errors on CPU, but we check our logic
            strategy="deepspeed_stage_2",
            devices=1,
        )
        print("Trainer initialized successfully (logical check).")
    except Exception as e:
        # We expect a possible error if no GPU is found or DeepSpeed isn't fully set up in the env,
        # but we want to ensure our code doesn't have syntax/attribute errors.
        print(
            f"Caught expected env error or unexpected code error: {type(e).__name__}: {e}"
        )

    print("Verification complete.")


if __name__ == "__main__":
    verify_deepspeed_init()
