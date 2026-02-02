"""Training orchestration with progress streaming."""

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union
import logging

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import Callback, ModelCheckpoint

from .data_module import PiDataModule
from .lightning_module import PiLightningModule
from .continual import EWCCallback, ReplayBuffer
from .replay_data import ReplayDataset
from .domain_adaptation import MMDLoss, DomainDiscriminator, GradientReversalLayer


logger = logging.getLogger(__name__)


class ProgressCallback(Callback):
    """Callback for streaming training progress."""

    def __init__(self, progress_fn: Optional[Callable[[Dict[str, Any]], None]] = None):
        """
        Initialize the callback.
        Args:
            self: The callback.
            progress_fn: The progress function.
        Returns:
            None
        """
        self.progress_fn = progress_fn
        self.current_epoch = 0
        self.total_epochs = 0

    def on_train_start(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        """
        Called when training starts.
        Args:
            self: The callback.
            trainer: The trainer.
            pl_module: The lightning module.
        Returns:
            None
        """
        self.total_epochs = trainer.max_epochs or 0

    def on_train_epoch_start(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        """
        Called when an epoch starts.
        Args:
            self: The callback.
            trainer: The trainer.
            pl_module: The lightning module.
        Returns:
            None
        """
        self.current_epoch = trainer.current_epoch
        self._emit_progress("epoch_start", {"epoch": self.current_epoch})

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        """
        Called when a batch ends.
        Args:
            self: The callback.
            trainer: The trainer.
            pl_module: The lightning module.
            outputs: The outputs.
            batch: The batch.
            batch_idx: The batch index.
        Returns:
            None
        """
        if batch_idx % 10 == 0:  # Report every 10 batches
            loss = (
                outputs["loss"].item() if isinstance(outputs, dict) else outputs.item()
            )
            self._emit_progress(
                "batch",
                {
                    "epoch": self.current_epoch,
                    "batch": batch_idx,
                    "loss": loss,
                },
            )

    def on_train_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        """
        Called when an epoch ends.
        Args:
            self: The callback.
            trainer: The trainer.
            pl_module: The lightning module.
        Returns:
            None
        """
        metrics = {
            k: v.item() if torch.is_tensor(v) else v
            for k, v in trainer.callback_metrics.items()
        }
        self._emit_progress(
            "epoch_end",
            {
                "epoch": self.current_epoch,
                "metrics": metrics,
            },
        )

    def on_train_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """
        Called when training ends.
        Args:
            self: The callback.
            trainer: The trainer.
            pl_module: The lightning module.
        Returns:
            None
        """
        self._emit_progress("complete", {"epochs": self.current_epoch + 1})

    def _emit_progress(self, event: str, data: Dict[str, Any]) -> None:
        """
        Emit progress.
        Args:
            self: The callback.
            event: The event.
            data: The data.
        Returns:
            None
        """
        # In distributed training, only emit progress from rank 0
        is_rank_zero = True
        try:
            if torch.distributed.is_initialized():
                is_rank_zero = torch.distributed.get_rank() == 0
        except (ImportError, RuntimeError):
            pass

        if is_rank_zero and self.progress_fn:
            self.progress_fn({"event": event, **data})


class TrainingOrchestrator:
    """Orchestrates model training with progress streaming."""

    def __init__(
        self,
        model_name: str = "gpt2",
        output_dir: Optional[Path] = None,
        progress_fn: Optional[Callable[[Dict[str, Any]], None]] = None,
    ):
        """
        Initialize the trainer.
        Args:
            self: The trainer.
            model_name: The name of the model.
            output_dir: The output directory.
            progress_fn: The progress function.
        Returns:
            None
        """
        self.model_name = model_name
        self.output_dir = output_dir or Path("./checkpoints")
        self.progress_fn = progress_fn

        self.module: Optional[PiLightningModule] = None
        self.trainer: Optional[pl.Trainer] = None
        self.ewc_callback: Optional[EWCCallback] = None
        self.replay_buffer: Optional[ReplayBuffer] = None

    def prepare(
        self,
        learning_rate: float = 5e-5,
        warmup_steps: int = 100,
        lora_config: Optional[Dict[str, Any]] = None,
        use_4bit: bool = False,
    ) -> None:
        """
        Prepare the model for training.
        Args:
            self: The trainer.
            learning_rate: The learning rate.
            warmup_steps: The number of warmup steps.
        Returns:
            None
        """
        self.module = PiLightningModule(
            model_name=self.model_name,
            learning_rate=learning_rate,
            warmup_steps=warmup_steps,
            lora_config=lora_config,
            use_4bit=use_4bit,
        )

    def train(
        self,
        train_texts: List[str],
        val_texts: Optional[List[str]] = None,
        epochs: int = 3,
        batch_size: int = 8,
        accelerator: str = "auto",
        strategy: str = "auto",
        devices: Union[str, int] = "auto",
        lora_config: Optional[Dict[str, Any]] = None,
        use_4bit: bool = False,
        deepspeed_config: Optional[Dict[str, Any]] = None,
        continual_mode: Optional[str] = None,
        ewc_lambda: float = 0.4,
        replay_ratio: float = 0.2,
        domain_adaptation_mode: Optional[str] = None,  # "mmd", "dann"
        target_texts: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Run training.
        Args:
            self: The trainer.
            train_texts: The training texts.
            val_texts: The validation texts.
            epochs: The number of epochs.
            batch_size: The batch size.
            accelerator: The accelerator.
            strategy: The strategy (e.g., 'ddp', 'deepspeed').
            devices: The devices to use.
        Returns:
            dict: The training results.
        """
        if self.module is None:
            self.prepare(lora_config=lora_config, use_4bit=use_4bit)

        # Create data module
        data_module = PiDataModule(
            tokenizer=self.module.tokenizer,
            train_texts=train_texts,
            val_texts=val_texts or [],
            batch_size=batch_size,
        )

        # Experience Replay: Wrap dataset
        if continual_mode == "replay" and self.replay_buffer:
            # We wrap the train dataset
            data_module.train_dataset = ReplayDataset(
                base_dataset=data_module.train_dataset,
                replay_buffer=self.replay_buffer,
                replay_ratio=replay_ratio,
            )
            logger.info("Wrapped training dataset for experience replay")

        # Domain Adaptation: Setup
        if domain_adaptation_mode == "mmd":
            self.mmd_loss = MMDLoss()
            logger.info("MMD Domain Adaptation enabled")
        elif domain_adaptation_mode == "dann":
            # We'd need to know the feature dimension for DANN
            # For now, we'll assume a default or infer it from the module
            feature_dim = getattr(self.module.model.config, "hidden_size", 768)
            self.domain_discriminator = DomainDiscriminator(input_dim=feature_dim)
            self.grl = GradientReversalLayer()
            logger.info("DANN Domain Adaptation enabled (feature_dim=%d)", feature_dim)

        # Callbacks
        callbacks = [
            ProgressCallback(self.progress_fn),
            ModelCheckpoint(
                dirpath=self.output_dir,
                filename="model-{epoch:02d}-{val_loss:.2f}",
                save_top_k=2,
                monitor="val_loss" if val_texts else "train_loss",
                mode="min",
            ),
        ]

        # Continual Learning: EWC
        if continual_mode == "ewc":
            if self.ewc_callback is None:
                self.ewc_callback = EWCCallback(ewc_lambda=ewc_lambda)
            callbacks.append(self.ewc_callback)
            logger.info("EWC enabled with lambda=%.2f", ewc_lambda)

        # Continual Learning: Replay
        if continual_mode == "replay":
            if self.replay_buffer is None:
                self.replay_buffer = ReplayBuffer()
            # Note: We need to add samples TO the buffer, usually from previous training data.
            # For simplicity, we'll assume the user might have provided old data or we sample current data.
            # In a real scenario, we'd add samples from train_texts to buffer AFTER training.
            logger.info("Experience Replay enabled with ratio=%.2f", replay_ratio)

        # Handle DeepSpeed strategy
        if isinstance(strategy, str) and strategy.startswith("deepspeed"):
            if deepspeed_config is None:
                # Default to Stage 2 with CPU offload if it's a "performance" phase
                deepspeed_config = self._get_deepspeed_config(stage=2, offload_cpu=True)
            strategy = pl.strategies.DeepSpeedStrategy(config=deepspeed_config)

        # Create trainer
        self.trainer = pl.Trainer(
            max_epochs=epochs,
            accelerator=accelerator,
            strategy=strategy,
            devices=devices,
            callbacks=callbacks,
            enable_progress_bar=False,  # We use our own progress
            logger=False,
        )

        # Train
        if self.trainer:
            self.trainer.fit(self.module, datamodule=data_module)

        # Post-training for Continual Learning
        if continual_mode == "ewc" and self.ewc_callback and self.module:
            self.ewc_callback.compute_fisher(
                self.trainer, self.module, data_module.train_dataloader()
            )

        if continual_mode == "replay" and self.replay_buffer:
            # Add a random subset of current data to replay buffer
            import random

            sample_size = min(100, len(train_texts))
            self.replay_buffer.add_samples(random.sample(train_texts, sample_size))

        return {
            "epochs": epochs,
            "final_loss": self.trainer.callback_metrics.get(
                "train_loss", torch.tensor(0.0)
            ).item(),
            "checkpoint_dir": str(self.output_dir),
        }

    def _get_deepspeed_config(
        self, stage: int = 2, offload_cpu: bool = True
    ) -> Dict[str, Any]:
        """
        Generate a default DeepSpeed configuration.
        Args:
            stage: ZeRO stage (1, 2, or 3).
            offload_cpu: Whether to offload optimizer states/params to CPU.
        Returns:
            dict: DeepSpeed configuration.
        """
        config = {
            "zero_optimization": {
                "stage": stage,
                "allgather_partitions": True,
                "allgather_bucket_size": 2e8,
                "overlap_comm": True,
                "reduce_scatter": True,
                "reduce_bucket_size": 2e8,
                "contiguous_gradients": True,
            },
            "gradient_clipping": 1.0,
            "train_micro_batch_size_per_gpu": 1,
        }

        if offload_cpu:
            config["zero_optimization"]["offload_optimizer"] = {
                "device": "cpu",
                "pin_memory": True,
            }
            if stage == 3:
                config["zero_optimization"]["offload_param"] = {
                    "device": "cpu",
                    "pin_memory": True,
                }

        return config

    def save(self, path: Optional[Path] = None) -> Path:
        """
        Save the trained model.
        Args:
            self: The trainer.
            path: The path to save the model.
        Returns:
            Path: The path to the saved model.
        """
        if self.module is None:
            raise RuntimeError("No model to save")

        save_path = path or self.output_dir / "final_model"
        save_path.mkdir(parents=True, exist_ok=True)

        if hasattr(self.module.model, "save_pretrained"):
            self.module.model.save_pretrained(save_path)
        else:
            # Fallback if somehow it's not a PEFT/HF model
            torch.save(self.module.model.state_dict(), save_path / "model.pt")

        self.module.tokenizer.save_pretrained(save_path)

        return save_path
