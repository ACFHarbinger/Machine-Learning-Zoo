"""Training orchestration with progress streaming."""

import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import Callback, ModelCheckpoint
from pytorch_lightning.loggers import MLFlowLogger, WandbLogger

from ..continual_learning.dataset import ReplayDataset
from ..continual_learning.engine import EWCCallback, ReplayBuffer
from ..domain_adaptation.engine import (
    DomainDiscriminator,
    GradientReversalLayer,
    MMDLoss,
)
from ..evaluation.engine import Evaluator
from ..explainability.engine import ExplainabilityModule
from .data_module import PiDataModule
from .lightning_module import PiLightningModule

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
        distillation_config: Optional[Dict[str, Any]] = None,
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
            distillation_config=distillation_config,
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
        distillation_config: Optional[Dict[str, Any]] = None,
        tracking_config: Optional[Dict[str, Any]] = None,
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
            self.prepare(
                lora_config=lora_config,
                use_4bit=use_4bit,
                distillation_config=distillation_config,
            )

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

        if isinstance(strategy, str) and strategy.startswith("deepspeed"):
            if deepspeed_config is None:
                # Default to Stage 2 with CPU offload if it's a "performance" phase
                deepspeed_config = self._get_deepspeed_config(stage=2, offload_cpu=True)
            strategy = pl.strategies.DeepSpeedStrategy(config=deepspeed_config)

        # Loggers
        loggers = []
        if tracking_config:
            if tracking_config.get("use_wandb"):
                loggers.append(
                    WandbLogger(
                        project=tracking_config.get("project", "ml-zoo"),
                        name=tracking_config.get("name"),
                    )
                )
            if tracking_config.get("use_mlflow"):
                loggers.append(
                    MLFlowLogger(
                        experiment_name=tracking_config.get("project", "ml-zoo"),
                        run_name=tracking_config.get("name"),
                    )
                )

        # Create trainer
        self.trainer = pl.Trainer(
            max_epochs=epochs,
            accelerator=accelerator,
            strategy=strategy,
            devices=devices,
            callbacks=callbacks,
            enable_progress_bar=False,  # We use our own progress
            logger=loggers if loggers else False,
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

        results = {
            "epochs": epochs,
            "final_loss": self.trainer.callback_metrics.get(
                "train_loss", torch.tensor(0.0)
            ).item(),
            "checkpoint_dir": str(self.output_dir),
        }

        # Run final evaluation if requested
        evaluation_results = {}
        if val_texts and self.module:
            try:
                evaluation_results = self.evaluate(val_texts)
                logger.info("Final Evaluation: %s", evaluation_results)
            except Exception as e:
                logger.error("Evaluation failed: %s", e)

        results["evaluation"] = evaluation_results
        return results

    def evaluate(self, texts: List[str], task: str = "generation") -> Dict[str, float]:
        """
        Evaluate the model on a set of texts.
        Args:
            texts: The evaluation texts.
            task: The task type.
        Returns:
            dict: Evaluation metrics.
        """
        if self.module is None or self.trainer is None:
            raise ValueError("Model not prepared or trained.")

        self.module.eval()
        import torch

        from .data_module import PiDataModule

        dm = PiDataModule(
            tokenizer=self.module.tokenizer,
            train_texts=[],
            val_texts=texts,
            batch_size=8,
        )
        dm.setup("validate")

        all_logits = []
        all_labels = []

        with torch.no_grad():
            for batch in dm.val_dataloader():
                batch = {k: v.to(self.module.device) for k, v in batch.items()}
                output = self.module.model(**batch)
                all_logits.append(output.logits.cpu())
                all_labels.append(batch["labels"].cpu())

        logits = torch.cat(all_logits, dim=0)
        labels = torch.cat(all_labels, dim=0)

        return Evaluator.evaluate(task, labels, logits)

    def explain(self, text: str, target_idx: int) -> Dict[str, Any]:
        """
        Generate explainability data for a single input.
        Args:
            text: The input text.
            target_idx: The target token/class index to explain.
        Returns:
            dict: Explainability data.
        """
        if self.module is None:
            raise ValueError("Model not prepared.")

        inputs = self.module.tokenizer(text, return_tensors="pt").to(self.module.device)
        input_ids = inputs["input_ids"]

        # Get embeddings for IG
        embeddings = self.module.model.get_input_embeddings()(input_ids)

        # Integrated Gradients
        def model_forward(emb):
            return self.module.model(inputs_embeds=emb).logits[:, -1, :]

        attributions = ExplainabilityModule.integrated_gradients(
            model_forward, embeddings, target_idx
        )

        # Attention maps
        attention_maps = ExplainabilityModule.get_attention_maps(
            self.module.model, input_ids
        )

        tokens = self.module.tokenizer.convert_ids_to_tokens(input_ids[0])
        viz = ExplainabilityModule.visualize_attention(attention_maps, tokens)

        return {
            "attributions": attributions.sum(dim=-1).cpu().numpy().tolist(),
            "attention": viz,
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
