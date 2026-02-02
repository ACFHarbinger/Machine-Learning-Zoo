"""PyTorch Lightning Module for model training."""

from typing import Any, Dict, Optional, Union

import pytorch_lightning as pl
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training


class PiLightningModule(pl.LightningModule):
    """Lightning module for fine-tuning language models."""

    def __init__(
        self,
        model_name: str = "gpt2",
        learning_rate: float = 5e-5,
        warmup_steps: int = 100,
        weight_decay: float = 0.01,
        lora_config: Optional[Dict[str, Any]] = None,
        use_4bit: bool = False,
        domain_adaptation_config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the lightning module.
        Args:
            self: The lightning module.
            model_name: The name of the model.
            learning_rate: The learning rate.
            warmup_steps: The number of warmup steps.
            weight_decay: The weight decay.
        Returns:
            None
        """
        super().__init__()
        self.save_hyperparameters()

        self.model_name = model_name
        self.learning_rate = learning_rate
        self.warmup_steps = warmup_steps
        self.weight_decay = weight_decay
        self.da_config = domain_adaptation_config or {}
        self.da_mode = self.da_config.get("mode")
        self.discriminator = self.da_config.get("discriminator")
        self.mmd_loss_fn = self.da_config.get("mmd_loss_fn")
        self.grl = self.da_config.get("grl")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Quantization config for QLoRA
        quant_config = None
        if use_4bit:
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )

        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quant_config,
            device_map="auto" if use_4bit else None,
        )

        # Setup PEFT if lora_config is provided
        if lora_config:
            if use_4bit:
                self.model = prepare_model_for_kbit_training(self.model)

            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                r=lora_config.get("r", 8),
                lora_alpha=lora_config.get("lora_alpha", 32),
                lora_dropout=lora_config.get("lora_dropout", 0.1),
                target_modules=lora_config.get("target_modules", ["q_proj", "v_proj"]),
            )
            self.model = get_peft_model(self.model, peft_config)
            self.model.print_trainable_parameters()

        # Ensure pad token exists
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            # For PEFT models, the config is in base_model
            config = getattr(
                self.model, "config", getattr(self.model, "model", None).config
            )
            config.pad_token_id = self.tokenizer.eos_token_id

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> Any:
        """
        Forward pass.
        Args:
            self: The lightning module.
            input_ids: The input IDs.
            attention_mask: The attention mask.
            labels: The labels.
        Returns:
            Any: The output of the model.
        """
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        """
        Training step.
        Args:
            self: The lightning module.
            batch: The batch.
            batch_idx: The batch index.
        Returns:
            torch.Tensor: The loss.
        """
        outputs = self(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )
        task_loss = outputs.loss
        total_loss = task_loss

        # Domain Adaptation Loss
        if self.da_mode == "mmd" and "target_batch" in batch and self.mmd_loss_fn:
            # We assume batch contains both source and target for MMD
            target_batch = batch["target_batch"]
            target_outputs = self.model(
                input_ids=target_batch["input_ids"],
                attention_mask=target_batch["attention_mask"],
                output_hidden_states=True,
            )
            # Use average pooling over hidden states
            source_features = outputs.hidden_states[-1].mean(dim=1)
            target_features = target_outputs.hidden_states[-1].mean(dim=1)
            da_loss = self.mmd_loss_fn(source_features, target_features)
            total_loss += self.da_config.get("lambda", 0.1) * da_loss
            self.log("da_mmd_loss", da_loss, on_step=True)

        elif self.da_mode == "dann" and self.discriminator and self.grl:
            # DANN: Task loss + Adversarial Domain loss
            # We need hidden states for features
            source_features = outputs.hidden_states[-1].mean(dim=1)
            reversed_features = self.grl(source_features)
            domain_preds = self.discriminator(reversed_features)
            # Binary labels: 0 for source (current batch)
            domain_labels = torch.zeros(
                domain_preds.size(0), dtype=torch.long, device=self.device
            )
            da_loss = nn.functional.cross_entropy(domain_preds, domain_labels)
            total_loss += self.da_config.get("lambda", 0.1) * da_loss
            self.log("da_dann_loss", da_loss, on_step=True)

        self.log("train_loss", total_loss, prog_bar=True, on_step=True, on_epoch=True)
        return total_loss

    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        """
        Validation step.
        Args:
            self: The lightning module.
            batch: The batch.
            batch_idx: The batch index.
        Returns:
            torch.Tensor: The loss.
        """
        outputs = self(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )
        loss = outputs.loss

        self.log("val_loss", loss, prog_bar=True, on_epoch=True)
        return loss

    def configure_optimizers(self) -> Dict[str, Any]:
        """
        Configure optimizer and scheduler.
        Args:
            self: The lightning module.
        Returns:
            dict: The optimizer and scheduler.
        """
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=self.warmup_steps,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }

    def generate(self, prompt: str, max_new_tokens: int = 100, **kwargs: Any) -> str:
        """
        Generate text from a prompt.
        Args:
            self: The lightning module.
            prompt: The prompt.
            max_new_tokens: The maximum number of new tokens.
            **kwargs: Additional arguments.
        Returns:
            str: The generated text.
        """
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                pad_token_id=self.tokenizer.pad_token_id,
                **kwargs,
            )

        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
