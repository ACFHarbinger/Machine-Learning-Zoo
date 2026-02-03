"""
Training Utilities and Unified Entry Point for Machine Learning Zoo.

This module serves as the main entry point for all training paradigms.
It provides:
- MODE_REGISTRY: Registry of training mode factory functions
- create_model(): Factory function to create models based on training mode
- Legacy utilities: rollout, train_epoch, train_batch for backward compatibility
"""

import math
import os
import time
from typing import Any, Callable, Dict, Optional, cast

import pytorch_lightning as pl
import torch
from tqdm import tqdm

from ..utils.functions.functions import move_to
from ..utils.functions.model_utils import get_inner_model
from ..utils.logging.log_utils import log_epoch, log_timeseries_values

# Lazy imports for training mode modules to avoid circular imports
_mode_factories: Optional[Dict[str, Callable[..., pl.LightningModule]]] = None


def _load_mode_factories() -> Dict[str, Callable[..., pl.LightningModule]]:
    """Lazily load training mode factory functions."""
    global _mode_factories
    if _mode_factories is not None:
        return _mode_factories

    from .reinforcement_learning import create_rl_model
    from .supervised_learning import create_supervised_model
    from .unsupervised_learning import create_unsupervised_model
    from .semi_supervised_learning import create_semi_supervised_model
    from .self_supervised_learning import create_self_supervised_model
    from .active_learning import create_active_learning_model
    from .continual_learning import create_continual_learning_model
    from .domain_adaptation import create_domain_adaptation_model
    from .federated_learning import create_federated_learning_model
    from .meta import create_meta_learning_model
    from .online_learning import create_online_learning_model

    _mode_factories = {
        "reinforcement_learning": create_rl_model,
        "supervised_learning": create_supervised_model,
        "unsupervised_learning": create_unsupervised_model,
        "semi_supervised_learning": create_semi_supervised_model,
        "self_supervised_learning": create_self_supervised_model,
        "active_learning": create_active_learning_model,
        "continual_learning": create_continual_learning_model,
        "domain_adaptation": create_domain_adaptation_model,
        "federated_learning": create_federated_learning_model,
        "meta": create_meta_learning_model,
        "online_learning": create_online_learning_model,
        # Convenience aliases
        "reinforcement": create_rl_model,
        "supervised": create_supervised_model,
        "unsupervised": create_unsupervised_model,
        "semi_supervised": create_semi_supervised_model,
        "self_supervised": create_self_supervised_model,
        "active": create_active_learning_model,
        "continual": create_continual_learning_model,
        "domain": create_domain_adaptation_model,
        "federated": create_federated_learning_model,
        "maml": create_meta_learning_model,
        "online": create_online_learning_model,
    }
    return _mode_factories


def get_mode_registry() -> Dict[str, Callable[..., pl.LightningModule]]:
    """Get the training mode registry."""
    return _load_mode_factories()


def create_model(cfg: Any, mode: Optional[str] = None) -> pl.LightningModule:
    """
    Factory function to create a model based on training mode.

    Args:
        cfg: Configuration object with train.mode and other settings.
        mode: Optional explicit mode override.

    Returns:
        pl.LightningModule: Configured model for the specified training paradigm.

    Raises:
        ValueError: If the mode is not in the registry.

    Example:
        >>> from src.pipeline.train import create_model
        >>> model = create_model(cfg, mode="reinforcement_learning")
    """
    registry = _load_mode_factories()

    # Determine mode from config or parameter
    if mode is None:
        mode = getattr(cfg.train, "mode", None) if hasattr(cfg, "train") else None
        if mode is None:
            mode = "supervised_learning"  # Default

    if mode not in registry:
        available = [k for k in registry if "_learning" in k]  # Filter aliases
        raise ValueError(f"Unknown training mode: {mode}. Available: {available}")

    factory = registry[mode]
    return factory(cfg)


def rollout(model: torch.nn.Module, dataset: torch.utils.data.Dataset[Any], opts: dict[str, Any]) -> torch.Tensor:
    """
    Perform evaluation rollout on a dataset.

    Args:
        model (nn.Module): The model to evaluate.
        dataset (Dataset): The dataset to roll out on.
        opts (Dict): Evaluation options.

    Returns:
        torch.Tensor: Concatenated results from the rollout.
    """
    model.eval()

    def eval_model_bat(bat: dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Evaluate a single batch.
        """
        with torch.no_grad():
            cost, _ = cast(tuple[torch.Tensor, Any], model(move_to(bat, opts["device"])))
        return cost.data.cpu()

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=opts["eval_batch_size"], pin_memory=True)

    results = []
    for bat in tqdm(dataloader, disable=opts["no_progress_bar"]):
        results.append(eval_model_bat(bat))

    return torch.cat(results, 0)


def clip_grad_norms(param_groups: Any, max_norm: float = math.inf) -> tuple[list[float], list[float]]:
    """
    Clips the norms for all param groups to max_norm and returns gradient norms before clipping
    :param param_groups:
    :param max_norm:
    :return: grad_norms, clipped_grad_norms: list with (clipped) gradient norms per group
    """
    grad_norms = [
        torch.nn.utils.clip_grad_norm_(
            group["params"],
            (max_norm if max_norm > 0 else math.inf),  # Inf so no clipping but still call to calc
            norm_type=2,
        ).item()
        for group in param_groups
    ]
    grad_norms_clipped = [min(g_norm, max_norm) for g_norm in grad_norms] if max_norm > 0 else grad_norms
    return grad_norms, grad_norms_clipped


def train_epoch(  # noqa: PLR0913
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    baseline: Any,
    lr_scheduler: Any,
    epoch: int,
    dataset: torch.utils.data.Dataset[Any],
    tb_logger: Any,
    opts: dict[str, Any],
) -> None:
    """
    Train the model for one epoch.
    """
    print("Start train epoch {}, lr={} for run {}".format(epoch, optimizer.param_groups[0]["lr"], opts["run_name"]))
    is_cuda = torch.cuda.is_available()
    start_time = time.time()
    if not opts["no_tensorboard"]:
        tb_logger.log_value("learnrate_pg0", optimizer.param_groups[0]["lr"], epoch)

    # Put model in train mode and setup dataloader
    step = epoch * opts["batch_size"]
    model.train()
    training_dataloader = torch.utils.data.DataLoader(dataset, batch_size=opts["batch_size"], shuffle=True)

    for batch_id, batch in enumerate(tqdm(training_dataloader, disable=opts["no_progress_bar"])):
        train_batch(model, optimizer, baseline, epoch, batch_id, batch, step, tb_logger, opts)
        step += 1

    epoch_duration = time.time() - start_time
    log_epoch(epoch, epoch_duration, optimizer, opts)
    if is_cuda:
        torch.cuda.empty_cache()

    if (opts["checkpoint_epochs"] != 0 and epoch % opts["checkpoint_epochs"] == 0) or epoch == opts["n_epochs"] - 1:
        print("Saving model and state...")
        torch.save(
            {
                "model": get_inner_model(model).state_dict(),
                "optimizer": optimizer.state_dict(),
                "rng_state": torch.get_rng_state(),
                "cuda_rng_state": torch.cuda.get_rng_state_all(),
                "baseline": baseline.state_dict(),
            },
            os.path.join(opts["save_dir"], f"epoch-{epoch}.pt"),
        )

    # lr_scheduler should be called at end of epoch
    lr_scheduler.step()


def train_batch(  # noqa: PLR0913
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    baseline: Any,
    epoch: int,
    batch_id: int,
    batch: dict[str, torch.Tensor],
    step: int,
    tb_logger: Any,
    opts: dict[str, Any],
) -> None:
    """
    Train the model on a single batch.
    """
    batch = move_to(batch, opts["device"])
    x = batch["Price"]
    y = batch["Labels"]

    # Compute output and loss
    # Cast model output to Tensor to avoid Any issues in strict mode
    output = cast(torch.Tensor, model(x))
    # Use MAE (L1 Loss) as requested for Polymarket
    loss = torch.nn.functional.l1_loss(output, y)

    # Perform backward pass and optimization step
    optimizer.zero_grad()
    loss.backward()

    # Clip gradient norms and get (clipped) gradient norms for logging
    grad_norms = clip_grad_norms(optimizer.param_groups, opts["max_grad_norm"])
    optimizer.step()

    # Logging
    if step % int(opts["log_step"]) == 0:
        log_timeseries_values(loss.item(), grad_norms, epoch, batch_id, step, output, tb_logger, opts)
