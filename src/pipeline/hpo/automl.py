"""
AutoML Module.
Provides hyperparameter optimization using Optuna.
"""

import logging
from typing import Any, Callable, Dict, Optional

import optuna
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


class HyperparameterOptimizer:
    """
    Automated hyperparameter optimization wrapper.
    """

    def __init__(
        self,
        study_name: str = "ml_zoo_optimization",
        storage: Optional[str] = None,
        direction: str = "maximize",
    ):
        self.study = optuna.create_study(
            study_name=study_name,
            storage=storage,
            direction=direction,
            load_if_exists=True,
        )

    def optimize(
        self,
        objective_fn: Callable[[optuna.Trial], float],
        n_trials: int = 100,
        timeout: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Run optimization trials.

        Args:
            objective_fn: A function that takes an optuna.Trial and returns a score.
            n_trials: Number of trials to run.
            timeout: Optimization timeout in seconds.

        Returns:
            Best parameters found.
        """
        logger.info(f"Starting optimization for {n_trials} trials...")
        self.study.optimize(objective_fn, n_trials=n_trials, timeout=timeout)

        logger.info(f"Optimization finished. Best value: {self.study.best_value}")
        return self.study.best_params

    @property
    def best_trial(self) -> optuna.trial.FrozenTrial:
        return self.study.best_trial


def create_lightning_objective(
    trainer_cls: type,
    model_cls: type,
    train_loader: DataLoader,
    val_loader: DataLoader,
    fixed_config: Dict[str, Any],
    param_space: Callable[[optuna.Trial], Dict[str, Any]],
    metric: str = "val_loss",
) -> Callable[[optuna.Trial], float]:
    """
    Helper to create an objective function for PyTorch Lightning models.
    """

    def objective(trial: optuna.Trial) -> float:
        # Sample hyperparameters
        sampled_params = param_space(trial)
        config = {**fixed_config, **sampled_params}

        # Instantiate model and trainer
        model = model_cls(config)
        trainer = trainer_cls(model, **config.get("trainer_kwargs", {}))

        # Run training
        # Note: In a real scenario, we might want to prune trials using Optuna callbacks
        trainer.train(train_loader, val_loader)

        # Return metric
        return trainer.get_best_metric(metric)

    return objective
