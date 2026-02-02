"""
Training Service.

Manages training runs with start/stop/status capabilities.
Exposes training pipeline to IPC for agent-initiated training.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from ..enums import RunStatus

from ..configs import RunInfo

logger = logging.getLogger(__name__)

__all__ = ["TrainingService"]


class TrainingService:
    """
    Service for managing training runs.

    Provides async interface for:
    - Starting new training runs
    - Stopping running jobs
    - Querying run status
    - Listing historical runs
    - Deploying trained models as inference tools
    - Running predictions on deployed models
    """

    def __init__(
        self,
        output_dir: str = "~/.pi-assistant/models",
        registry: Any | None = None,
        device_manager: Any | None = None,
    ) -> None:
        self.output_dir = Path(output_dir).expanduser()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.registry = registry
        self.device_manager = device_manager

        # Active runs
        self._runs: dict[str, RunInfo] = {}
        self._tasks: dict[str, asyncio.Task] = {}  # type: ignore

        # Load run history from disk
        self._load_history()

    def _load_history(self) -> None:
        """Load historical run info from disk."""
        # Check for runs.json in output dir
        history_file = self.output_dir / "runs.json"
        if history_file.exists():
            try:
                import json

                with open(history_file) as f:
                    data = json.load(f)
                for run_data in data.get("runs", []):
                    self._runs[run_data["run_id"]] = RunInfo(
                        run_id=run_data["run_id"],
                        status=RunStatus(run_data["status"]),
                        config=run_data.get("config", {}),
                        started_at=datetime.fromisoformat(run_data["started_at"])
                        if run_data.get("started_at")
                        else None,
                        completed_at=datetime.fromisoformat(run_data["completed_at"])
                        if run_data.get("completed_at")
                        else None,
                        metrics=run_data.get("metrics", {}),
                        error=run_data.get("error"),
                        model_path=run_data.get("model_path"),
                        tool_name=run_data.get("tool_name"),
                        deployed=run_data.get("deployed", False),
                        deploy_device=run_data.get("deploy_device"),
                        task_type=run_data.get("task_type"),
                    )
            except Exception as e:
                logger.warning(f"Failed to load run history: {e}")

    def _save_history(self) -> None:
        """Persist run history to disk."""
        import json

        history_file = self.output_dir / "runs.json"
        data = {
            "runs": [
                {
                    "run_id": run.run_id,
                    "status": run.status.value,
                    "config": run.config,
                    "started_at": run.started_at.isoformat()
                    if run.started_at
                    else None,
                    "completed_at": run.completed_at.isoformat()
                    if run.completed_at
                    else None,
                    "metrics": run.metrics,
                    "error": run.error,
                    "model_path": run.model_path,
                    "tool_name": run.tool_name,
                    "deployed": run.deployed,
                    "deploy_device": run.deploy_device,
                    "task_type": run.task_type,
                }
                for run in self._runs.values()
            ]
        }

        with open(history_file, "w") as f:
            json.dump(data, f, indent=2)

    async def start(self, config: dict[str, Any]) -> str:
        """
        Start a new training run.

        Args:
            config: Training configuration dict with:
                - backbone: Name of backbone to use
                - head: Name of head to use
                - backbone_config: Dict of backbone parameters
                - head_config: Dict of head parameters
                - training: Training hyperparameters
                - data: Dataset configuration

        Returns:
            run_id: Unique identifier for this run
        """
        run_id = str(uuid.uuid4())[:8]

        run_info = RunInfo(
            run_id=run_id,
            status=RunStatus.PENDING,
            config=config,
            started_at=datetime.now(),
        )
        self._runs[run_id] = run_info

        # Spawn training task
        task = asyncio.create_task(self._run_training(run_id, config))
        self._tasks[run_id] = task

        logger.info(f"Started training run: {run_id}")
        return run_id

    async def _run_training(self, run_id: str, config: dict[str, Any]) -> None:
        """Execute training in background."""
        run_info = self._runs[run_id]
        run_info.status = RunStatus.RUNNING

        try:
            # Import here to avoid circular deps and lazy load heavy modules
            from ..models.composed import build_model
            from ..pipeline.accelerated import (
                AcceleratedTrainer,
                AcceleratedTrainerConfig,
            )

            # Build model
            model = build_model(
                backbone_name=config.get("backbone", "transformer"),
                head_name=config.get("head", "classification"),
                backbone_config=config.get("backbone_config", {}),
                head_config=config.get("head_config", {}),
            )

            # Create dummy data for now (would be replaced by actual data loading)
            import torch
            from torch.utils.data import DataLoader, TensorDataset

            data_config = config.get("data", {})
            batch_size = data_config.get("batch_size", 32)
            num_samples = data_config.get("num_samples", 1000)
            seq_len = data_config.get("seq_len", 50)
            input_dim = config.get("backbone_config", {}).get("input_dim", 10)

            # Generate synthetic data
            x = torch.randn(num_samples, seq_len, input_dim)
            y = torch.randint(
                0, config.get("head_config", {}).get("num_classes", 10), (num_samples,)
            )

            dataset = TensorDataset(x, y)
            train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

            # Configure trainer
            training_config = config.get("training", {})
            trainer_config = AcceleratedTrainerConfig(
                max_epochs=training_config.get("max_epochs", 10),
                learning_rate=training_config.get("learning_rate", 3e-4),
                batch_size=batch_size,
                output_dir=str(self.output_dir),
                run_name=run_id,
                mixed_precision=training_config.get("mixed_precision", "no"),
            )

            # Train
            trainer = AcceleratedTrainer(
                model=model,
                train_loader=train_loader,
                config=trainer_config,
                loss_fn=torch.nn.CrossEntropyLoss(),
            )

            result = trainer.train()

            # Update run info
            run_info.status = RunStatus.COMPLETED
            run_info.completed_at = datetime.now()
            run_info.metrics = result
            run_info.model_path = str(self.output_dir / run_id / "best_model.pt")
            run_info.task_type = config.get("head", "classification")

            # Cleanup GPU memory after training
            del model, trainer
            import gc

            gc.collect()
            try:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass

            logger.info(f"Training completed: {run_id}")

        except asyncio.CancelledError:
            run_info.status = RunStatus.CANCELLED
            run_info.completed_at = datetime.now()
            logger.info(f"Training cancelled: {run_id}")

        except Exception as e:
            run_info.status = RunStatus.FAILED
            run_info.completed_at = datetime.now()
            run_info.error = str(e)
            logger.error(f"Training failed: {run_id} - {e}")

        finally:
            self._save_history()
            if run_id in self._tasks:
                del self._tasks[run_id]

    async def stop(self, run_id: str) -> bool:
        """
        Stop a running training job.

        Args:
            run_id: ID of the run to stop

        Returns:
            True if stopped, False if not found or not running
        """
        if run_id not in self._tasks:
            return False

        task = self._tasks[run_id]
        task.cancel()

        try:
            await task
        except asyncio.CancelledError:
            pass

        return True

    async def status(self, run_id: str) -> dict[str, Any]:
        """
        Get status of a training run.

        Args:
            run_id: ID of the run

        Returns:
            Dict with run status and metrics
        """
        if run_id not in self._runs:
            return {"error": f"Run not found: {run_id}"}

        run = self._runs[run_id]
        return {
            "run_id": run.run_id,
            "status": run.status.value,
            "started_at": run.started_at.isoformat() if run.started_at else None,
            "completed_at": run.completed_at.isoformat() if run.completed_at else None,
            "metrics": run.metrics,
            "error": run.error,
            "model_path": run.model_path,
            "tool_name": run.tool_name,
            "deployed": run.deployed,
            "deploy_device": run.deploy_device,
            "task_type": run.task_type,
        }

    async def list_runs(self) -> list[dict[str, Any]]:
        """
        List all training runs.

        Returns:
            List of run status dicts
        """
        return [
            await self.status(run_id)
            for run_id in sorted(self._runs.keys(), reverse=True)
        ]

    async def deploy(
        self, run_id: str, tool_name: str, device: str | None = None
    ) -> dict[str, Any]:
        """
        Deploy a trained model as an inference tool.

        Loads the checkpoint, rebuilds the model, and registers it in the model registry
        so it can be used for predictions via the agent's tool system.

        Args:
            run_id: ID of the completed training run.
            tool_name: Name to register the deployed model under.
            device: Target device (e.g. "cpu", "cuda:0"). Auto-selected if None.

        Returns:
            Dict with deployment status.
        """
        if self.registry is None:
            raise ValueError("ModelRegistry not available — cannot deploy")

        run = self._runs.get(run_id)
        if run is None:
            raise ValueError(f"Run not found: {run_id}")
        if run.status != RunStatus.COMPLETED:
            raise ValueError(
                f"Run {run_id} is not completed (status={run.status.value})"
            )
        if not run.model_path:
            raise ValueError(f"Run {run_id} has no saved model path")

        import torch

        from ..models.composed import build_model

        checkpoint_path = Path(run.model_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        # Rebuild model architecture from run config
        model = build_model(
            backbone_name=run.config.get("backbone", "transformer"),
            head_name=run.config.get("head", "classification"),
            backbone_config=run.config.get("backbone_config", {}),
            head_config=run.config.get("head_config", {}),
        )

        # Load trained weights
        state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state_dict)
        model.eval()

        # Determine target device
        if device is None:
            if self.device_manager:
                size_mb = sum(
                    p.nelement() * p.element_size() for p in model.parameters()
                ) // (1024 * 1024)
                device = self.device_manager.best_device_for("inference", size_mb)
            else:
                device = "cuda:0" if torch.cuda.is_available() else "cpu"

        model = model.to(device)

        # Register as a LoadedModel in the registry
        from ..configs.sidecar_model import LoadedModel

        size_mb = sum(p.nelement() * p.element_size() for p in model.parameters()) // (
            1024 * 1024
        )

        loaded = LoadedModel(
            model_id=tool_name,
            model=model,
            tokenizer=None,
            backend="transformers",
            device=device,
            model_size_mb=size_mb,
            metadata={
                "run_id": run_id,
                "task_type": run.task_type,
                "config": run.config,
            },
        )
        self.registry._loaded[tool_name] = loaded

        # Update run info
        run.tool_name = tool_name
        run.deployed = True
        run.deploy_device = device
        self._save_history()

        logger.info(
            "Deployed run %s as tool '%s' on %s (%d MB)",
            run_id,
            tool_name,
            device,
            size_mb,
        )
        return {
            "status": "deployed",
            "run_id": run_id,
            "tool_name": tool_name,
            "device": device,
            "model_size_mb": size_mb,
            "task_type": run.task_type,
        }

    async def predict(self, tool_name: str, input_data: Any) -> dict[str, Any]:
        """
        Run inference on a deployed model.

        Args:
            tool_name: Name of the deployed model tool.
            input_data: Input data (list of numbers, list of lists, etc.).

        Returns:
            Dict with prediction results.
        """
        if self.registry is None:
            raise ValueError("ModelRegistry not available — cannot predict")

        loaded = self.registry.get_model(tool_name)
        if loaded is None:
            raise ValueError(f"Deployed model not found: {tool_name}")

        import torch

        model = loaded.model
        model.eval()

        # Convert input to tensor
        if isinstance(input_data, list):
            tensor_input = torch.tensor(input_data, dtype=torch.float32)
        else:
            raise ValueError(f"Unsupported input_data type: {type(input_data)}")

        # Ensure correct shape (add batch dim if needed)
        if tensor_input.dim() == 1:
            tensor_input = tensor_input.unsqueeze(0)
        elif tensor_input.dim() == 2:
            tensor_input = tensor_input.unsqueeze(0)

        tensor_input = tensor_input.to(loaded.device)

        with torch.no_grad():
            output = model(tensor_input)

        # Post-process based on task type
        task_type = loaded.metadata.get("task_type", "classification")
        if task_type == "classification":
            probs = torch.softmax(output, dim=-1)
            predicted_class = int(torch.argmax(probs, dim=-1).item())
            confidence = float(probs[0, predicted_class].item())
            return {
                "prediction": predicted_class,
                "confidence": confidence,
                "probabilities": probs[0].tolist(),
                "tool_name": tool_name,
                "task_type": task_type,
            }
        else:
            # Regression or raw output
            return {
                "output": output.tolist(),
                "tool_name": tool_name,
                "task_type": task_type,
            }

    def list_deployed(self) -> list[dict[str, Any]]:
        """List all deployed model tools."""
        deployed = []
        for run in self._runs.values():
            if run.deployed and run.tool_name:
                deployed.append(
                    {
                        "run_id": run.run_id,
                        "tool_name": run.tool_name,
                        "deploy_device": run.deploy_device,
                        "task_type": run.task_type,
                        "metrics": run.metrics,
                        "loaded": (
                            self.registry.get_model(run.tool_name) is not None
                            if self.registry
                            else False
                        ),
                    }
                )
        return deployed
