"""Visual Training Dashboard API for Machine Learning Zoo.

This module provides a FastAPI-based WebSocket API for real-time training monitoring
and pipeline management. It connects to the React dashboard for:
- Listing available model types from Enums
- WebSocket streaming of training metrics
- Pipeline validation and export
"""

from __future__ import annotations

import json
import logging
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from ..enums.models import DeepModelType, HelperModelType, MacModelType

app = FastAPI(
    title="ML Zoo Training Dashboard API",
    description="Real-time monitoring and management of ML training runs.",
    version="2.0.0",
)

# Enable CORS for dashboard connection
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Request/Response Models
# -----------------------------------------------------------------------------


class PipelineNode(BaseModel):
    """A node in the model pipeline graph."""

    id: str
    type: str
    data: dict[str, Any]
    position: dict[str, float]


class PipelineEdge(BaseModel):
    """A connection between nodes."""

    source: str
    target: str


class TrainingConfig(BaseModel):
    """Training configuration from the dashboard."""

    epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 0.001
    seq_len: int = 30
    pred_len: int = 1
    train_split: float = 0.8
    mode: str = "supervised"
    hpo_algorithm: str = "none"
    hpo_trials: int = 20


class PipelineRequest(BaseModel):
    """Complete pipeline request from dashboard."""

    nodes: list[PipelineNode]
    edges: list[PipelineEdge]
    config: TrainingConfig


class ModelTypeResponse(BaseModel):
    """Response containing all available model types."""

    deep: list[str]
    mac: list[str]
    helper: list[str]


# -----------------------------------------------------------------------------
# Active WebSocket Connections
# -----------------------------------------------------------------------------

active_connections: list[WebSocket] = []


async def broadcast_training_update(data: dict[str, Any]) -> None:
    """Broadcast training progress to all connected clients."""
    for connection in active_connections:
        try:
            await connection.send_json(data)
        except Exception:
            pass


# -----------------------------------------------------------------------------
# REST Endpoints
# -----------------------------------------------------------------------------


@app.get("/")
async def root() -> dict[str, str]:
    """Health check and welcome message."""
    return {"status": "healthy", "service": "ML Zoo Dashboard API"}


@app.get("/api/models/types", response_model=ModelTypeResponse)
async def get_model_types() -> ModelTypeResponse:
    """Return all available model types from Enums."""
    return ModelTypeResponse(
        deep=[m.value for m in DeepModelType],
        mac=[m.value for m in MacModelType],
        helper=[m.value for m in HelperModelType],
    )


@app.post("/api/pipeline/validate")
async def validate_pipeline(pipeline: PipelineRequest) -> JSONResponse:
    """Validate a model pipeline configuration."""
    errors = []

    # Check for at least one model node
    model_nodes = [n for n in pipeline.nodes if n.type == "model"]
    if not model_nodes:
        errors.append("Pipeline must contain at least one model node")

    # Check for data input node
    data_inputs = [n for n in pipeline.nodes if n.type == "data" and n.data.get("nodeKind") == "input"]
    if not data_inputs:
        errors.append("Pipeline must have a data input node")

    # Check for valid connections
    if not pipeline.edges:
        errors.append("Pipeline nodes must be connected")

    if errors:
        return JSONResponse(status_code=400, content={"valid": False, "errors": errors})

    return JSONResponse(content={"valid": True, "node_count": len(pipeline.nodes)})


@app.post("/api/pipeline/export")
async def export_pipeline(pipeline: PipelineRequest) -> JSONResponse:
    """Export pipeline to Python training config."""
    # Build the training command
    model_nodes = [n for n in pipeline.nodes if n.type == "model"]

    if not model_nodes:
        return JSONResponse(status_code=400, content={"error": "No model nodes in pipeline"})

    # For simplicity, take the first model node
    main_model = model_nodes[0]
    model_name = main_model.data.get("label", "LSTM")
    model_params = main_model.data.get("params", {})

    config_dict = {
        "model": {
            "name": model_name,
            "params": model_params,
        },
        "training": {
            "epochs": pipeline.config.epochs,
            "batch_size": pipeline.config.batch_size,
            "learning_rate": pipeline.config.learning_rate,
            "seq_len": pipeline.config.seq_len,
            "pred_len": pipeline.config.pred_len,
            "train_split": pipeline.config.train_split,
        },
        "mode": pipeline.config.mode,
        "hpo": {
            "algorithm": pipeline.config.hpo_algorithm,
            "trials": pipeline.config.hpo_trials,
        },
    }

    return JSONResponse(content={"config": config_dict})


# -----------------------------------------------------------------------------
# WebSocket Endpoints
# -----------------------------------------------------------------------------


@app.websocket("/ws/training/{run_id}")
async def training_websocket(websocket: WebSocket, run_id: str) -> None:
    """WebSocket endpoint for real-time training metrics."""
    await websocket.accept()
    active_connections.append(websocket)
    logger.info(f"Training WebSocket connected: run_id={run_id}")

    try:
        while True:
            # Receive messages from client (e.g., pause/resume commands)
            data = await websocket.receive_text()
            message = json.loads(data)

            if message.get("type") == "ping":
                await websocket.send_json({"type": "pong"})

    except WebSocketDisconnect:
        active_connections.remove(websocket)
        logger.info(f"Training WebSocket disconnected: run_id={run_id}")


@app.websocket("/ws/inference")
async def inference_websocket(websocket: WebSocket) -> None:
    """WebSocket endpoint for live inference results."""
    await websocket.accept()
    logger.info("Inference WebSocket connected")

    try:
        while True:
            data = await websocket.receive_text()
            # Process inference request
            request = json.loads(data)
            # TODO: Route to inference engine
            await websocket.send_json({"status": "received", "request": request})

    except WebSocketDisconnect:
        logger.info("Inference WebSocket disconnected")


# -----------------------------------------------------------------------------
# Entrypoint
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8001)
