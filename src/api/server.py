"""Production inference server for the Machine Learning Zoo."""

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from ..models.time_series import TimeSeriesBackbone
from ..utils.io.model_versioning import ModelMetadata, load_model_with_metadata
from .ab_testing import ABTestingManager

AB_MANAGER = ABTestingManager()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Machine Learning Zoo - Inference API",
    description="Production-ready inference service for multiple model types.",
    version="1.0.0",
)

# Global model cache to avoid reloading
MODEL_CACHE: Dict[str, Any] = {}


class PredictionRequest(BaseModel):
    model_path: Optional[str] = Field(
        None, description="Path to the model checkpoint or HF repo ID"
    )
    inputs: List[Any] = Field(..., description="Input data for inference")
    task: str = Field("time_series", description="Type of task (time_series, text)")
    engine: str = Field("torch", description="Inference engine to use (torch, vllm)")
    ab_test_id: Optional[str] = Field(None, description="ID of the A/B test to use")
    session_id: Optional[str] = Field(
        None, description="Session ID for sticky traffic splitting"
    )


class PredictionResponse(BaseModel):
    status: str
    prediction: List[Any]
    latency_ms: float
    model_metadata: Optional[Dict[str, Any]] = None
    variant_info: Optional[Dict[str, Any]] = None


class ABExperimentRequest(BaseModel):
    test_id: str
    variants: List[Dict[str, Any]]
    traffic_split: Optional[List[float]] = None


def get_model(
    model_path: str, task: str, engine: str = "torch"
) -> Tuple[Any, ModelMetadata]:
    """Load and cache model."""
    cache_key = f"{model_path}:{engine}"
    if cache_key in MODEL_CACHE:
        return MODEL_CACHE[cache_key]

    path = Path(model_path)
    # vLLM can load from HF ID directly, torch needs path
    if not path.exists() and engine != "vllm":
        raise FileNotFoundError(f"Model not found at {model_path}")

    metadata = None
    if path.exists():
        metadata_path = path.with_suffix(".json")
        if metadata_path.exists():
            with open(metadata_path) as f:
                metadata = ModelMetadata.from_json(f.read())
        else:
            # Try to load from checkpoint directly
            try:
                checkpoint = torch.load(path, map_location="cpu")
                if "metadata" in checkpoint:
                    metadata = ModelMetadata.from_dict(checkpoint["metadata"])
            except Exception:
                pass

    # Fallback/Default metadata for vLLM HF models
    if metadata is None:
        metadata = ModelMetadata(
            version="1.0.0",
            model_type="llm" if task == "text" else "time_series",
            training_config={"engine": engine, "model_name": model_path},
        )

    if task == "time_series":
        ts_model = TimeSeriesBackbone(metadata.training_config.get("model", {}))
        ts_model, _ = load_model_with_metadata(ts_model, path, map_location="cpu")
        ts_model.eval()
        MODEL_CACHE[cache_key] = (ts_model, metadata)
        return ts_model, metadata
    elif task == "text" and engine == "vllm":
        from ..pipeline.inference.vllm_engine import VLLMEngine

        vllm_model = VLLMEngine(model_name=model_path)
        MODEL_CACHE[cache_key] = (vllm_model, metadata)
        return vllm_model, metadata
    else:
        raise NotImplementedError(
            f"Task {task} with engine {engine} not yet implemented."
        )


@app.get("/health")
async def health() -> Dict[str, Any]:
    """Health check endpoint."""
    return {"status": "healthy", "gpu": torch.cuda.is_available()}


@app.get("/models")
async def list_loaded_models() -> Dict[str, List[str]]:
    """List models currently in memory."""
    return {"loaded_models": list(MODEL_CACHE.keys())}


@app.post("/v1/ab_experiment")
async def create_ab_experiment(request: ABExperimentRequest) -> Dict[str, str]:
    """
    Create a new A/B experiment.
    """
    try:
        AB_MANAGER.create_experiment(
            request.test_id, request.variants, request.traffic_split
        )
        return {"status": "success", "message": f"Experiment {request.test_id} created"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/v1/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest) -> PredictionResponse:
    """
    Unified prediction endpoint with A/B testing support.
    """
    start_time = time.perf_counter()
    variant_info = None

    try:
        model_path = request.model_path
        engine = request.engine

        # A/B Testing Logic
        if request.ab_test_id:
            variant = AB_MANAGER.get_variant(request.ab_test_id, request.session_id)
            model_path = variant["model_path"]
            engine = variant.get("engine", engine)
            variant_info = variant

        if not model_path:
            raise HTTPException(
                status_code=400,
                detail="model_path is required if ab_test_id is not set",
            )

        model, metadata = get_model(model_path, request.task, engine)

        if request.task == "time_series":
            # Basic preprocessing (convert to tensor)
            x = torch.tensor(request.inputs, dtype=torch.float32)
            if x.dim() == 1:
                x = x.unsqueeze(0).unsqueeze(-1)  # (1, seq, 1)
            elif x.dim() == 2:
                x = x.unsqueeze(-1)  # (batch, seq, 1)

            with torch.no_grad():
                output = model(x)

            result = output.cpu().numpy().tolist()
        elif request.task == "text" and engine == "vllm":
            # request.inputs should be a list of strings
            result = model.generate(request.inputs)
        else:
            raise HTTPException(status_code=400, detail="Unsupported task or engine")

        latency = (time.perf_counter() - start_time) * 1000

        return PredictionResponse(
            status="success",
            prediction=result,
            latency_ms=round(latency, 2),
            model_metadata=metadata.to_dict(),
            variant_info=variant_info,
        )

    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
