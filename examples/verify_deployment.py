"""Verification script for Phase 7: Deployment & Serving."""

import torch
import torch.nn as nn
from fastapi.testclient import TestClient
from src.api.server import app
from src.utils.export.onnx_exporter import ONNXExporter
from src.models.time_series import TimeSeriesBackbone
from pathlib import Path


def test_server():
    print("Testing Inference Server...")
    client = TestClient(app)

    # Test health
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

    # Test models list
    response = client.get("/models")
    assert response.status_code == 200
    assert "loaded_models" in response.json()

    print("Server tests passed!")


def test_onnx_export():
    print("\nTesting ONNX Export...")
    config = {
        "name": "LSTM",
        "feature_dim": 1,
        "hidden_dim": 32,
        "output_dim": 1,
        "num_layers": 2,
    }
    model = TimeSeriesBackbone(config)
    output_path = "outputs/test_model.onnx"

    ONNXExporter.export_time_series(model, output_path, dynamic_axes={})

    onnx_file = Path(output_path)
    print(f"ONNX file created: {onnx_file.exists()}")
    assert onnx_file.exists()
    assert onnx_file.stat().st_size > 0

    # Cleanup
    # onnx_file.unlink()
    print("ONNX export tests passed!")


if __name__ == "__main__":
    try:
        # 1. Test Server (Logic only as we need real models for full flow)
        test_server()

        # 2. Test ONNX Export
        test_onnx_export()

        print("\nAll Phase 7 tests passed!")
    except Exception as e:
        print(f"\nTests failed: {e}")
        import traceback

        traceback.print_exc()
        exit(1)
