import pytest
from fastapi.testclient import TestClient
import json
from pathlib import Path
import asyncio
import websockets
import numpy as np

def test_health_check(client):
    """Test the health check endpoint."""
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_process_single_exr(client, sample_exr):
    """Test processing a single EXR file."""
    with open(sample_exr, "rb") as f:
        files = {"image": ("test.exr", f, "application/octet-stream")}
        response = client.post(
            "/api/v1/process_exr",
            files=files,
            data={"bbox": "0,0,50,50"}
        )
    assert response.status_code == 200
    assert "result" in response.json()

def test_process_sequence(client, sample_sequence):
    """Test processing an EXR sequence."""
    request_data = {
        "sequence_path": str(sample_sequence),
        "frame_range": [1, 3],
        "bbox": "0,0,50,50",
        "original_fps": 24,
        "target_fps": 24,
        "bits": 32
    }

    response = client.post(
        "/api/v1/process_sequence",
        json=request_data
    )
    assert response.status_code == 200
    assert "result" in response.json()
    assert "task_id" in response.json()

def test_invalid_bbox(client):
    """Test invalid bbox handling."""
    response = client.post(
        "/api/v1/process_exr",
        files={"image": ("test.exr", b"dummy", "application/octet-stream")},
        data={"bbox": "invalid"}
    )
    assert response.status_code == 400

def test_missing_file(client):
    """Test missing file handling."""
    response = client.post(
        "/api/v1/process_exr",
        files={},
        data={"bbox": "0,0,50,50"}
    )
    assert response.status_code == 422

@pytest.mark.asyncio
async def test_progress_websocket(client, sample_sequence):
    """Test WebSocket progress updates."""
    # Start processing in background
    request_data = {
        "sequence_path": str(sample_sequence),
        "frame_range": [1, 3],
        "bbox": "0,0,50,50"
    }

    response = client.post(
        "/api/v1/process_sequence",
        json=request_data
    )
    assert response.status_code == 200
    task_id = response.json()["task_id"]

    # Connect to WebSocket
    with client.websocket_connect(f"/api/v1/ws/{task_id}") as websocket:
        data = websocket.receive_json()
        assert "progress" in data
        assert "status" in data

def test_gpu_memory_info(client):
    """Test GPU memory info endpoint."""
    response = client.get("/api/v1/gpu/info")
    assert response.status_code == 200
    assert "memory_stats" in response.json()
    assert "device_info" in response.json()

def test_model_loading(client):
    """Test model loading endpoint."""
    response = client.post(
        "/api/v1/models/load",
        json={"model_type": "small"}
    )
    assert response.status_code == 200
    assert "status" in response.json()
    assert response.json()["status"] == "success"

def test_error_handling(client, tmp_path):
    """Test error handling."""
    # Test missing file (should return 404)
    request_data = {
        "sequence_path": "nonexistent.exr",
        "frame_range": [1, 3],
        "bbox": "0,0,50,50"
    }
    response = client.post(
        "/api/v1/process_sequence",
        json=request_data
    )
    assert response.status_code == 404

    # Test invalid frame range (should return 400)
    # Create a dummy file
    dummy_file = tmp_path / "dummy.exr"
    dummy_file.write_bytes(b"dummy")
    request_data = {
        "sequence_path": str(dummy_file),
        "frame_range": [-1, 100],
        "bbox": "0,0,50,50"
    }
    response = client.post(
        "/api/v1/process_sequence",
        json=request_data
    )
    assert response.status_code == 400

@pytest.mark.asyncio
async def test_batch_processing(client, sample_sequence):
    """Test batch processing endpoint."""
    request_data = {
        "sequences": [
            {
                "path": str(sample_sequence),
                "frame_range": [1, 3],
                "bbox": "0,0,50,50"
            },
            {
                "path": str(sample_sequence),
                "frame_range": [4, 6],
                "bbox": "10,10,60,60"
            }
        ]
    }

    response = client.post(
        "/api/v1/process_batch",
        json=request_data
    )
    assert response.status_code == 200
    assert "batch_id" in response.json()
    assert "task_ids" in response.json()

def test_process_exr_points_positive(client, sample_exr):
    """Test processing a single EXR file with positive points only."""
    with open(sample_exr, "rb") as f:
        files = {"image": ("test.exr", f, "application/octet-stream")}
        data = {"points_positive": json.dumps([[10, 10], [20, 20]])}
        response = client.post(
            "/api/v1/process_exr",
            files=files,
            data=data
        )
    assert response.status_code == 200
    assert "result" in response.json()

def test_process_exr_points_negative(client, sample_exr):
    """Test processing a single EXR file with negative points only."""
    with open(sample_exr, "rb") as f:
        files = {"image": ("test.exr", f, "application/octet-stream")}
        data = {"points_negative": json.dumps([[30, 30], [40, 40]])}
        response = client.post(
            "/api/v1/process_exr",
            files=files,
            data=data
        )
    assert response.status_code == 200
    assert "result" in response.json()

def test_process_exr_bbox_and_points(client, sample_exr):
    """Test processing a single EXR file with bbox and both positive/negative points."""
    with open(sample_exr, "rb") as f:
        files = {"image": ("test.exr", f, "application/octet-stream")}
        data = {
            "bbox": "0,0,50,50",
            "points_positive": json.dumps([[10, 10]]),
            "points_negative": json.dumps([[30, 30]])
        }
        response = client.post(
            "/api/v1/process_exr",
            files=files,
            data=data
        )
    assert response.status_code == 200
    assert "result" in response.json()

def test_process_sequence_points_positive(client, sample_sequence):
    """Test processing an EXR sequence with positive points only."""
    request_data = {
        "sequence_path": str(sample_sequence),
        "frame_range": [1, 3],
        "points_positive": json.dumps([[10, 10], [20, 20]])
    }
    response = client.post(
        "/api/v1/process_sequence",
        json=request_data
    )
    assert response.status_code == 200
    assert "result" in response.json() or "task_id" in response.json()

def test_process_sequence_bbox_and_points(client, sample_sequence):
    """Test processing an EXR sequence with bbox and both positive/negative points."""
    request_data = {
        "sequence_path": str(sample_sequence),
        "frame_range": [1, 3],
        "bbox": "0,0,50,50",
        "points_positive": json.dumps([[10, 10]]),
        "points_negative": json.dumps([[30, 30]])
    }
    response = client.post(
        "/api/v1/process_sequence",
        json=request_data
    )
    assert response.status_code == 200
    assert "result" in response.json() or "task_id" in response.json()

def test_process_sequence_tracking_bbox(client, sample_sequence):
    """Test sequence tracking with bbox only."""
    request_data = {
        "sequence_path": str(sample_sequence),
        "frame_range": [1, 3],
        "bbox": "0,0,50,50"
    }
    response = client.post(
        "/api/v1/process_sequence",
        json=request_data
    )
    assert response.status_code == 200
    assert "result" in response.json() or "task_id" in response.json()

def test_process_sequence_tracking_points_positive(client, sample_sequence):
    """Test sequence tracking with positive points only."""
    request_data = {
        "sequence_path": str(sample_sequence),
        "frame_range": [1, 3],
        "points_positive": json.dumps([[10, 10], [20, 20]])
    }
    response = client.post(
        "/api/v1/process_sequence",
        json=request_data
    )
    assert response.status_code == 200
    assert "result" in response.json() or "task_id" in response.json()

def test_process_sequence_tracking_points_negative(client, sample_sequence):
    """Test sequence tracking with negative points only."""
    request_data = {
        "sequence_path": str(sample_sequence),
        "frame_range": [1, 3],
        "points_negative": json.dumps([[30, 30], [40, 40]])
    }
    response = client.post(
        "/api/v1/process_sequence",
        json=request_data
    )
    assert response.status_code == 200
    assert "result" in response.json() or "task_id" in response.json()

def test_process_sequence_tracking_bbox_and_points(client, sample_sequence):
    """Test sequence tracking with bbox and both positive/negative points."""
    request_data = {
        "sequence_path": str(sample_sequence),
        "frame_range": [1, 3],
        "bbox": "0,0,50,50",
        "points_positive": json.dumps([[10, 10]]),
        "points_negative": json.dumps([[30, 30]])
    }
    response = client.post(
        "/api/v1/process_sequence",
        json=request_data
    )
    assert response.status_code == 200
    assert "result" in response.json() or "task_id" in response.json() 