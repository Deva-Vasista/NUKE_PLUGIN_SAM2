# NukeSamurai Backend API Reference

## Overview
This backend exposes endpoints for EXR image and sequence segmentation/tracking using SAM2. It supports bounding box and point prompts, multi-object, and multi-frame workflows. All endpoints return robust error messages and support both file and JSON outputs.

## Base URL
```
http://localhost:8000/api/v1
```

## Endpoints

### 1. Load Model
```
POST /models/load
```

Load a specific model type into memory.

Request Body:
```json
{
    "model_type": "base"  // One of: "base", "large", "small", "tiny"
}
```

Response:
```json
{
    "status": "success",
    "message": "Model loaded successfully",
    "model_type": "base"
}
```

Example:
```bash
curl -X POST "http://localhost:8000/api/v1/models/load" \
  -H "Content-Type: application/json" \
  -d '{
    "model_type": "base"
  }'
```

### 2. Process Single Frame
```
POST /process_exr
```

Process a single EXR frame with bounding box and/or point prompts. The model will generate a mask for the selected object.

Query Parameters:
- `as_file` (boolean): Return mask as EXR file if true, else as list

Request Body:
```json
{
    "image_path": "/path/to/frame.exr",
    "bbox": [x1, y1, x2, y2],  // Optional: Bounding box coordinates
    "points_positive": [[x1,y1], [x2,y2]],  // Optional: Points to select
    "points_negative": [[x1,y1], [x2,y2]],  // Optional: Points to remove
    "bits": "32-bit float"  // Optional, defaults to "32-bit float"
}
```

Example with bbox:
```bash
curl -X POST "http://localhost:8000/api/v1/process_exr?as_file=true" \
  -H "Content-Type: application/json" \
  -d '{
    "image_path": "/path/to/frame.exr",
    "bbox": [454, 185, 500, 633],
    "bits": "32-bit float"
  }' \
  --output mask.exr
```

Example with points:
```bash
curl -X POST "http://localhost:8000/api/v1/process_exr?as_file=true" \
  -H "Content-Type: application/json" \
  -d '{
    "image_path": "/path/to/frame.exr",
    "points_positive": [[610, 728], [612, 730]],
    "points_negative": [[500, 500]],
    "bits": "32-bit float"
  }' \
  --output mask.exr
```

Example with both bbox and points:
```bash
curl -X POST "http://localhost:8000/api/v1/process_exr?as_file=true" \
  -H "Content-Type: application/json" \
  -d '{
    "image_path": "/path/to/frame.exr",
    "bbox": [454, 185, 500, 633],
    "points_positive": [[610, 728]],
    "points_negative": [[500, 500]],
    "bits": "32-bit float"
  }' \
  --output mask.exr
```

### 3. Process Sequence
```
POST /process_sequence
```

Process a sequence of EXR files with multi-object tracking. The model will track objects across frames and generate masks for each frame.

Query Parameters:
- `as_file` (boolean): Return masks as ZIP if true, else as JSON list

Request Body:
```json
{
    "sequence_path": "/path/to/frame_%04d.exr",
    "frame_range": [start_frame, end_frame],
    "prompts": [
        {
            "frame_index": 0,
            "object_id": 0,
            "bbox": [x1, y1, x2, y2],  // Optional
            "points_positive": [[x1,y1]],  // Optional
            "points_negative": [[x1,y1]]  // Optional
        }
    ],
    "bits": "32-bit float"  // Optional, defaults to "32-bit float"
}
```

Example with single object:
```bash
curl -X POST "http://localhost:8000/api/v1/process_sequence?as_file=true" \
  -H "Content-Type: application/json" \
  -d '{
    "sequence_path": "/path/to/frames/frame_%04d.exr",
    "frame_range": [1, 20],
    "prompts": [
      { "frame_index": 0, "bbox": [454, 185, 500, 633] }
    ],
    "bits": "32-bit float"
  }' \
  --output masks.zip
```

Example with multiple objects:
```bash
curl -X POST "http://localhost:8000/api/v1/process_sequence?as_file=true" \
  -H "Content-Type: application/json" \
  -d '{
    "sequence_path": "/path/to/frames/frame_%04d.exr",
    "frame_range": [1, 20],
    "prompts": [
      { "frame_index": 0, "object_id": 0, "bbox": [454, 185, 500, 633] },
      { "frame_index": 10, "object_id": 1, "points_positive": [[1689, 216]] }
    ],
    "bits": "32-bit float"
  }' \
  --output masks.zip
```

Example with points and bbox:
```bash
curl -X POST "http://localhost:8000/api/v1/process_sequence?as_file=true" \
  -H "Content-Type: application/json" \
  -d '{
    "sequence_path": "/path/to/frames/frame_%04d.exr",
    "frame_range": [1, 20],
    "prompts": [
      { 
        "frame_index": 0, 
        "object_id": 0, 
        "bbox": [454, 185, 500, 633],
        "points_positive": [[610, 728]],
        "points_negative": [[500, 500]]
      }
    ],
    "bits": "32-bit float"
  }' \
  --output masks.zip
```

Response:
```json
{
    "status": "success",
    "task_id": "uuid-string",
    "output_dir": "Output/uuid-string",
    "zip_path": "Output/uuid-string/masks_uuid-string.zip",
    "file_count": 15,
    "frames": [0, 1, 2, ...]
}
```

### 4. Monitor Progress
```
WebSocket /ws/{task_id}
```

Connect to this WebSocket endpoint to receive progress updates:
```json
{
    "progress": 85,
    "message": "Processing frame 15/16",
    "status": "running"  // One of: "running", "completed", "failed"
}
```

Example using Python:
```python
import websockets
import asyncio
import json

async def monitor_progress(task_id):
    uri = f"ws://localhost:8000/api/v1/ws/{task_id}"
    async with websockets.connect(uri) as websocket:
        while True:
            response = await websocket.recv()
            status = json.loads(response)
            print(f"Progress: {status['progress']}% - {status['message']}")
            if status['status'] in ['completed', 'failed']:
                break
```

### 5. Get Task Output
```
GET /output/{task_id}
```

Response:
```json
{
    "files": [
        {
            "filename": "masks_uuid-string.zip",
            "size": 1234567,
            "created": "2024-05-15T12:34:56"
        }
    ]
}
```

### 6. Download Results
```
GET /download/{filename}
```

Returns the requested file (ZIP or EXR) as a binary response.

Example:
```bash
curl -X GET "http://localhost:8000/api/v1/download/masks_uuid-string.zip" \
  --output masks.zip
```

### 7. Download All Results
```
GET /download_all/{task_id}
```

Returns a ZIP file containing all output files for the task.

Example:
```bash
curl -X GET "http://localhost:8000/api/v1/download_all/uuid-string" \
  --output all_files.zip
```

### 8. Get Task Status
```
GET /status/{task_id}
```

Response:
```json
{
    "status": "completed",  // One of: "running", "completed", "failed"
    "progress": 100,
    "message": "Processing completed successfully",
    "error": "Error message if status is failed"  // Optional
}
```

### 9. Reset Model State
```
POST /reset
```

Reset the model state, clear GPU memory, and remove any cached data.

Response:
```json
{
    "status": "success",
    "message": "Model state reset successfully"
}
```

### 10. Health Check
```
GET /health
```

Check if the API server is running.

Response:
```json
{
    "status": "healthy",
    "version": "v1"
}
```

## Common Issues and Solutions

1. **File Not Found**
   - Ensure the image path is correct and accessible to the backend
   - For Windows paths, they will be automatically converted to WSL paths
   - Check file permissions

2. **No Prompt Provided**
   - Must provide at least one of: bbox, points_positive, or points_negative
   - For sequences, at least one prompt must be provided in the prompts list

3. **Invalid Coordinates**
   - Coordinates should be within the image dimensions
   - Bbox format: [x1, y1, x2, y2] where (x1,y1) is top-left and (x2,y2) is bottom-right
   - Points format: [[x1,y1], [x2,y2], ...]

4. **Memory Issues**
   - If you encounter GPU memory errors, try:
     - Using a smaller model (e.g., "small" instead of "large")
     - Processing smaller frame ranges
     - Resetting the model state between operations

5. **Output Format**
   - Single frame: Returns either an EXR file or a JSON array
   - Sequence: Returns either a ZIP of EXR files or a JSON object with frame indices
   - Use `as_file=true` to get file outputs, omit for JSON responses

---

## 4. `/api/v1/models/status`
**GET** — Get current model status.

### **Example curl:**
```bash
curl -X GET "http://localhost:8000/api/v1/models/status"
```

### **Response:**
- `{ "model_loaded": true/false, "model_path": ..., "device": ..., "vram_usage": ... }`

---

## 5. `/api/v1/models/unload`
**POST** — Unload the current model and clear GPU memory.

### **Example curl:**
```bash
curl -X POST "http://localhost:8000/api/v1/models/unload"
```

### **Response:**
- `{ "status": "success", "message": "Model unloaded" }`

---

## 6. `/api/v1/health`
**GET** — Health check endpoint.

### **Example curl:**
```bash
curl -X GET "http://localhost:8000/api/v1/health"
```

### **Response:**
- `{ "status": "healthy", "gpu_available": true/false, "gpu_count": ..., "gpu_info": ... }`

---

## 7. `/api/v1/gpu/info`
**GET** — Get GPU memory and device information.

### **Example curl:**
```bash
curl -X GET "http://localhost:8000/api/v1/gpu/info"
```

### **Response:**
- `{ "memory_stats": ..., "device_info": ... }`

---

## 8. `/api/v1/gpu/memory`
**GET** — Get detailed GPU memory statistics.

### **Example curl:**
```bash
curl -X GET "http://localhost:8000/api/v1/gpu/memory"
```

### **Response:**
- `{ ... }` (detailed memory stats)

---

## 9. `/api/v1/gpu/clear-cache`
**POST** — Clear GPU memory cache.

### **Example curl:**
```bash
curl -X POST "http://localhost:8000/api/v1/gpu/clear-cache"
```

### **Response:**
- `{ "status": "success", "message": "GPU cache cleared" }`

---

## 10. `/api/v1/batch/process_batch`
**POST** — Process multiple sequences in batch.

### **Request (application/json):**
- `sequences`: (list, required) — Each with:
  - `path`: (str, required) — Sequence path.
  - `frame_range`: (list, required) — `[start, end]`.
  - `bbox`: (str, required) — Comma-separated bbox string (e.g., "454,185,500,633").

### **Example curl:**
```bash
curl -X POST "http://localhost:8000/api/v1/batch/process_batch" \
  -H "Content-Type: application/json" \
  -d '{
    "sequences": [
      { "path": "/path/to/frames/frame_%04d.exr", "frame_range": [1, 5], "bbox": "454,185,500,633" }
    ]
  }'
```

### **Response:**
- `{ "batch_id": ..., "task_ids": [...], "status": "processing" }`

### **Common Problems & Solutions:**
- **File(s) not found:** Check `path` and frame numbers.
- **Invalid bbox:** Must be a comma-separated string of four numbers.

---

## 11. `/api/v1/test_form`
**POST** — Debug endpoint for form field parsing (for development/testing only).

### **Request (multipart/form-data):**
- `bbox`: (list, optional)
- `image`: (file, optional)

### **Example curl:**
```bash
curl -X POST "http://localhost:8000/api/v1/test_form" \
  -F "bbox=454" -F "bbox=185" -F "bbox=500" -F "bbox=633" \
  -F "image=@/path/to/frame_0001.exr"
```

### **Response:**
- `{ "bbox": [...] }`

---

## 12. WebSocket `/api/v1/ws/{task_id}`
- Used for progress tracking (if enabled in your workflow).
- Connect and receive JSON status updates for long-running tasks.

---

## Maintenance & Troubleshooting
- **Model/config mismatch:** Ensure you use compatible model weights and config files.
- **GPU/CPU issues:** Check logs for CUDA errors; fallback to CPU if needed.
- **File not found:** Double-check all file paths and frame numbers.
- **Prompt errors:** Always provide at least one prompt (bbox or points).
- **FPS errors:** Always ensure `original_fps` and `target_fps` are set (now defaulted to 24).
- **Debugging:** Use logs for detailed error messages. Most errors are logged with context.

---

## Updating/Extending
- To add new prompt types, extend the `prompts` schema in `/process_sequence`.
- To support new file formats, update `read_exr_to_numpy` and `write_exr` helpers.
- For new endpoints, follow the FastAPI pattern and update this documentation.

---

**For further help, check the backend logs or contact the maintainers.** 



## Setup
- Setup a virtual environment using uv, conda or venv
- `cd Docker_plugin` and Install dependencies using `pip install -r requirements.txt`
- `cd sam2_repo` and run the command `pip install -e .`, `pip install -e ".[notebooks]"`
- `cd sam2/checkpoints` and run `./download_checkpoints.sh`
- Go to the `Docker_plugin` directory
- run the uvicorn command `uvicorn src.main:app --reload --host 0.0.0.0 --port 8000` to start the server.
-  Please use the above API reference to make API requests