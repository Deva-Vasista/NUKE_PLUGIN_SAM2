# NukeSamurai Backend API Reference

## Overview
This backend exposes endpoints for EXR image and sequence segmentation/tracking using SAM2. It supports bounding box and point prompts, multi-object, and multi-frame workflows. All endpoints return robust error messages and support both file and JSON outputs.

---

## 1. `/api/v1/process_exr`
**POST** — Process a single EXR file with bbox and/or point prompts.

### **Request (application/json):**
- `image_path`: (str, required) — Path to EXR file on disk (must be accessible to backend).
- `bbox`: (list, optional) — `[x1, y1, x2, y2]`.
- `points_positive`: (list, optional) — List of `[x, y]` points to select.
- `points_negative`: (list, optional) — List of `[x, y]` points to remove.
- `bits`: (str, optional) — Bit depth, default: `32-bit float`.
- `as_file`: (bool, query param, optional) — If true, returns EXR mask file; else, returns mask as JSON array.

### **Example JSON:**
```json
{
  "image_path": "/path/to/frame_0001.exr",
  "bbox": [454, 185, 500, 633],
  "points_positive": [[610,728]],
  "bits": "32-bit float"
}
```

### **Example curl:**
```bash
curl -X POST "http://localhost:8000/api/v1/process_exr?as_file=true" \
  -H "Content-Type: application/json" \
  -d '{
    "image_path": "/path/to/frame_0001.exr",
    "bbox": [454, 185, 500, 633],
    "bits": "32-bit float"
  }' \
  --output mask.exr
```

### **Response:**
- If `as_file=true`: EXR mask file.
- Else: `{ "result": [[...mask array...]] }`

### **Common Problems & Solutions:**
- **File not found:** Ensure `image_path` is correct and readable by backend.
- **No prompt provided:** Must provide at least one of `bbox` or `points_positive`/`points_negative`.
- **Failed to read image:** File is not a valid EXR.
- **Internal error:** See backend logs for details.
- **Do not use multipart/form-data:** Only JSON is accepted for this endpoint.

---

## 2. `/api/v1/process_sequence`
**POST** — Process a sequence of EXR files with multi-object, multi-frame prompts.

### **Request (application/json):**
- `sequence_path`: (str, required) — Path pattern to EXR sequence (e.g., `/path/to/frames/frame_%04d.exr`).
- `frame_range`: (list, required) — `[start, end]` (inclusive start, exclusive end).
- `bits`: (str, optional) — Bit depth, default: `32-bit float`.
- `reverse`: (bool, optional) — Whether to track in reverse time order, default: `false`.
- `prompts`: (list, required) — List of prompt dicts:
  - `frame_index`: (int, required) — Frame index (0-based).
  - `object_id`: (int, optional) — Object ID (defaults to 0 if omitted).
  - `points_positive`: (list, optional) — List of `[x, y]` points to select.
  - `points_negative`: (list, optional) — List of `[x, y]` points to remove.
  - `bbox`: (list, optional) — `[x1, y1, x2, y2]`.
- `as_file`: (bool, query param, optional) — If true, returns ZIP of EXR masks; else, returns JSON.

### **Example curl:**
```bash
curl -X POST "http://localhost:8000/api/v1/process_sequence?as_file=true" \
  -H "Content-Type: application/json" \
  -d '{
    "sequence_path": "/path/to/frames/frame_%04d.exr",
    "frame_range": [1, 20],
    "reverse": false,
    "prompts": [
      { "frame_index": 0, "bbox": [454, 185, 500, 633] },
      { "frame_index": 10, "object_id": 1, "points_positive": [[1689, 216]] }
    ],
    "bits": "32-bit float"
  }' \
  --output masks.zip
```

### **Response:**
- If `as_file=true`: ZIP file with EXR masks named `mask_{frame_idx:04d}.exr`.
- Else: `{ "result": [ {frame_idx: {object_id: mask_array, ...}, ... } ] }`

### **Common Problems & Solutions:**
- **No prompt provided:** Must provide at least one prompt (bbox or points).
- **File(s) not found:** Check `sequence_path` and frame numbers.
- **FPS errors:** Ensure backend sets `original_fps` and `target_fps` (defaults to 24).
- **Internal error:** See backend logs for details.

---

## 3. `/api/v1/models/load`
**POST** — Load a specific SAM model.

### **Request (application/json):**
- `model_type`: (str, required) — One of: `large`, `base`, `small`, `tiny`.

### **Example curl:**
```bash
curl -X POST "http://localhost:8000/api/v1/models/load" \
  -H "Content-Type: application/json" \
  -d '{ "model_type": "large" }'
```

### **Response:**
- `{ "status": "success", "model_loaded": "large", "vram_usage": ... }`

### **Common Problems & Solutions:**
- **Model file not found:** Ensure the correct model weights are present in the checkpoints directory.
- **Invalid model type:** Must be one of the allowed values.

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