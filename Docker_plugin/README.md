# NukeSamurai Backend

NukeSamurai Backend is a FastAPI server that brings Meta's Segment Anything Model 2 (SAM2) to VFX workflows, enabling high-quality, automated rotoscoping and object segmentation for EXR image sequences. It is designed to be used with the NukeSamurai plugin for NUKE, but can be accessed by any client via HTTP APIs.

---

## Features
- **EXR sequence segmentation and tracking** using SAM2
- **REST API** for easy integration with Nuke or other tools
- **GPU acceleration** (CUDA) and CPU fallback
- **Progress tracking** via WebSocket or polling
#- **Batch and single-frame processing**
- **Dockerized for easy deployment**

---

## API Endpoints (Summary)

All endpoints are under `/api/v1` unless otherwise noted.

### Core Endpoints
- `POST   /process_exr`         — Segment a single EXR image (bbox/points)
- `POST   /process_sequence`    — Track/segment across a sequence (multi-frame, multi-object)
- `POST   /batch/process_batch` — Batch process multiple sequences
- `POST   /reset`               — Reset model state and clear GPU memory
- `GET    /output/{task_id}`    — Get output files for a task
- `GET    /download/{filename}` — Download a result file
- `GET    /download_all/{task_id}` — Download all results for a task
- `GET    /status/{task_id}`    — Get status of a processing task
- `WS     /ws/{task_id}`        — WebSocket for progress updates

### Model Management
- `POST   /models/load`         — Load a specific SAM2 model (large, base-plus, small, tiny)
- `GET    /models/status`       — Get current model status
- `POST   /models/unload`       — Unload the current model

### GPU & Health
- `GET    /health`              — Health check
- `GET    /gpu/info`            — GPU info
- `GET    /gpu/memory`          — GPU memory stats
- `POST   /gpu/clear-cache`     — Clear GPU memory cache

For detailed request/response formats, see [Wikis/API.md](../Wikis/API.md).

---

## Quickstart: Docker Deployment

### 1. Prerequisites
- [Docker](https://docs.docker.com/get-docker/) and [Docker Compose](https://docs.docker.com/compose/install/)
- (Optional) NVIDIA GPU and [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) for GPU acceleration

### 2. Build and Run (GPU)
```bash
cd Docker_plugin
# Build and start the API (GPU version)
docker-compose up --build
```
- The API will be available at `http://localhost:8000`
- EXR data and output directories are mounted via volumes (see `docker-compose.yml`)

### 3. Build and Run (CPU-only)
```bash
cd Docker_plugin
docker-compose --profile cpu up --build
```
- The API will be available at `http://localhost:8001`

### 4. Customizing Paths
- To use custom data/output/checkpoints, set environment variables:
  - `EXR_DATA_PATH`, `OUTPUT_PATH`, `CHECKPOINTS_PATH`
- Example:
  ```bash
  EXR_DATA_PATH=/my/exr INPUT_PATH OUTPUT_PATH=/my/output docker-compose up --build
  ```

---

## Manual Setup (Without Docker)

### 1. Prerequisites
- Python 3.8+
- CUDA-capable GPU (recommended) or CPU
- pip, git

### 2. Install dependencies and package
>[!NOTE]
>This Application requires a python version 3.10
```bash
cd Docker_plugin
python3.10 -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install --upgrade pip setuptools wheel
pip install -e .
```
- This will:
  - Check Python version
  - Install all dependencies
  - Set up the SAM2 repo and config
  - Register CLI tools: `nuke-samurai-server`, `nuke-samurai-test`

### 3. Run the API server
```bash
nuke-samurai-server
# or with custom options:
nuke-samurai-server --host 0.0.0.0 --port 8000 --log-level info
```

or 

```bash
uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload
```

### 4. Verify
```bash
curl http://localhost:8000/api/v1/health
```

---

## About the Project
- **Backend:** FastAPI, PyTorch, OpenCV, SAM2
- **Frontend:** Nuke plugin (see Nuke_plugin/)
- **Docs:** See [Wikis/](../Wikis/) for API, architecture, and usage details

---

## Troubleshooting
- **CUDA errors:** Check your GPU drivers and CUDA install, or run in CPU mode
- **Missing checkpoints:** Ensure all required `.pt` files are in `NukeSamurai/sam2_repo/checkpoints/`
- **Permissions:** Make sure output directories are writable
- **See logs:** Check `logs/api.log` for backend errors

---

## License
MIT License — see LICENSE for details 