from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import torch
import sys
import os
from loguru import logger
from src.routes import exr, models, health, gpu, batch
from src.utils.gpu_manager import GPUManager, GPUMemoryError

# Add the project root to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

# Enable OpenEXR support in OpenCV
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

# Monkeypatch nuke.tprint, nuke.ProgressTask, and common stubs for environments without real Nuke
try:
    import nuke
    if not hasattr(nuke, "tprint"):
        def tprint(msg):
            print("[NUKE TPRINT]", msg)
        nuke.tprint = tprint
    if not hasattr(nuke, "ProgressTask"):
        class ProgressTask:
            def __init__(self, name, total_steps=None):
                self.name = name
                self.total_steps = total_steps
                self.current_step = 0
                self._cancelled = False
                print(f"[NUKE ProgressTask] Starting: {name}")
            def setProgress(self, step):
                self.current_step = step
                print(f"[NUKE ProgressTask] Progress: {step}")
            def setMessage(self, message):
                print(f"[NUKE ProgressTask] Message: {message}")
            def isCancelled(self):
                return self._cancelled
            def cancel(self):
                self._cancelled = True
                print(f"[NUKE ProgressTask] Cancelled: {self.name}")
        nuke.ProgressTask = ProgressTask
    # Patch other common Nuke stubs if needed
    if not hasattr(nuke, "message"):
        def message(msg):
            print(f"[NUKE MESSAGE] {msg}")
        nuke.message = message
    if not hasattr(nuke, "executeInMainThreadWithResult"):
        def executeInMainThreadWithResult(func, *args, **kwargs):
            return func(*args, **kwargs)
        nuke.executeInMainThreadWithResult = executeInMainThreadWithResult
    if not hasattr(nuke, "executeInMainThread"):
        def executeInMainThread(func, *args, **kwargs):
            return func(*args, **kwargs)
        nuke.executeInMainThread = executeInMainThread
except ImportError:
    pass

app = FastAPI(title="Nuke Samurai API", version="0.1.0")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(exr.router, prefix="/api/v1", tags=["exr"])
app.include_router(models.router, prefix="/api/v1/models", tags=["models"])
app.include_router(health.router, prefix="/api/v1", tags=["health"])
app.include_router(gpu.router, prefix="/api/v1/gpu", tags=["gpu"])
app.include_router(batch.router, prefix="/api/v1", tags=["batch"])

# Add GPU memory middleware
@app.middleware("http")
async def gpu_memory_middleware(request: Request, call_next):
    try:
        if torch.cuda.is_available():
            gpu_manager = GPUManager()
            await gpu_manager.ensure_memory()
        return await call_next(request)
    except GPUMemoryError as e:
        return JSONResponse(
            status_code=503,
            content={"error": str(e)},
            headers={"Retry-After": "5"}
        )

# Error handlers
@app.exception_handler(GPUMemoryError)
async def gpu_memory_error_handler(request: Request, exc: GPUMemoryError):
    return JSONResponse(
        status_code=503,
        content={"error": str(exc)},
        headers={"Retry-After": "5"}
    )

@app.on_event("startup")
async def startup_event():
    """Initialize on startup."""
    # Configure logging
    logger.add(
        "logs/api.log",
        rotation="500 MB",
        retention="10 days",
        level="INFO"
    )
    
    # Initialize GPU
    if torch.cuda.is_available():
        logger.info(f"Using GPU: {torch.cuda.get_device_name()}")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    else:
        logger.warning("No GPU available, running in CPU mode")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
