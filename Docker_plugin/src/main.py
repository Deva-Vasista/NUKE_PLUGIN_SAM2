from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware  # noqa: F401
from fastapi.middleware.cors import CORSMiddleware
import torch
import numpy as np
import OpenEXR
import Imath
from pathlib import Path
import tempfile
import shutil
import os
from typing import List, Optional
import cv2
from loguru import logger as guru
from src.routes import exr, models, health
import sys

# Add the project root to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

# Import the nuke patch before any NukeSamurai imports
from src.utils.nuke_patch import MockNuke

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
app.include_router(models.router, prefix="/api/v1", tags=["models"])
app.include_router(health.router, prefix="/api/v1", tags=["health"])

print("Starting Server at port 8000")

# Add GPU memory middleware
@app.middleware("http")
async def gpu_memory_middleware(request: Request, call_next):
    if torch.cuda.is_available():
        free_mem = torch.cuda.mem_get_info()[0]
        if free_mem < 2 * 1024**3:  # 2GB threshold
            return JSONResponse(
                status_code=503,
                content={"error": "Insufficient GPU memory"},
                headers={"Retry-After": "5"}
            )
    return await call_next(request)

@app.on_event("startup")
async def startup_event():
    # Preload default model using the singleton instance
    from src.models.sam_wrapper import SAMProcessor
    processor = SAMProcessor()  # This will return the singleton instance
    processor.load_model("NukeSamurai/sam2_repo/checkpoints/sam2.1_hiera_small.pt")

# GPU Management
def setup_gpu():
    """Initialize GPU settings"""
    if not torch.cuda.is_available():
        guru.warning("No GPU available, running in CPU mode")
        return False
    
    # Disable TF32 settings
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    
    # Set device properties
    try:
        gpu_id = 0
        guru.info(f"Using GPU {gpu_id} for processing")
        return True
    except Exception as e:
        guru.error(f"Error initializing GPU: {e}")
        return False

# EXR Processing
def read_exr(exr_path: str) -> np.ndarray:
    """Read EXR file and return normalized numpy array"""
    try:
        exr_file = OpenEXR.InputFile(exr_path)
        dw = exr_file.header()["dataWindow"]
        size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)
        
        # Extract RGB channels
        channels = ["R", "G", "B"]
        exr_channels = []
        for c in channels:
            channel_data = np.frombuffer(exr_file.channel(c), dtype=np.float32)
            channel_data = channel_data.reshape(size[1], size[0])
            # Normalize each channel
            if channel_data.max() > channel_data.min():
                channel_data = (channel_data - channel_data.min()) / (channel_data.max() - channel_data.min())
            exr_channels.append(channel_data)
        
        return np.stack(exr_channels, axis=-1)
    except Exception as e:
        guru.error(f"Error reading EXR: {e}")
        raise HTTPException(status_code=500, detail=f"Error reading EXR: {str(e)}")

def save_exr(image: np.ndarray, mask: np.ndarray, output_path: str):
    """Save image and mask as EXR file"""
    try:
        # Create header
        header = OpenEXR.Header(image.shape[1], image.shape[0])
        header["channels"] = {
            "R": Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT)),
            "G": Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT)),
            "B": Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT)),
            "Mask": Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT)),
        }
        
        # Create EXR file
        exr_file = OpenEXR.OutputFile(output_path, header)
        
        # Write channels
        exr_file.writePixels({
            "R": image[:, :, 0].tobytes(),
            "G": image[:, :, 1].tobytes(),
            "B": image[:, :, 2].tobytes(),
            "Mask": mask.tobytes(),
        })
        exr_file.close()
    except Exception as e:
        guru.error(f"Error saving EXR: {e}")
        raise HTTPException(status_code=500, detail=f"Error saving EXR: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "gpu_available": torch.cuda.is_available(),
        "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0
    }

if __name__ == "__main__":
    import uvicorn
    setup_gpu()
    uvicorn.run(app, host="0.0.0.0", port=8000)
