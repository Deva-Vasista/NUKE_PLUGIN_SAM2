from fastapi import APIRouter
import torch
from src.utils.gpu_manager import GPUManager

router = APIRouter()
gpu_manager = GPUManager()

@router.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "gpu_available": torch.cuda.is_available(),
        "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "gpu_info": gpu_manager.get_device_info() if torch.cuda.is_available() else None
    } 