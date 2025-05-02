from fastapi import APIRouter
from src.utils.gpu_manager import GPUManager

router = APIRouter()
gpu_manager = GPUManager()

@router.get("/info")
async def get_gpu_info():
    """Get GPU memory and device information."""
    return {
        "memory_stats": gpu_manager.get_memory_stats(),
        "device_info": gpu_manager.get_device_info()
    }

@router.get("/memory")
async def get_memory_stats():
    """Get detailed GPU memory statistics."""
    return gpu_manager.get_memory_stats()

@router.post("/clear-cache")
async def clear_gpu_cache():
    """Clear GPU memory cache."""
    gpu_manager.clear_cache()
    return {"status": "success", "message": "GPU cache cleared"} 