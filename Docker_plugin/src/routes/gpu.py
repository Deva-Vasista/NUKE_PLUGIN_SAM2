from fastapi import APIRouter, HTTPException
from src.utils.gpu_manager import GPUManager
from loguru import logger

router = APIRouter()
gpu_manager = GPUManager()

@router.get("/info")
async def get_gpu_info():
    """Get GPU memory and device information."""
    return {
        "memory_stats": gpu_manager.get_memory_stats(),
        "device_info": gpu_manager.get_device_info()
    }

@router.get("/memory-stats")
async def get_memory_stats():
    """Get current GPU memory statistics."""
    try:
        stats = gpu_manager.get_memory_stats()
        return {
            "status": "success",
            "memory_stats": stats,
            "recommendation": "Use /api/v1/reset for basic cleanup or /api/v1/models/unload for complete cleanup"
        }
    except Exception as e:
        logger.error(f"Error getting memory stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/device-info")
async def get_device_info():
    """Get GPU device information."""
    try:
        info = gpu_manager.get_device_info()
        return {
            "status": "success",
            "device_info": info
        }
    except Exception as e:
        logger.error(f"Error getting device info: {e}")
        raise HTTPException(status_code=500, detail=str(e)) 