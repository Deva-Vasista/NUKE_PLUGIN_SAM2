import torch
import gc
from typing import Optional
from loguru import logger

class GPUMemoryError(Exception):
    """Raised when there's insufficient GPU memory."""
    pass

class GPUManager:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(GPUManager, cls).__new__(cls)
            cls._instance.initialized = False
        return cls._instance
    
    def __init__(self):
        if self.initialized:
            return
            
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.min_free_memory = 2 * 1024**3  # 2GB minimum free memory
        self.initialized = True
        self.last_cleanup_time = 0
        self.cleanup_cooldown = 60  # seconds between cleanups
    
    def get_free_memory(self) -> int:
        """Get free GPU memory in bytes."""
        if not torch.cuda.is_available():
            return 0
        return torch.cuda.mem_get_info()[0]
    
    def get_total_memory(self) -> int:
        """Get total GPU memory in bytes."""
        if not torch.cuda.is_available():
            return 0
        return torch.cuda.mem_get_info()[1]
    
    def get_memory_usage(self) -> float:
        """Get memory usage as a percentage."""
        if not torch.cuda.is_available():
            return 0.0
        free, total = torch.cuda.mem_get_info()
        return (total - free) / total * 100
    
    async def ensure_memory(self, required_bytes: Optional[int] = None) -> bool:
        """Ensure there's enough GPU memory available."""
        if not torch.cuda.is_available():
            return True
            
        required = required_bytes if required_bytes else self.min_free_memory
        free_memory = self.get_free_memory()
        
        if free_memory >= required:
            return True
            
        # Try to free memory
        logger.info("Attempting to free GPU memory...")
        self.clear_cache()
        
        # Check again
        free_memory = self.get_free_memory()
        if free_memory >= required:
            return True
            
        raise GPUMemoryError(
            f"Insufficient GPU memory. Need {required/1024**3:.2f}GB, "
            f"but only {free_memory/1024**3:.2f}GB available"
        )
    
    def clear_cache(self):
        """Clear GPU cache and garbage collect."""
        if not torch.cuda.is_available():
            return
            
        # Log memory stats before cleanup
        stats_before = self.get_memory_stats()
        logger.info(f"Memory before cleanup: {stats_before}")
        
        # Clear PyTorch cache
        torch.cuda.empty_cache()
        
        # Clear CUDA cache
        if hasattr(torch.cuda, 'memory_summary'):
            torch.cuda.memory_summary(device=None, abbreviated=True)
        
        # Force garbage collection
        gc.collect()
        
        # Clear any cached tensors
        for obj in gc.get_objects():
            try:
                if torch.is_tensor(obj):
                    if obj.device.type == 'cuda':
                        del obj
            except:
                pass
        
        # Force another garbage collection
        gc.collect()
        torch.cuda.empty_cache()
        
        # Log memory stats after cleanup
        stats_after = self.get_memory_stats()
        logger.info(f"Memory after cleanup: {stats_after}")
        
        # Calculate memory freed
        freed_gb = (stats_before['used_gb'] - stats_after['used_gb'])
        logger.info(f"Freed {freed_gb:.2f}GB of GPU memory")
    
    def get_memory_stats(self) -> dict:
        """Get detailed memory statistics."""
        if not torch.cuda.is_available():
            return {"status": "No GPU available"}
            
        free, total = torch.cuda.mem_get_info()
        used = total - free
        
        return {
            "total_gb": total / 1024**3,
            "used_gb": used / 1024**3,
            "free_gb": free / 1024**3,
            "usage_percent": (used / total) * 100,
            "max_allocated_gb": torch.cuda.max_memory_allocated() / 1024**3,
            "max_reserved_gb": torch.cuda.max_memory_reserved() / 1024**3
        }
    
    def is_gpu_available(self) -> bool:
        """Check if GPU is available."""
        return torch.cuda.is_available()
    
    def get_device_info(self) -> dict:
        """Get GPU device information."""
        if not torch.cuda.is_available():
            return {"status": "No GPU available"}
            
        return {
            "name": torch.cuda.get_device_name(),
            "capability": torch.cuda.get_device_capability(),
            "current_device": torch.cuda.current_device(),
            "device_count": torch.cuda.device_count()
        } 