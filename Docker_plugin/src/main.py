from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import torch
import sys
import os
import json
from loguru import logger
from src.routes import exr, models, health, gpu, batch
from src.utils.gpu_manager import GPUManager, GPUMemoryError
from src.utils.json_utils import ensure_serializable

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

def _recursive_float_conversion(obj):
    """Recursively convert numeric values in dictionaries and lists to float32"""
    if isinstance(obj, dict):
        for key, value in obj.items():
            obj[key] = _recursive_float_conversion(value)
        return obj
    elif isinstance(obj, list):
        return [_recursive_float_conversion(item) for item in obj]
    elif isinstance(obj, (int, float)):
        return float(obj)
    else:
        return obj

# Add input validation middleware for numeric types
@app.middleware("http")
async def numeric_validation_middleware(request: Request, call_next):
    # Process the request body if it's a POST or PUT and has JSON content
    if request.method in ("POST", "PUT") and (
        "application/json" in request.headers.get("content-type", "")
    ):
        try:
            # Read the request body
            body = await request.body()
            if body:
                # Parse JSON content
                json_body = json.loads(body)
                
                # Convert numeric values to float
                json_body = _recursive_float_conversion(json_body)
                
                # Create a new request with the modified body
                # This is a bit of a hack since we can't modify the original request
                async def receive():
                    return {
                        "type": "http.request",
                        "body": json.dumps(json_body).encode(),
                        "more_body": False,
                    }
                
                # Preserve the original receive method
                original_receive = request.receive
                
                # Replace with our modified version for one call
                request.receive = receive
                
                # Process with the next middleware/endpoint
                response = await call_next(request)
                
                # Restore original receive for future middleware
                request.receive = original_receive
                
                return response
        except Exception as e:
            logger.error(f"Error in numeric validation middleware: {e}")
            # Continue with original request if there's an error processing
    
    # For all other requests, just pass through
    return await call_next(request)

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

# Add response serialization middleware
@app.middleware("http")
async def serialization_middleware(request: Request, call_next):
    response = await call_next(request)
    
    # If the response is a JSONResponse, ensure its content is serializable
    if hasattr(response, "body"):
        try:
            content_type = response.headers.get("content-type", "")
            if "application/json" in content_type:
                # Decode the JSON
                body = response.body.decode("utf-8")
                data = json.loads(body)
                
                # Ensure all values are serializable
                data = ensure_serializable(data)
                
                # Create a new response with the serializable data
                new_body = json.dumps(data).encode("utf-8")
                response.body = new_body
                response.headers["content-length"] = str(len(new_body))
        except Exception as e:
            logger.error(f"Error in serialization middleware: {e}")
    
    return response

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
    
    # Ensure Output directory exists with proper permissions
    output_dir = os.path.abspath("Output")
    os.makedirs(output_dir, exist_ok=True)
    
    # Ensure the Output directory is accessible and writable
    try:
        test_file_path = os.path.join(output_dir, "test_write_access.txt")
        with open(test_file_path, 'w') as f:
            f.write("Testing write access")
        os.remove(test_file_path)
        logger.info(f"Ensured Output directory {output_dir} exists and is writable")
    except Exception as e:
        logger.error(f"Error verifying Output directory access: {e}")
        logger.warning(f"Attempting to set permissions on Output directory")
        try:
            # Try to fix permissions
            os.chmod(output_dir, 0o777)  # Full access
            logger.info(f"Set permissions on Output directory to 777")
        except Exception as perm_error:
            logger.error(f"Failed to set permissions: {perm_error}")
    
    # Apply global BFloat16 to Float32 conversion patches
    logger.info("Applying global BFloat16 safety patches")
    
    # Patch matrix multiplication operations
    for op_name, op_func in [
        ("matmul", torch.matmul),
        ("bmm", torch.bmm),
        ("mm", torch.mm),
        ("linear", torch.nn.functional.linear)
    ]:
        if hasattr(torch, op_name) or (op_name == "linear" and hasattr(torch.nn.functional, "linear")):
            original_func = op_func
            def make_safe_op(orig_func, name):
                def safe_op(*args, **kwargs):
                    # Convert all tensor args from BFloat16 to Float32
                    new_args = []
                    for arg in args:
                        if isinstance(arg, torch.Tensor) and arg.dtype == torch.bfloat16:
                            arg = arg.to(torch.float32)
                        new_args.append(arg)
                    
                    # Convert all tensor kwargs from BFloat16 to Float32
                    new_kwargs = {}
                    for k, v in kwargs.items():
                        if isinstance(v, torch.Tensor) and v.dtype == torch.bfloat16:
                            v = v.to(torch.float32)
                        new_kwargs[k] = v
                    
                    return orig_func(*new_args, **new_kwargs)
                return safe_op
            
            safe_func = make_safe_op(original_func, op_name)
            
            if op_name == "linear":
                torch.nn.functional.linear = safe_func
            else:
                setattr(torch, op_name, safe_func)
            
            logger.info(f"Patched {op_name} for BFloat16 safety")
    
    # Override tensor.matmul and tensor.mm methods at class level
    orig_tensor_matmul = torch.Tensor.matmul
    def safe_tensor_matmul(self, other):
        if self.dtype == torch.bfloat16:
            self = self.to(torch.float32)
        if isinstance(other, torch.Tensor) and other.dtype == torch.bfloat16:
            other = other.to(torch.float32)
        return orig_tensor_matmul(self, other)
    torch.Tensor.matmul = safe_tensor_matmul
    
    orig_tensor_mm = torch.Tensor.mm
    def safe_tensor_mm(self, other):
        if self.dtype == torch.bfloat16:
            self = self.to(torch.float32)
        if other.dtype == torch.bfloat16:
            other = other.to(torch.float32)
        return orig_tensor_mm(self, other)
    torch.Tensor.mm = safe_tensor_mm
    
    logger.info("Applied tensor method patches for BFloat16 safety")
    
    # Force default dtype to float32 for all new tensors
    torch.set_default_dtype(torch.float32)
    logger.info("Set default dtype to float32")
    
    # Initialize GPU
    if torch.cuda.is_available():
        logger.info(f"Using GPU: {torch.cuda.get_device_name()}")
        # Disable TF32 to ensure consistent float32 operations
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.preferred_linalg_library(backend='cusolver')
        
        # Force non-mixed precision
        if hasattr(torch.cuda, 'amp'):
            torch.cuda.amp.autocast(enabled=False)
            
        logger.info("Disabled mixed precision and TF32 for consistent float32 operations")
    else:
        logger.warning("No GPU available, running in CPU mode")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
