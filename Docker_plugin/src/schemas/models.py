from pydantic import BaseModel, Field
from typing import Optional, List, Tuple, Dict, Any, Union, Literal

class EXRResponse(BaseModel):
    result: bytes

class Point(BaseModel):
    x: float = Field(..., description="X coordinate")
    y: float = Field(..., description="Y coordinate")

class BoundingBox(BaseModel):
    x1: float = Field(..., description="Left coordinate")
    y1: float = Field(..., description="Top coordinate")
    x2: float = Field(..., description="Right coordinate")
    y2: float = Field(..., description="Bottom coordinate")

class ProcessSequenceRequest(BaseModel):
    sequence_path: str
    frame_range: Tuple[int, int] = Field(..., description="(start_frame, end_frame)")
    original_fps: float = Field(24.0, description="Original frames per second")
    target_fps: float = Field(24.0, description="Target frames per second")
    bits: int = Field(32, description="Bit depth of the EXR files (8, 16, or 32)")
    dimensions: Optional[Tuple[int, int]] = Field(None, description="Expected image dimensions (width, height)")

class ProcessEXRRequest(BaseModel):
    image_path: str
    bbox: Optional[List[float]] = Field(None, description="Bounding box [x1, y1, x2, y2]")
    points_positive: Optional[List[List[float]]] = Field(None, description="List of positive points [[x1, y1], [x2, y2], ...]")
    points_negative: Optional[List[List[float]]] = Field(None, description="List of negative points [[x1, y1], [x2, y2], ...]")
    bits: Literal["8-bit integer", "16-bit integer", "32-bit float"] = Field("32-bit float", description="Bit depth format")
    dimensions: Optional[Tuple[int, int]] = Field(None, description="Expected image dimensions (width, height)")

class ModelLoadRequest(BaseModel):
    model_type: Literal["large", "base-plus", "small", "tiny"] = Field(..., description="Type of SAM2 model to load")

class ModelLoadResponse(BaseModel):
    status: str
    model_loaded: str
    vram_usage: float
    bfloat16_tensors_fixed: Optional[int] = Field(0, description="Number of BFloat16 tensors that were converted to Float32")

class ModelStatusResponse(BaseModel):
    model_loaded: bool
    model_path: str
    device: str
    vram_usage: float

# Generic serializable response types to ensure proper JSON serialization
class SerializableData(BaseModel):
    data: Dict[str, Any]

class SerializableResponse(BaseModel):
    success: bool = True
    result: Any = None
    error: Optional[str] = None
