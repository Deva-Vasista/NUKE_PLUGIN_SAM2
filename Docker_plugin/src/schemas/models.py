from pydantic import BaseModel
from typing import Optional, List, Tuple

class EXRResponse(BaseModel):
    result: bytes

class ProcessSequenceRequest(BaseModel):
    sequence_path: str
    frame_range: Tuple[int, int]  # (start_frame, end_frame)
    bbox: Optional[str] = None  # "x1,y1,x2,y2"
    original_fps: int = 24
    target_fps: int = 24
    bits: int = 32  # bit depth of the EXR files

class ModelLoadResponse(BaseModel):
    status: str
    model_loaded: str
    vram_usage: float
