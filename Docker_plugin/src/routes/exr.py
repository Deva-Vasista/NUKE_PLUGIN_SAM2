from fastapi import APIRouter, UploadFile, File, HTTPException, Form, Query, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
import numpy as np
import OpenEXR
import Imath
import array
from src.models.sam_wrapper import SAMProcessor
from src.schemas.models import EXRResponse, ProcessSequenceRequest
import tempfile
import os
import torch
from pathlib import Path
from loguru import logger
from src.utils.sequence_handler import EXRSequenceHandler
from src.utils.progress_tracker import ProgressTracker
import uuid
import asyncio
import cv2

router = APIRouter()
processor = SAMProcessor()
progress_tracker = ProgressTracker()

def read_exr_to_numpy(file_path: str) -> np.ndarray:
    """Read EXR file and convert to numpy array."""
    exr_file = OpenEXR.InputFile(file_path)
    
    # Get data window
    dw = exr_file.header()['dataWindow']
    width = dw.max.x - dw.min.x + 1
    height = dw.max.y - dw.min.y + 1

    # Read all channels
    FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)
    channels = ['R', 'G', 'B']
    channel_data = []
    
    for channel in channels:
        data = array.array('f', exr_file.channel(channel, FLOAT)).tolist()
        channel_data.append(np.array(data).reshape(height, width))
    
    # Stack channels
    img = np.dstack(channel_data)
    return img

def write_exr(mask: np.ndarray, output_path: str):
    """Write numpy array to EXR file."""
    height, width = mask.shape[-2:]
    
    header = OpenEXR.Header(width, height)
    header['channels'] = {
        'R': Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT)),
        'G': Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT)),
        'B': Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT))
    }
    
    out = OpenEXR.OutputFile(output_path, header)
    mask_float = mask.astype(np.float32)
    
    out.writePixels({
        'R': mask_float.tobytes(),
        'G': mask_float.tobytes(),
        'B': mask_float.tobytes()
    })
    out.close()

def parse_bbox(bbox_str: str) -> np.ndarray:
    """Parse bbox string into numpy array."""
    try:
        bbox = [float(x) for x in bbox_str.split(",")]
        if len(bbox) != 4:
            raise ValueError("Bbox must have 4 values")
        return np.array(bbox)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid bbox format: {str(e)}")

@router.post("/process_exr")
async def process_exr(
    image: UploadFile = File(...),
    bbox: str = Form(...),
    as_file: bool = Query(False, description="Return mask as EXR file if true, else as list")
):
    """Process a single EXR file."""
    try:
        # Parse bbox
        bbox_array = parse_bbox(bbox)
        
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.exr') as tmp:
            contents = await image.read()
            tmp.write(contents)
            tmp_path = tmp.name
        
        try:
            # Read EXR using OpenEXR
            img = read_exr_to_numpy(tmp_path)
            
            if img is None:
                raise HTTPException(status_code=400, detail="Failed to read image")

            # Resize image to model's expected input size
            image_size = getattr(processor.model, 'image_size', 256)  # Default to 256 if not set
            if img.shape[0] != image_size or img.shape[1] != image_size:
                img = cv2.resize(img, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
            
            # Process with SAM
            result = await processor.generate_mask_async(img, bbox_array)
            
            if as_file:
                # Save mask as EXR and return as file
                with tempfile.NamedTemporaryFile(delete=False, suffix='.exr') as mask_tmp:
                    write_exr(result.squeeze(), mask_tmp.name)
                    return FileResponse(mask_tmp.name, media_type="image/exr", filename="mask.exr")
            else:
                return {"result": result.tolist()}
            
        finally:
            # Clean up temporary file
            os.unlink(tmp_path)
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"EXR processing failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/process_sequence")
async def process_sequence(request: dict, as_file: bool = Query(False, description="Return masks as EXR files if true, else as list")):
    """Process an EXR sequence."""
    try:
        sequence_path = request.get("sequence_path")
        frame_range = request.get("frame_range", [1, 1])
        bbox = request.get("bbox")
        
        if not sequence_path:
            raise HTTPException(status_code=400, detail="Missing sequence_path")
            
        if not bbox:
            raise HTTPException(status_code=400, detail="Missing bbox")
            
        # Parse bbox
        bbox_array = parse_bbox(bbox)

        # --- BEGIN: frame_range validation ---
        if not (isinstance(frame_range, (list, tuple)) and len(frame_range) == 2):
            raise ValueError("frame_range must be a list or tuple of two integers [start, end]")
        start, end = frame_range
        if not (isinstance(start, int) and isinstance(end, int)):
            raise ValueError("frame_range values must be integers")
        if start < 1 or end < 1:
            raise ValueError("frame_range values must be >= 1")
        if start > end:
            raise ValueError("frame_range start must be <= end")
        # --- END: frame_range validation ---
        
        # Create task for tracking
        task_id = str(uuid.uuid4())
        await progress_tracker.create_task(task_id)
        
        try:
            # --- BEGIN: File/sequence existence check (moved inside inner try) ---
            file_exists = False
            if "%04d" in sequence_path or "%03d" in sequence_path:
                frame_pattern = "%04d" if "%04d" in sequence_path else "%03d"
                for frame_num in range(frame_range[0], frame_range[1] + 1):
                    padded_frame = f"{frame_num:04d}" if frame_pattern == "%04d" else f"{frame_num:03d}"
                    frame_path = sequence_path.replace(frame_pattern, padded_frame)
                    if os.path.exists(frame_path):
                        file_exists = True
                        break
            else:
                file_exists = Path(sequence_path).exists()
            if not file_exists:
                raise FileNotFoundError(f"No file(s) found for sequence_path: {sequence_path}")
            # --- END: File/sequence existence check ---

            result = await processor.generate_mask_async(
                sequence_path,
                bbox_array,
                frame_range=frame_range,
                progress_callback=lambda p, m: progress_tracker.update_progress(task_id, p, m)
            )
            
            await progress_tracker.complete_task(task_id, success=True)
            if as_file:
                # Save each mask as EXR and return list of file paths (for demo, return first mask as file)
                if isinstance(result, np.ndarray) and result.ndim >= 3:
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.exr') as mask_tmp:
                        write_exr(result[0].squeeze(), mask_tmp.name)
                        return FileResponse(mask_tmp.name, media_type="image/exr", filename="mask_seq_0.exr")
                else:
                    raise HTTPException(status_code=500, detail="Result is not a valid mask array")
            else:
                return {"result": result.tolist(), "task_id": task_id}
            
        except FileNotFoundError as e:
            await progress_tracker.complete_task(task_id, success=False, error=str(e))
            raise HTTPException(status_code=404, detail=str(e))
        except ValueError as e:
            await progress_tracker.complete_task(task_id, success=False, error=str(e))
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            await progress_tracker.complete_task(task_id, success=False, error=str(e))
            raise
            
    except HTTPException:
        raise
    except ValueError as e:
        logger.error(f"SAM processing failed for sequence: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"SAM processing failed for sequence: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.websocket("/ws/{task_id}")
async def websocket_progress(websocket: WebSocket, task_id: str):
    await websocket.accept()
    try:
        # Register this websocket connection for the task
        if not hasattr(progress_tracker, 'connections'):
            progress_tracker.connections = {}
        if task_id not in progress_tracker.connections:
            progress_tracker.connections[task_id] = set()
        progress_tracker.connections[task_id].add(websocket)
        # Send updates until task is done
        while True:
            status = progress_tracker.get_task_status(task_id)
            if status:
                await websocket.send_json(status.to_dict())
                if status.status in ["completed", "failed"]:
                    break
            await asyncio.sleep(0.5)
    except WebSocketDisconnect:
        if task_id in progress_tracker.connections:
            progress_tracker.connections[task_id].discard(websocket)
    except Exception as e:
        await websocket.close(code=1011, reason=str(e))
    finally:
        if task_id in progress_tracker.connections:
            progress_tracker.connections[task_id].discard(websocket)
