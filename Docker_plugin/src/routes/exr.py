from fastapi import APIRouter, UploadFile, File, HTTPException, Form, Query, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, StreamingResponse
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
import zipfile
import io
import json

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

def write_exr_to_bytes(mask: np.ndarray) -> bytes:
    """Write numpy array to EXR format in memory and return bytes."""
    height, width = mask.shape[-2:]
    header = OpenEXR.Header(width, height)
    header['channels'] = {
        'R': Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT)),
        'G': Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT)),
        'B': Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT))
    }
    mask_float = mask.astype(np.float32)
    # Write to BytesIO
    buf = io.BytesIO()
    # OpenEXR does not support writing to BytesIO directly, so use temp file workaround
    with tempfile.NamedTemporaryFile(delete=False, suffix='.exr') as tmp:
        out = OpenEXR.OutputFile(tmp.name, header)
        out.writePixels({
            'R': mask_float.tobytes(),
            'G': mask_float.tobytes(),
            'B': mask_float.tobytes()
        })
        out.close()
        tmp.seek(0)
        buf.write(tmp.read())
    return buf.getvalue()

@router.post("/process_exr")
async def process_exr(
    image: UploadFile = File(...),
    bbox: list = Form(None),
    points_positive: str = Form(None, description="JSON list of [x, y] points to select"),
    points_negative: str = Form(None, description="JSON list of [x, y] points to remove"),
    bits: str = Form("32-bit float", description="Bit depth, e.g. '32-bit float'"),
    as_file: bool = Query(False, description="Return mask as EXR file if true, else as list")
):
    """Process a single EXR file with bbox and/or point selection/removal."""
    try:
        # Parse points
        pos_points = json.loads(points_positive) if points_positive else None
        neg_points = json.loads(points_negative) if points_negative else None
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.exr') as tmp:
            contents = await image.read()
            tmp.write(contents)
            tmp_path = tmp.name
        try:
            img = read_exr_to_numpy(tmp_path)
            if img is None:
                raise HTTPException(status_code=400, detail="Failed to read image")
            image_size = getattr(processor.model, 'image_size', 256)
            if img.shape[0] != image_size or img.shape[1] != image_size:
                img = cv2.resize(img, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
            # --- Robust prompt handling ---
            prompt_type = None
            if bbox and (pos_points or neg_points):
                prompt_type = 'both'
            elif bbox:
                prompt_type = 'bbox'
            elif pos_points or neg_points:
                prompt_type = 'points'
            else:
                raise HTTPException(status_code=400, detail="Must provide bbox or points")
            print(f"[DEBUG] Adding prompt: {prompt_type}")
            # Ensure bbox is a list of 4 floats if provided
            bbox_array = [float(x) for x in bbox] if bbox else None
            # Call model
            result = await processor.generate_mask_async(img, bbox_array, pos_points, neg_points, bits=bits)
            if as_file:
                with tempfile.NamedTemporaryFile(delete=False, suffix='.exr') as mask_tmp:
                    write_exr(result.squeeze(), mask_tmp.name)
                    return FileResponse(mask_tmp.name, media_type="image/exr", filename="mask.exr")
            else:
                return {"result": result.tolist()}
        finally:
            os.unlink(tmp_path)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"EXR processing failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/process_sequence")
async def process_sequence(request: dict, as_file: bool = Query(False, description="Return masks as EXR files if true, else as list")):
    """Process an EXR sequence with bbox and/or point selection/removal."""
    try:
        sequence_path = request.get("sequence_path")
        frame_range = request.get("frame_range", [1, 1])
        bbox = request.get("bbox")
        points_positive = request.get("points_positive")
        points_negative = request.get("points_negative")
        bits = request.get("bits", "32-bit float")
        if not sequence_path:
            raise HTTPException(status_code=400, detail="Missing sequence_path")
        if not bbox and not points_positive and not points_negative:
            raise HTTPException(status_code=400, detail="Must provide bbox or points")
        # Parse points
        pos_points = points_positive if isinstance(points_positive, list) else json.loads(points_positive) if points_positive else None
        neg_points = points_negative if isinstance(points_negative, list) else json.loads(points_negative) if points_negative else None
        # --- Robust prompt handling ---
        prompt_type = None
        if bbox and (pos_points or neg_points):
            prompt_type = 'both'
        elif bbox:
            prompt_type = 'bbox'
        elif pos_points or neg_points:
            prompt_type = 'points'
        else:
            raise HTTPException(status_code=400, detail="Must provide bbox or points")
        print(f"[DEBUG] Adding prompt: {prompt_type}")
        # Ensure bbox is a list of 4 floats if provided
        bbox_array = [float(x) for x in bbox] if bbox else None
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
            # File/sequence existence check (unchanged)
            file_exists = False
            frame_files = []
            if isinstance(sequence_path, str) and ("%04d" in sequence_path or "%03d" in sequence_path):
                frame_pattern = "%04d" if "%04d" in sequence_path else "%03d"
                for frame_num in range(frame_range[0], frame_range[1] + 1):
                    padded_frame = f"{frame_num:04d}" if frame_pattern == "%04d" else f"{frame_num:03d}"
                    frame_path = sequence_path.replace(frame_pattern, padded_frame)
                    print(f"[DEBUG] Checking: {frame_path} Exists: {os.path.exists(frame_path)}")
                    if os.path.exists(frame_path):
                        file_exists = True
                        frame_files.append(frame_path)
                if not file_exists:
                    raise FileNotFoundError(f"No file(s) found for sequence_path: {sequence_path}")
            elif isinstance(sequence_path, str) and os.path.exists(sequence_path):
                file_exists = True
                frame_files = [sequence_path]
            else:
                raise FileNotFoundError(f"No file(s) found for sequence_path: {sequence_path}")
            # --- END: File/sequence existence check ---
            print(f"[DEBUG] Calling track_sequence with: sequence_path={sequence_path}, frame_range={frame_range}, bbox={bbox_array}, pos_points={pos_points}, neg_points={neg_points}")
            masks_list = processor.track_sequence(
                sequence_path,
                frame_range,
                bbox=bbox_array,
                points_positive=pos_points,
                points_negative=neg_points,
                bits=bits
            )
            print(f"[DEBUG] track_sequence returned {len(masks_list)} masks. Type of first mask: {type(masks_list[0]) if masks_list else 'None'}")
            if as_file:
                zip_buffer = io.BytesIO()
                with zipfile.ZipFile(zip_buffer, "w") as zipf:
                    for idx, mask in enumerate(masks_list):
                        print(f"[DEBUG] Writing mask {idx}, type: {type(mask)}, shape: {getattr(mask, 'shape', None)}")
                        mask_bytes = write_exr_to_bytes(mask.squeeze())
                        zipf.writestr(f"mask_{idx:04d}.exr", mask_bytes)
                zip_buffer.seek(0)
                return StreamingResponse(zip_buffer, media_type="application/zip", headers={"Content-Disposition": "attachment; filename=masks.zip"})
            else:
                masks_json = [mask.tolist() for mask in masks_list]
                return {"result": masks_json, "task_id": task_id}
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
