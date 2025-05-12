from fastapi import APIRouter, UploadFile, File, HTTPException, Form, Query, WebSocket, WebSocketDisconnect, Body
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
    """Write numpy array to EXR format in a temp file and return bytes.
    Note: OpenEXR does not support writing to BytesIO; must use a real file.
    For speed, this uses NamedTemporaryFile and deletes after reading.
    Future speedup: If OpenCV EXR writing is hardware-accelerated, could use cv2.imwrite to a buffer.
    For many masks, consider parallelizing this step.
    """
    height, width = mask.shape[-2:]
    header = OpenEXR.Header(width, height)
    header['channels'] = {
        'R': Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT)),
        'G': Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT)),
        'B': Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT))
    }
    mask_float = mask.astype(np.float32)
    with tempfile.NamedTemporaryFile(delete=False, suffix='.exr') as tmp:
        tmp_path = tmp.name
    try:
        out = OpenEXR.OutputFile(tmp_path, header)
        out.writePixels({
            'R': mask_float.tobytes(),
            'G': mask_float.tobytes(),
            'B': mask_float.tobytes()
        })
        out.close()
        with open(tmp_path, 'rb') as f:
            data = f.read()
    finally:
        os.remove(tmp_path)
    return data

@router.post("/process_exr")
async def process_exr(
    request: dict = Body(...),
    as_file: bool = Query(False, description="Return mask as EXR file if true, else as list")
):
    """Process a single EXR file with bbox and/or point selection/removal, using JSON input."""
    try:
        image_path = request["image_path"]
        bbox = request.get("bbox", None)
        points_positive = request.get("points_positive", [])
        points_negative = request.get("points_negative", [])
        bits = request.get("bits", "32-bit float")

        # Read EXR file from disk
        img = read_exr_to_numpy(image_path)
        if img is None:
            raise HTTPException(status_code=400, detail="Failed to read image")
        image_size = getattr(processor.model, 'image_size', 256)
        if img.shape[0] != image_size or img.shape[1] != image_size:
            img = cv2.resize(img, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
        # Call model (pass bbox as box, points/labels as points_positive/points_negative)
        result = await processor.generate_mask_async(
            img,
            bbox,
            points_positive if points_positive else None,
            points_negative if points_negative else None,
            bits=bits
        )
        if as_file:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.exr') as mask_tmp:
                write_exr(result.squeeze(), mask_tmp.name)
                return FileResponse(mask_tmp.name, media_type="image/exr", filename="mask.exr")
        else:
            return {"result": result.tolist()}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"EXR processing failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/process_sequence")
async def process_sequence(
    request: dict = Body(...),
    as_file: bool = Query(False, description="Return masks as zip if true, else as JSON list"),
):
    """
    Process a sequence of EXR files with multi-object, multi-frame prompts.
    Request body should include:
      - sequence_path: str
      - frame_range: [start, end]
      - bits: str (optional)
      - prompts: list of dicts, each with:
          - frame_index: int
          - object_id: int
          - points_positive: list of [x, y]
          - points_negative: list of [x, y]
          - bbox: [x1, y1, x2, y2] (optional)
    """
    sequence_path = request["sequence_path"]
    frame_range = request["frame_range"]
    bits = request.get("bits", "32-bit float")
    prompts = request.get("prompts", [])

    # Load model/state
    state, images, frame_start = processor.model.init_state(
        sequence_path,
        offload_video_to_cpu=True,
        frame_range_min=frame_range[0],
        frame_range_max=frame_range[1],
        original_fps=24,
        target_fps=24,
        bits=bits,
    )

    # Add all prompts
    for prompt in prompts:
        obj_id = prompt.get("object_id", 0)
        frame_idx = prompt["frame_index"]
        points_positive = prompt.get("points_positive", [])
        points_negative = prompt.get("points_negative", [])
        points = points_positive + points_negative
        labels = [1] * len(points_positive) + [0] * len(points_negative)
        bbox = prompt.get("bbox")
        logger.info(f"[API] Adding prompt for obj_id={obj_id} at frame_idx={frame_idx}: points+labels={list(zip(points, labels))} bbox={bbox}")
        processor.model.add_new_points_or_box(state, box=bbox, points=points, labels=labels, frame_idx=frame_idx, obj_id=obj_id)

    # Propagate
    start_frame_idx = 0
    max_frame_num_to_track = frame_range[1] - 1
    logger.info(f"[API] Propagating: start_frame_idx={start_frame_idx}, max_frame_num_to_track={max_frame_num_to_track}")
    masks_by_frame = {}
    for frame_idx, object_ids, masks in processor.model.propagate_in_video(state, start_frame_idx=start_frame_idx, max_frame_num_to_track=max_frame_num_to_track):
        masks_tensor = masks if isinstance(masks, torch.Tensor) else masks[0]
        if frame_idx not in masks_by_frame:
            masks_by_frame[frame_idx] = {}
        for obj_id, mask in zip(object_ids, masks_tensor):
            mask_np = mask[0].cpu().numpy() > 0.0
            masks_by_frame[frame_idx][obj_id] = mask_np

    if as_file:
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w") as zipf:
            for frame_idx, masks in masks_by_frame.items():
                # Combine all object masks into a single mask per frame
                shape = next(iter(masks.values())).shape
                combined_mask = np.zeros(shape, dtype=np.float32)
                for obj_id, mask in masks.items():
                    combined_mask[mask > 0] = obj_id + 1  # unique label per object
                exr_bytes = write_exr_to_bytes(combined_mask)
                zipf.writestr(f"mask_{frame_idx:04d}.exr", exr_bytes)
        zip_buffer.seek(0)
        return StreamingResponse(zip_buffer, media_type="application/zip", headers={"Content-Disposition": "attachment; filename=masks.zip"})
    else:
        masks_json = [masks for frame_idx, masks in masks_by_frame.items()]
        return {"result": masks_json}

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
