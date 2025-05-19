import torch
from fastapi import APIRouter, UploadFile, File, HTTPException, Form, Query, WebSocket, WebSocketDisconnect, Body
from fastapi.responses import FileResponse, StreamingResponse
import numpy as np
import OpenEXR
import Imath
import array
from src.models.sam_wrapper import SAMProcessor
from src.schemas.models import EXRResponse, ProcessSequenceRequest
from src.utils.json_utils import convert_response
import tempfile
import os
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

# Global monkey patching for PyTorch operations to prevent BFloat16 errors
if hasattr(torch, "matmul"):
    original_matmul = torch.matmul
    def safe_matmul(input, other, *args, **kwargs):
        if hasattr(input, 'dtype') and input.dtype == torch.bfloat16:
            input = input.to(torch.float32)
        if hasattr(other, 'dtype') and other.dtype == torch.bfloat16:
            other = other.to(torch.float32)
        return original_matmul(input, other, *args, **kwargs)
    torch.matmul = safe_matmul
    logger.info("Applied safe matmul patching")

if hasattr(torch, "bmm"):
    original_bmm = torch.bmm
    def safe_bmm(input, mat2, *args, **kwargs):
        if hasattr(input, 'dtype') and input.dtype == torch.bfloat16:
            input = input.to(torch.float32)
        if hasattr(mat2, 'dtype') and mat2.dtype == torch.bfloat16:
            mat2 = mat2.to(torch.float32)
        return original_bmm(input, mat2, *args, **kwargs)
    torch.bmm = safe_bmm
    logger.info("Applied safe bmm patching")

if hasattr(torch, "mm"):
    original_mm = torch.mm
    def safe_mm(input, mat2, *args, **kwargs):
        if hasattr(input, 'dtype') and input.dtype == torch.bfloat16:
            input = input.to(torch.float32)
        if hasattr(mat2, 'dtype') and mat2.dtype == torch.bfloat16:
            mat2 = mat2.to(torch.float32)
        return original_mm(input, mat2, *args, **kwargs)
    torch.mm = safe_mm
    logger.info("Applied safe mm patching")

if hasattr(torch.nn.functional, "linear"):
    original_linear = torch.nn.functional.linear
    def safe_linear(input, weight, bias=None):
        if hasattr(input, 'dtype') and input.dtype == torch.bfloat16:
            input = input.to(torch.float32)
        if hasattr(weight, 'dtype') and weight.dtype == torch.bfloat16:
            weight = weight.to(torch.float32)
        if bias is not None and hasattr(bias, 'dtype') and bias.dtype == torch.bfloat16:
            bias = bias.to(torch.float32)
        return original_linear(input, weight, bias)
    torch.nn.functional.linear = safe_linear
    logger.info("Applied safe linear patching")

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

        # Convert Windows path to WSL path if needed
        if image_path.startswith("C:") or image_path.startswith("c:"):
            # Convert Windows path to WSL path
            drive_letter = image_path[0].lower()
            path_part = image_path[3:]
            path_part = path_part.replace('\\', '/')
            wsl_path = f"/mnt/{drive_letter}/{path_part}"
            logger.info(f"Converting Windows path '{image_path}' to WSL path '{wsl_path}'")
            image_path = wsl_path

        # Ensure file exists
        if not os.path.exists(image_path):
            raise HTTPException(status_code=404, detail=f"File not found: {image_path}")

        # Read EXR file from disk
        try:
            img = read_exr_to_numpy(image_path)
        except Exception as e:
            logger.error(f"Failed to read EXR file: {str(e)}")
            raise HTTPException(status_code=400, detail=f"Failed to read image: {str(e)}")
            
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
            # Ensure result is serializable
            return {"result": convert_response(result)}
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
    reverse = request.get("reverse", False)

    # Create a task ID for progress tracking
    task_id = str(uuid.uuid4())
    await progress_tracker.create_task(task_id)
    await progress_tracker.update_progress(task_id, 0, "Initializing model and loading sequence")

    # Load model/state
    try:
        # Enforce float32 precision throughout the model
        enforce_model_float32()
        
        # Use patched init_state method that handles Windows paths and tensor types
        state, images, frame_start = processor.patched_init_state(
        sequence_path,
        offload_video_to_cpu=True,
        frame_range_min=frame_range[0],
        frame_range_max=frame_range[1],
        original_fps=24,
        target_fps=24,
        bits=bits,
    )
        
        await progress_tracker.update_progress(task_id, 10, "Sequence loaded successfully")
    except FileNotFoundError as e:
        error_msg = str(e)
        logger.error(f"File not found error: {error_msg}")
        await progress_tracker.complete_task(task_id, False, f"File not found: {error_msg}")
        raise HTTPException(status_code=404, detail=f"File not found: {error_msg}")
    except Exception as e:
        logger.error(f"Error initializing state: {str(e)}")
        await progress_tracker.complete_task(task_id, False, f"Error processing sequence: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing sequence: {str(e)}")

    # Add all prompts
    try:
        await progress_tracker.update_progress(task_id, 20, "Adding prompts")
        for i, prompt in enumerate(prompts):
            obj_id = prompt.get("object_id", 0)
            frame_idx = prompt["frame_index"]
            points_positive = prompt.get("points_positive", [])
            points_negative = prompt.get("points_negative", [])
            points = points_positive + points_negative
            labels = [1] * len(points_positive) + [0] * len(points_negative)
            bbox = prompt.get("bbox")

            # Convert to tensors with right types
            points_tensor = torch.tensor(points, dtype=torch.float32) if points else None
            labels_tensor = torch.tensor(labels, dtype=torch.int32) if labels else None
            box_tensor = torch.tensor(bbox, dtype=torch.float32) if bbox is not None else None

            logger.info(f"[API] Adding prompt for obj_id={obj_id} at frame_idx={frame_idx}: points+labels={list(zip(points, labels))} bbox={bbox}")

            # Add prompt to model
            processor.model.add_new_points_or_box(
                state, 
                box=box_tensor,
                points=points_tensor,
                labels=labels_tensor,
                frame_idx=frame_idx,
                obj_id=obj_id
            )

            # Ensure all tensors are float32 after adding prompt
            with torch.no_grad():
                state = processor._convert_tensors_to_float32(state)

            # Update progress for each prompt
            prompt_progress = 20 + (i+1) * 10 / len(prompts) if prompts else 30
            await progress_tracker.update_progress(
                task_id, 
                prompt_progress, 
                f"Added prompt {i+1}/{len(prompts)}"
            )
    except Exception as e:
        logger.error(f"Error adding prompts: {str(e)}")
        await progress_tracker.complete_task(task_id, False, f"Error adding prompts: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error adding prompts: {str(e)}")

    # Propagate
    await progress_tracker.update_progress(task_id, 30, "Starting mask propagation")
    start_frame_idx = 0
    max_frame_num_to_track = frame_range[1] - 1
    total_frames = frame_range[1] - frame_range[0] + 1
    logger.info(f"[API] Propagating: start_frame_idx={start_frame_idx}, max_frame_num_to_track={max_frame_num_to_track}, reverse={reverse}")
    masks_by_frame = {}
    
    # Enforce float32 one more time right before propagation
    enforce_model_float32()
    
    try:
        # Create custom generator wrapper to enforce float32 during propagation
        def propagate_with_float32_conversion():
            frame_count = 0
            for frame_idx, object_ids, masks in processor.model.propagate_in_video(
                state, 
                start_frame_idx=start_frame_idx, 
                max_frame_num_to_track=max_frame_num_to_track, 
                reverse=reverse
            ):
                # Track progress
                frame_count += 1
                progress = 30 + (frame_count / total_frames) * 60
                asyncio.create_task(
                    progress_tracker.update_progress(
                        task_id, 
                        progress, 
                        f"Processing frame {frame_count}/{total_frames}"
                    )
                )
                
                # Ensure masks are float32
                if isinstance(masks, torch.Tensor) and masks.dtype != torch.float32:
                    masks = masks.to(torch.float32)
                yield frame_idx, object_ids, masks
        
        # Use our float32-enforcing generator
        for frame_idx, object_ids, masks in propagate_with_float32_conversion():
            masks_tensor = masks if isinstance(masks, torch.Tensor) else masks[0]
            if frame_idx not in masks_by_frame:
                masks_by_frame[frame_idx] = {}
            for obj_id, mask in zip(object_ids, masks_tensor):
                mask_np = mask[0].cpu().numpy() > 0.0
                masks_by_frame[frame_idx][obj_id] = mask_np
                
    except RuntimeError as e:
        if "expected scalar type Float but found BFloat16" in str(e):
            # Last resort fix: try monkey patching torch's module to force float32
            logger.error(f"[ERROR] Data type mismatch in propagate_in_video: {str(e)}")
            logger.warning("Attempting deep tensor type conversion...")
            await progress_tracker.update_progress(task_id, 50, "Attempting recovery from type mismatch error")
            
            # Try one more time with aggressive monkey patching
            old_linear = torch.nn.functional.linear
            def patched_linear(input, weight, bias=None):
                input_f32 = input.to(torch.float32) if input.dtype != torch.float32 else input
                weight_f32 = weight.to(torch.float32) if weight.dtype != torch.float32 else weight
                bias_f32 = bias.to(torch.float32) if bias is not None and bias.dtype != torch.float32 else bias
                return old_linear(input_f32, weight_f32, bias_f32)
                
            try:
                # Monkey patch the linear function temporarily
                torch.nn.functional.linear = patched_linear
                
                # Force convert all state tensors
                state = processor._convert_tensors_to_float32(state)
                
                # Try again with our patched function
                frame_count = 0
                for frame_idx, object_ids, masks in processor.model.propagate_in_video(
                    state, 
                    start_frame_idx=start_frame_idx, 
                    max_frame_num_to_track=max_frame_num_to_track, 
                    reverse=reverse
                ):
                    # Track progress during recovery
                    frame_count += 1
                    progress = 50 + (frame_count / total_frames) * 40
                    asyncio.create_task(
                        progress_tracker.update_progress(
                            task_id, 
                            progress, 
                            f"Recovery mode: Processing frame {frame_count}/{total_frames}"
                        )
                    )
                    
                    masks_tensor = masks if isinstance(masks, torch.Tensor) else masks[0]
                    if frame_idx not in masks_by_frame:
                        masks_by_frame[frame_idx] = {}
                    for obj_id, mask in zip(object_ids, masks_tensor):
                        mask_np = mask[0].cpu().numpy() > 0.0
                        masks_by_frame[frame_idx][obj_id] = mask_np
            finally:
                # Restore original function
                torch.nn.functional.linear = old_linear
                
            if not masks_by_frame:
                # If we still don't have masks, we've failed
                await progress_tracker.complete_task(task_id, False, "Failed to process sequence due to BFloat16 vs Float type mismatch")
                raise HTTPException(status_code=500, 
                                detail="Failed to process sequence due to BFloat16 vs Float type mismatch. Please restart the server.")
        else:
            logger.error(f"[ERROR] Error in propagate_in_video: {e}")
            await progress_tracker.complete_task(task_id, False, f"Error propagating masks: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error propagating masks: {str(e)}")
    except Exception as e:
        logger.error(f"[ERROR] Unexpected error in propagate_in_video: {e}")
        await progress_tracker.complete_task(task_id, False, f"Error processing sequence: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing sequence: {str(e)}")

    # Process and return results
    try:
        await progress_tracker.update_progress(task_id, 90, "Finalizing output")
        if as_file:
            # Save to output directory instead of streaming
            output_dir = os.path.join("Output", task_id)
            os.makedirs(output_dir, exist_ok=True)
            # Save individual EXR files to output directory
            saved_files = []
            for frame_idx, masks in masks_by_frame.items():
                # Combine all object masks into a single mask per frame
                shape = next(iter(masks.values())).shape
                combined_mask = np.zeros(shape, dtype=np.float32)
                for obj_id, mask in masks.items():
                    combined_mask[mask > 0] = obj_id + 1  # unique label per object
                # Save as EXR file
                output_path = os.path.join(output_dir, f"mask_{frame_idx:04d}.exr")
                write_exr(combined_mask, output_path)
                saved_files.append(output_path)
            # Also create a ZIP file for convenience
            zip_path = os.path.join(output_dir, f"masks_{task_id}.zip")
            with zipfile.ZipFile(zip_path, "w") as zipf:
                for file_path in saved_files:
                    zipf.write(file_path, os.path.basename(file_path))
            logger.info(f"Saved {len(saved_files)} mask files to {output_dir}")
            await progress_tracker.update_progress(task_id, 100, f"Completed processing {len(saved_files)} frames")
            await progress_tracker.complete_task(task_id, True)
            # Return file paths instead of streaming content
            return {
                "status": "success",
                "task_id": task_id,
                "output_dir": output_dir,
                "zip_path": zip_path,
                "file_count": len(saved_files),
                "frames": list(masks_by_frame.keys())
            }
        else:
            # Make sure all data is JSON-serializable
            serializable_masks = convert_response(masks_by_frame)
            # Also save JSON data to file for debugging
            output_dir = os.path.join("Output", task_id)
            os.makedirs(output_dir, exist_ok=True)
            json_path = os.path.join(output_dir, f"masks_{task_id}.json")
            with open(json_path, "w") as f:
                json.dump(serializable_masks, f)
            await progress_tracker.complete_task(task_id, True)
            return {
                "result": serializable_masks, 
                "task_id": task_id,
                "output_dir": output_dir,
                "json_path": json_path
            }
    except Exception as e:
        logger.error(f"[ERROR] Error finalizing output: {e}")
        await progress_tracker.complete_task(task_id, False, f"Error finalizing output: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error finalizing output: {str(e)}")

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
                await websocket.send_json(convert_response(status.to_dict()))
                if status.status in ["completed", "failed"]:
                    break
            await asyncio.sleep(0.5)
    except WebSocketDisconnect:
        pass  # Don't remove from connections here, do it in finally
    except Exception as e:
        try:
            await websocket.close(code=1011, reason=str(e))
        except:
            pass
    finally:
        # Always send the final status if possible
        status = progress_tracker.get_task_status(task_id)
        if status and status.status in ["completed", "failed"]:
            try:
                await websocket.send_json(convert_response(status.to_dict()))
            except:
                pass
        if task_id in progress_tracker.connections:
            progress_tracker.connections[task_id].discard(websocket)

def enforce_model_float32():
    """Enforce float32 throughout the entire model to prevent BFloat16 errors"""
    # First, ensure model is float32
    processor.model = processor.model.to(torch.float32)
    
    # Convert all model parameters
    for name, module in processor.model.named_modules():
        for param_name, param in module.named_parameters(recurse=False):
            if param.dtype != torch.float32:
                logger.warning(f"Converting parameter {name}.{param_name} from {param.dtype} to float32")
                param.data = param.data.to(torch.float32)
        
        # Convert all module buffers
        for buffer_name, buffer in module.named_buffers(recurse=False):
            if hasattr(buffer, 'dtype') and buffer.dtype != torch.float32:
                logger.warning(f"Converting buffer {name}.{buffer_name} from {buffer.dtype} to float32")
                buffer.data = buffer.data.to(torch.float32)
                
        # Set module computation to float32
        if hasattr(module, 'to'):
            module.to(torch.float32)
            
    logger.info("Model float32 enforcement complete")

@router.get("/download/{filename}")
async def download_file(filename: str):
    """
    Download a previously generated file by filename.
    This allows the client to access files that were saved to the server's output directory.
    """
    # Safety check to prevent path traversal attacks
    safe_filename = os.path.basename(filename)
    
    # First, check in the root Output directory
    output_path = os.path.join("Output", safe_filename)
    if os.path.exists(output_path):
        return FileResponse(output_path)
    
    # Handle absolute paths - convert Windows paths to WSL paths
    if filename.startswith('C:') or filename.startswith('c:'):
        drive_letter = filename[0].lower()
        path_part = filename[3:]
        path_part = path_part.replace('\\', '/')
        wsl_path = f"/mnt/{drive_letter}/{path_part}"
        
        if os.path.exists(wsl_path):
            return FileResponse(wsl_path)
    
    # Then check in subdirectories of Output
    for root, dirs, files in os.walk("Output"):
        for file in files:
            if file == safe_filename:
                file_path = os.path.join(root, file)
                return FileResponse(file_path)
    
    # If file is not found, look for UUID directories that might contain it
    for item in os.listdir("Output"):
        dir_path = os.path.join("Output", item)
        if os.path.isdir(dir_path):
            file_path = os.path.join(dir_path, safe_filename)
            if os.path.exists(file_path):
                return FileResponse(file_path)
    
    # File not found
    raise HTTPException(status_code=404, detail=f"File not found: {safe_filename}")

@router.get("/output/{task_id}")
async def get_task_output(task_id: str):
    """
    Get metadata about a task's output files.
    """
    output_dir = os.path.join("Output", task_id)
    if not os.path.exists(output_dir):
        raise HTTPException(status_code=404, detail=f"No output directory found for task: {task_id}")
    # Try to load status.json if task not in memory
    status = progress_tracker.get_task_status(task_id)
    if not status:
        status_path = os.path.join(output_dir, "status.json")
        if os.path.exists(status_path):
            with open(status_path, "r") as f:
                data = json.load(f)
            # Optionally, return status info in the response
    # Collect file information
    files = []
    for file in os.listdir(output_dir):
        file_path = os.path.join(output_dir, file)
        files.append({
            "filename": file,
            "path": file_path,
            "size": os.path.getsize(file_path),
            "url": f"/api/v1/download/{file}"
        })
    return {
        "task_id": task_id,
        "output_dir": output_dir,
        "files": files
    }

@router.get("/download_all/{task_id}")
async def download_all_task_files(task_id: str):
    """
    Download all files for a task as a single ZIP file.
    This is useful for clients that need reliable access to all outputs.
    """
    output_dir = os.path.join("Output", task_id)
    if not os.path.exists(output_dir):
        raise HTTPException(status_code=404, detail=f"No output directory found for task: {task_id}")
    
    # Create a temporary ZIP file with all files from the task directory
    try:
        zip_filename = f"all_files_{task_id}.zip"
        zip_path = os.path.join(output_dir, zip_filename)
        
        # Create the ZIP file
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, _, files in os.walk(output_dir):
                for file in files:
                    # Skip the ZIP file itself
                    if file == zip_filename:
                        continue
                    file_path = os.path.join(root, file)
                    # Add file to ZIP with a relative path
                    arcname = os.path.relpath(file_path, output_dir)
                    zipf.write(file_path, arcname=arcname)
        
        # Return the ZIP file
        return FileResponse(
            zip_path, 
            filename=zip_filename,
            media_type="application/zip"
        )
    except Exception as e:
        logger.error(f"Error creating ZIP of all files for task {task_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Error creating ZIP file: {str(e)}")

@router.get("/status/{task_id}")
async def get_task_status(task_id: str):
    """
    Get the current status of a processing task.
    This is useful for clients that need to check if processing is complete.
    """
    status = progress_tracker.get_task_status(task_id)
    if status:
        return convert_response(status.to_dict())
    else:
        # Try loading from disk (status.json)
        output_dir = os.path.join("Output", task_id)
        status_path = os.path.join(output_dir, "status.json")
        if os.path.exists(status_path):
            with open(status_path, "r") as f:
                data = json.load(f)
            return convert_response(data)
        raise HTTPException(status_code=404, detail=f"Task not found: {task_id}")

@router.post("/reset")
async def reset_model_state():
    """Reset the model state and clear any cached data."""
    try:
        # Reset the SAM processor
        success = processor.reset_state()
        if not success:
            raise RuntimeError("Failed to reset SAM processor state")
        
        # Clear progress tracker
        try:
            progress_tracker.clear_all()
        except Exception as e:
            logger.warning(f"Non-critical error clearing progress tracker: {e}")
        
        # Force GPU memory cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return {
            "status": "success",
            "message": "Model state reset successfully"
        }
    except Exception as e:
        logger.error(f"Error resetting model state: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to reset model state: {str(e)}"
        )
