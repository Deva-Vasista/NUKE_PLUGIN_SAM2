from fastapi import APIRouter, UploadFile, File, HTTPException
import cv2
import numpy as np
import OpenEXR
import Imath
from src.models.sam_wrapper import SAMProcessor
from src.schemas.models import EXRResponse, ProcessSequenceRequest
import tempfile
import os
import torch
from pathlib import Path
from loguru import logger
from src.utils.sequence_handler import EXRSequenceHandler

router = APIRouter()
processor = SAMProcessor()

def read_exr(file_content: bytes) -> str:
    """Read EXR file content and save as temporary file, return path"""
    # Create a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.exr') as tmp:
        tmp.write(file_content)
        return tmp.name

def write_exr(mask: np.ndarray) -> bytes:
    """Convert numpy mask to EXR bytes"""
    # Create a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.exr') as tmp:
        tmp_path = tmp.name
    
    try:
        # Create header
        header = OpenEXR.Header(mask.shape[1], mask.shape[0])
        header['channels'] = {
            'R': Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT)),
            'G': Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT)),
            'B': Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT))
        }
        
        # Create output file
        out = OpenEXR.OutputFile(tmp_path, header)
        
        # Convert mask to float and write
        mask_float = mask.astype(np.float32)
        out.writePixels({
            'R': mask_float.tobytes(),
            'G': mask_float.tobytes(),
            'B': mask_float.tobytes()
        })
        out.close()
        
        # Read the file back as bytes
        with open(tmp_path, 'rb') as f:
            return f.read()
    finally:
        # Clean up the temporary file
        os.unlink(tmp_path)

@router.post("/process_sequence")
async def process_sequence(request: ProcessSequenceRequest):
    try:
        logger.info(f"Received sequence path: {request.sequence_path}")
        logger.info(f"Sequence path type: {type(request.sequence_path)}")
        logger.info(f"Sequence path contains %04d: {'%04d' in request.sequence_path}")
        logger.info(f"Sequence path contains %03d: {'%03d' in request.sequence_path}")

        # Validate sequence path
        if not request.sequence_path:
            logger.error("Sequence path is required")
            raise HTTPException(status_code=400, detail="Sequence path is required")

        if not os.path.exists(os.path.dirname(request.sequence_path)):
            logger.error("Sequence directory does not exist")
            raise HTTPException(status_code=400, detail="Sequence directory does not exist")

        # Check for sequence pattern
        has_pattern = "%04d" in request.sequence_path or "%03d" in request.sequence_path
        logger.info(f"Sequence path has pattern: {has_pattern}")

        if not has_pattern:
            logger.error("Sequence path must contain %04d or %03d pattern")
            raise HTTPException(status_code=400, detail="Sequence path must contain %04d or %03d pattern")

        # Validate all frames exist
        sequence_handler = EXRSequenceHandler(
            sequence_path=request.sequence_path,
            frame_range=request.frame_range,
            original_fps=request.original_fps,
            target_fps=request.target_fps,
            bits=request.bits
        )
        missing_frames = [f for f in sequence_handler.frame_paths if not os.path.exists(f)]
        if missing_frames:
            logger.error(f"Missing frames: {missing_frames}")
            raise HTTPException(status_code=400, detail=f"Missing frames: {missing_frames}")

        # Call model on the whole sequence
        try:
            mask = processor.generate_mask(
                request.sequence_path,
                [float(x) for x in request.bbox.split(",")] if request.bbox else None,
                frame_range=request.frame_range,
                original_fps=request.original_fps,
                target_fps=request.target_fps,
                bits=request.bits
            )
        except Exception as e:
            logger.error(f"SAM processing failed for sequence: {str(e)}")
            raise HTTPException(status_code=500, detail=f"SAM processing failed for sequence: {str(e)}")

        # Save the first mask as EXR and return
        with tempfile.NamedTemporaryFile(delete=False, suffix='.exr') as tmp:
            tmp_path = tmp.name
        try:
            header = OpenEXR.Header(mask.shape[1], mask.shape[0])
            FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)
            mask_float = mask.astype(np.float32)
            out = OpenEXR.OutputFile(tmp_path, header)
            out.writePixels({'R': mask_float.tobytes(), 'G': mask_float.tobytes(), 'B': mask_float.tobytes()})
            out.close()
            with open(tmp_path, "rb") as f:
                result_bytes = f.read()
        finally:
            os.unlink(tmp_path)
        return {"result": result_bytes}
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Sequence processing failed: {str(e)}")
        logger.error(f"Exception type: {type(e)}")
        logger.error(f"Exception args: {e.args}")
        raise HTTPException(status_code=500, detail=f"Sequence processing failed: {str(e)}")

@router.post("/process_exr")
async def process_exr(
    image: UploadFile = File(...),
    bbox: str = None,
    frame_number: int = 1
):
    try:
        # Create temporary directory for processing
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Save uploaded file
            input_path = temp_path / "input.exr"
            with open(input_path, "wb") as f:
                f.write(await image.read())
            
            # Read EXR file
            frame = cv2.imread(str(input_path), cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
            if frame is None:
                raise HTTPException(status_code=400, detail="Failed to read EXR file")
            
            # Get original dimensions
            original_height, original_width = frame.shape[:2]
            
            # Convert to float32 if not already
            frame = frame.astype(np.float32)
            
            # Normalize based on bit depth
            if frame.dtype == np.uint8:
                frame = frame / 255.0
            elif frame.dtype == np.uint16:
                frame = frame / 65535.0
            elif frame.dtype == np.float32:
                # Already in float format, no normalization needed
                pass
            else:
                raise HTTPException(status_code=400, detail="Unsupported bit depth")
            
            # Convert to RGB if needed
            if len(frame.shape) == 2:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
            elif frame.shape[2] == 4:  # RGBA
                frame = frame[:, :, :3]  # Drop alpha channel
            
            # Resize to model input size (1024x1024)
            model_size = 1024
            frame = cv2.resize(frame, (model_size, model_size))
            
            # Convert to torch tensor and normalize
            frame_tensor = torch.from_numpy(frame).permute(2, 0, 1)
            img_mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)[:, None, None]
            img_std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)[:, None, None]
            frame_tensor = (frame_tensor - img_mean) / img_std
            
            # Convert back to numpy and save as temporary file
            frame_np = frame_tensor.permute(1, 2, 0).numpy()
            processed_path = temp_path / "processed.exr"
            cv2.imwrite(str(processed_path), frame_np)
            
            # Process with SAM
            if bbox:
                bbox = [float(x) for x in bbox.split(",")]
                # Scale bbox to model size
                bbox = [
                    bbox[0] * (model_size / original_width),
                    bbox[1] * (model_size / original_height),
                    bbox[2] * (model_size / original_width),
                    bbox[3] * (model_size / original_height)
                ]
                mask = processor.generate_mask(str(processed_path), bbox)
            else:
                # If no bbox provided, use the whole image as bbox
                bbox = [0, 0, model_size, model_size]
                mask = processor.generate_mask(str(processed_path), bbox)
            
            # Resize mask back to original dimensions
            mask = cv2.resize(mask, (original_width, original_height))
            
            # Save result
            output_path = temp_path / "output.exr"
            cv2.imwrite(str(output_path), mask.astype(np.float32))
            
            # Read and return result
            with open(output_path, "rb") as f:
                return {"result": f.read()}
            
    except Exception as e:
        logger.error(f"EXR processing failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"EXR processing failed: {str(e)}")
