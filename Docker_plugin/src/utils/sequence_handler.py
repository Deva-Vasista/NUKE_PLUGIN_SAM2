import os
import re
import cv2
import numpy as np
import torch
from pathlib import Path
from typing import List, Tuple, Optional
from loguru import logger
from tqdm import tqdm

class EXRSequenceHandler:
    def __init__(
        self,
        sequence_path: str,
        frame_range: Tuple[int, int],
        original_fps: int = 24,
        target_fps: int = 24,
        bits: int = 32,
        image_size: int = 1024
    ):
        self.sequence_path = sequence_path
        self.frame_range_min, self.frame_range_max = frame_range
        self.original_fps = original_fps
        self.target_fps = target_fps
        self.bits = bits
        self.image_size = image_size
        
        # Parse sequence path
        self.input_path_folder = os.path.dirname(sequence_path)
        self.file_extension = os.path.splitext(sequence_path)[1]
        
        # Check for sequence pattern in the full path
        if "%04d" in sequence_path:
            self.frame_pattern = "%04d"
            # Split on the last occurrence of %04d to get the base name
            parts = sequence_path.rsplit("%04d", 1)
            self.input_file_name_split = parts[0]
        elif "%03d" in sequence_path:
            self.frame_pattern = "%03d"
            # Split on the last occurrence of %03d to get the base name
            parts = sequence_path.rsplit("%03d", 1)
            self.input_file_name_split = parts[0]
        else:
            raise ValueError("Input is not a sequence. Expected frame pattern like %04d or %03d")
        
        # Calculate stride for frame rate conversion
        self.stride = max(round(original_fps / target_fps), 1)
        
        # Get all frames in the sequence
        self.frame_paths = self._get_frame_paths()
        if not self.frame_paths:
            raise ValueError(f"No frames found in sequence: {sequence_path}")
        
        logger.info(f"Found {len(self.frame_paths)} frames in sequence")
        
    def _get_frame_paths(self) -> List[str]:
        """Get all frame paths in the sequence within the specified range."""
        frame_paths = []
        total_frames = self.frame_range_max - self.frame_range_min + 1
        
        # Create progress bar
        pbar = tqdm(total=total_frames, desc="Scanning sequence", unit="frames")
        
        for frame_num in range(self.frame_range_min, self.frame_range_max + 1):
            # Format frame number with correct padding
            if self.frame_pattern == "%04d":
                padded_frame = f"{frame_num:04d}"
            else:  # %03d
                padded_frame = f"{frame_num:03d}"
            
            # Construct frame path by replacing the pattern in the full sequence path
            frame_path = self.sequence_path.replace(self.frame_pattern, padded_frame)
            if os.path.exists(frame_path):
                frame_paths.append(frame_path)
            
            # Update progress
            pbar.update(1)
            
        pbar.close()
        return sorted(frame_paths)
    
    def read_frame(self, frame_path: str) -> torch.Tensor:
        """Read and process a single frame."""
        # Read EXR file
        frame = cv2.imread(frame_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        if frame is None:
            raise ValueError(f"Failed to read frame: {frame_path}")
        
        # Convert to float32 if not already
        frame = frame.astype(np.float32)
        
        # Normalize based on bit depth
        if self.bits == 8:
            frame = frame / 255.0
        elif self.bits == 16:
            frame = frame / 65535.0
        elif self.bits == 32:
            # Already in float format, no normalization needed
            pass
        else:
            raise ValueError(f"Unsupported bit depth: {self.bits}")
        
        # Convert to RGB if needed
        if len(frame.shape) == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        elif frame.shape[2] == 4:  # RGBA
            frame = frame[:, :, :3]  # Drop alpha channel
        
        # Resize to model input size
        frame = cv2.resize(frame, (self.image_size, self.image_size))
        
        # Convert to torch tensor and normalize
        frame_tensor = torch.from_numpy(frame).permute(2, 0, 1)
        img_mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)[:, None, None]
        img_std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)[:, None, None]
        frame_tensor = (frame_tensor - img_mean) / img_std
        
        return frame_tensor
    
    def get_original_dimensions(self) -> Tuple[int, int]:
        """Get the original dimensions of the first frame."""
        first_frame = cv2.imread(self.frame_paths[0], cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        return first_frame.shape[:2]
    
    def get_frame_count(self) -> int:
        """Get the total number of frames in the sequence."""
        return len(self.frame_paths) 