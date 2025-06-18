import os
import re
import cv2
import numpy as np
import torch
from pathlib import Path
from typing import List, Tuple, Optional, Dict
from loguru import logger
from tqdm import tqdm

def detect_sequence_pattern(path: str) -> Tuple[Optional[str], Optional[int], Optional[str], Optional[str]]:
    """
    Detect sequence pattern in path and extract naming components.
    Returns:
    - pattern: The sequence pattern (e.g., %05d)
    - padding: Number of digits
    - base_name: The part before the frame number
    - extension: The file extension
    """
    # First try to find standard sequence patterns (%03d to %10d)
    for i in range(3, 11):
        pattern = f"%0{i}d"
        if pattern in path:
            # Split the path into components
            parts = path.split(pattern)
            if len(parts) == 2:  # We have a base name and extension
                base_name = parts[0]
                extension = os.path.splitext(parts[1])[1]
                return pattern, i, base_name, extension
    
    # If no standard pattern found, try to detect custom patterns
    # Look for a sequence of digits that could be a frame number
    match = re.search(r'(\D+)(\d+)(\.[^.]+)$', path)
    if match:
        base_name = match.group(1)
        frame_num = match.group(2)
        extension = match.group(3)
        padding = len(frame_num)
        pattern = f"%0{padding}d"
        return pattern, padding, base_name, extension
    
    return None, None, None, None

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
        
        # Detect sequence pattern and components
        pattern, digits, base_name, extension = detect_sequence_pattern(sequence_path)
        if pattern:
            self.frame_pattern = pattern
            self.padding = digits
            self.base_name = base_name
            self.extension = extension
            # Create output base name by adding "_mask_" before the frame number
            self.output_base_name = f"{base_name}mask_"
        else:
            raise ValueError("Input is not a sequence. Expected frame pattern or custom naming pattern")
        
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
            padded_frame = f"{frame_num:0{self.padding}d}"
            
            # Construct frame path
            if self.frame_pattern:
                frame_path = self.sequence_path.replace(self.frame_pattern, padded_frame)
            else:
                # For custom patterns, construct the path manually
                frame_path = f"{self.base_name}{padded_frame}{self.extension}"
                frame_path = os.path.join(self.input_path_folder, frame_path)
            
            if os.path.exists(frame_path):
                frame_paths.append(frame_path)
            
            # Update progress
            pbar.update(1)
            
        pbar.close()
        return sorted(frame_paths)
    
    def get_output_path(self, frame_number: int) -> str:
        """Generate output path for a given frame number."""
        padded_frame = f"{frame_number:0{self.padding}d}"
        if self.frame_pattern:
            # For standard patterns, replace the pattern in the full path
            output_path = self.sequence_path.replace(self.frame_pattern, f"mask_{padded_frame}")
        else:
            # For custom patterns, construct the path manually
            output_path = f"{self.output_base_name}{padded_frame}{self.extension}"
            output_path = os.path.join(self.input_path_folder, output_path)
        return output_path
    
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