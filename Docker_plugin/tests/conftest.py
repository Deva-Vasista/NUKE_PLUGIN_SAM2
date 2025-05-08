import pytest
from fastapi.testclient import TestClient
from pathlib import Path
import sys
import os
import numpy as np
import OpenEXR
import Imath
import torch
from unittest.mock import MagicMock

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.main import app
from src.models.sam_wrapper import SAMProcessor

@pytest.fixture(scope="session")
def mock_sam_model():
    """Create a mock SAM model for testing."""
    class MockSAM:
        def __init__(self):
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.image_size = 256  # SAM's internal size
            
        def init_state(self, video_path, frame_range_min=None, frame_range_max=None, original_fps=None, target_fps=None, bits=None):
            """Mock init_state method."""
            # Raise error if file/sequence does not exist
            if isinstance(video_path, str) and not os.path.exists(video_path.replace('%04d', '0001')):
                raise FileNotFoundError(f"File not found: {video_path}")
            if frame_range_min is not None and frame_range_max is not None:
                if frame_range_min < 1 or frame_range_max < frame_range_min:
                    raise ValueError("Invalid frame range")
            mock_state = {
                "images": [torch.ones(3, 256, 256)],
                "num_frames": 1,
                "device": self.device,
                "video_height": 256,
                "video_width": 256,
                "storage_device": self.device,
                "point_inputs_per_obj": {},
                "mask_inputs_per_obj": {},
                "cached_features": {},
                "constants": {},
                "obj_id_to_idx": {},
                "obj_idx_to_id": {},
                "obj_ids": [],
                "output_dict": {
                    "cond_frame_outputs": {},
                    "non_cond_frame_outputs": {},
                },
                "output_dict_per_obj": {},
                "temp_output_dict_per_obj": {},
                "consolidated_frame_inds": {
                    "cond_frame_outputs": set(),
                    "non_cond_frame_outputs": set(),
                },
                "tracking_has_started": False,
                "frames_already_tracked": {}
            }
            return mock_state, [torch.ones(3, 256, 256)], 0
            
        def add_new_points_or_box(self, state, frame_idx, obj_id, points=None, labels=None, box=None):
            """Mock add_new_points_or_box method."""
            h, w = 256, 256
            masks = torch.ones((1, 1, h, w))
            return frame_idx, [obj_id], masks
            
        def propagate_in_video(self, state, start_frame_idx=None, max_frame_num_to_track=None, reverse=False):
            """Mock propagate_in_video method."""
            h, w = state["video_height"], state["video_width"]
            masks = torch.ones((1, 1, h, w))
            yield 0, state["obj_ids"], masks
            
        def predict(self, image, bbox=None, points_positive=None, points_negative=None, **kwargs):
            # Return a dummy mask with the same height/width as the input image
            if isinstance(image, np.ndarray):
                h, w = image.shape[:2]
            else:
                h, w = 256, 256  # fallback
            return np.ones((h, w), dtype=np.float32)
            
    return MockSAM()

@pytest.fixture(scope="session")
def mock_processor(mock_sam_model):
    """Create a mock SAM processor for testing."""
    processor = SAMProcessor()
    processor.model = mock_sam_model
    processor.current_model = "mock_model"
    processor.device = "cuda" if torch.cuda.is_available() else "cpu"
    return processor

@pytest.fixture
def client(mock_processor):
    """Create a test client with mocked SAM processor."""
    # Override the processor in the routes
    from src.routes import exr, models, batch
    exr.processor = mock_processor
    models.processor = mock_processor
    batch.processor = mock_processor
    
    app.dependency_overrides = {}
    return TestClient(app)

@pytest.fixture
def sample_exr():
    """Create a sample EXR file for testing."""
    # Create a temporary directory for test files
    test_dir = Path(__file__).parent / "test_files"
    test_dir.mkdir(exist_ok=True)
    
    # Create a simple test image
    width, height = 100, 100
    image = np.ones((height, width, 3), dtype=np.float32)
    
    # Create EXR header
    header = OpenEXR.Header(width, height)
    header['channels'] = {
        'R': Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT)),
        'G': Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT)),
        'B': Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT))
    }
    
    # Save test EXR file
    test_file = test_dir / "test.exr"
    out = OpenEXR.OutputFile(str(test_file), header)
    out.writePixels({
        'R': image[:,:,0].tobytes(),
        'G': image[:,:,1].tobytes(),
        'B': image[:,:,2].tobytes()
    })
    out.close()
    
    yield test_file
    
    # Cleanup
    test_file.unlink(missing_ok=True)
    test_dir.rmdir()

@pytest.fixture
def sample_sequence():
    """Create a sample EXR sequence for testing."""
    test_dir = Path(__file__).parent / "test_files" / "sequence"
    test_dir.mkdir(exist_ok=True, parents=True)
    
    files = []
    for i in range(1, 4):  # Create 3 frames
        width, height = 100, 100
        image = np.ones((height, width, 3), dtype=np.float32) * (i / 3)
        
        header = OpenEXR.Header(width, height)
        header['channels'] = {
            'R': Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT)),
            'G': Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT)),
            'B': Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT))
        }
        
        test_file = test_dir / f"frame_{i:04d}.exr"
        out = OpenEXR.OutputFile(str(test_file), header)
        out.writePixels({
            'R': image[:,:,0].tobytes(),
            'G': image[:,:,1].tobytes(),
            'B': image[:,:,2].tobytes()
        })
        out.close()
        files.append(test_file)
    
    yield test_dir / "frame_%04d.exr"
    
    # Cleanup
    for file in files:
        file.unlink(missing_ok=True)
    test_dir.rmdir()
    test_dir.parent.rmdir()

@pytest.fixture
def gpu_manager():
    """Get the GPU manager instance."""
    from src.utils.gpu_manager import GPUManager
    return GPUManager()

@pytest.fixture
def progress_tracker():
    """Get the progress tracker instance."""
    from src.utils.progress_tracker import ProgressTracker
    return ProgressTracker() 