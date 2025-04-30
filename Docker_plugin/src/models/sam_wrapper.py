import torch
import numpy as np
from NukeSamurai.sam2_repo.sam2.build_sam import build_sam2_video_predictor
import os

class SAMProcessor:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(SAMProcessor, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
            
        self.model = None
        self.current_model = ""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._initialized = True
        
    def load_model(self, model_path: str):
        if model_path == self.current_model:
            return
            
        # Map model types to their config paths in the sam2 package
        config_map = {
            "large": "configs/sam2.1/sam2.1_hiera_l.yaml",
            "base-plus": "configs/sam2.1/sam2.1_hiera_b+.yaml",
            "small": "configs/sam2.1/sam2.1_hiera_s.yaml",
            "tiny": "configs/sam2.1/sam2.1_hiera_t.yaml"
        }
        
        # Extract model type from the path
        model_type = model_path.split("_")[-1].split(".")[0]
        config = config_map.get(model_type, "configs/sam2.1/sam2.1_hiera_b+.yaml")
        
        # Set up the Hydra config path
        os.environ["HYDRA_CONFIG_PATH"] = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            "NukeSamurai/sam2_repo/sam2/configs"
        )
        
        with torch.inference_mode():
            self.model = build_sam2_video_predictor(
                config,  # Pass the full config path with .yaml extension
                model_path,
                device=self.device
            )
            
        self.current_model = model_path
        
    def generate_mask(self, image, bbox: list, frame_range=None, original_fps=24, target_fps=24, bits=32):
        if self.model is None:
            raise RuntimeError("No model loaded")

        # Ensure bbox is a numpy array of shape (4,) if provided
        box = None
        if bbox is not None:
            box = np.array(bbox, dtype=np.float32)
            if box.shape != (4,):
                raise ValueError(f"Box must be a list of 4 elements, got shape {box.shape} and value {box}")
            print(f"[DEBUG] Passing box to model: {box}, type: {type(box)}, shape: {box.shape}")

        if isinstance(image, str) and ("%04d" in image or "%03d" in image):
            # image is a sequence path
            if frame_range is None:
                raise ValueError("frame_range must be provided for sequence processing")
            frame_range_min, frame_range_max = frame_range
            with torch.autocast(self.device, dtype=torch.float16):
                state, _, _ = self.model.init_state(
                    image,
                    frame_range_min=frame_range_min,
                    frame_range_max=frame_range_max,
                    original_fps=original_fps,
                    target_fps=target_fps,
                    bits=bits
                )
                _, _, masks = self.model.add_new_points_or_box(
                    state, 0, 0, box=box
                )
                return masks[0].cpu().numpy()
        elif isinstance(image, str):
            # image is a file path (EXR), treat as a one-frame sequence
            with torch.autocast(self.device, dtype=torch.float16):
                state, _, _ = self.model.init_state(
                    image,
                    frame_range_min=1,
                    frame_range_max=1,
                    original_fps=1,
                    target_fps=1,
                    bits=32  # or detect from file if needed
                )
                _, _, masks = self.model.add_new_points_or_box(
                    state, 0, 0, box=box
                )
                return masks[0].cpu().numpy()
        else:
            # Original logic for numpy array
            with torch.autocast(self.device, dtype=torch.float16):
                state = self.model.init_state(image)
                _, _, masks = self.model.add_new_points_or_box(
                    state, 0, 0, box=box
                )
                return masks[0].cpu().numpy()
