import torch
import numpy as np
from NukeSamurai.sam2_repo.sam2.build_sam import build_sam2_video_predictor
import os
from loguru import logger

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
        self.current_state = None
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

        try:
            # If image is a numpy array, treat as single image (not sequence)
            if isinstance(image, np.ndarray):
                # Single image inference (NukeSamurai style)
                # Convert to torch tensor and normalize
                img = torch.from_numpy(image).float()
                if img.ndim == 2:
                    img = img.unsqueeze(2)
                if img.shape[-1] == 1:
                    img = img.repeat(1, 1, 3)
                if img.shape[-1] == 3:
                    img = img.permute(2, 0, 1)  # HWC to CHW
                img = img.unsqueeze(0)  # Add batch dimension
                img = img / 255.0 if img.max() > 1.0 else img
                img_mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)[:, None, None]
                img_std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)[:, None, None]
                img = (img - img_mean) / img_std
                img = img.to(self.device)
                # Run model's single image inference (simulate first frame of sequence)
                # Use model internals to get mask for single image
                # This assumes the model has a method for single image inference, otherwise use first frame logic
                # We'll use the same add_new_points_or_box logic as for the first frame
                # Create a dummy inference state
                inference_state = {
                    "images": img,
                    "num_frames": 1,
                    "offload_video_to_cpu": False,
                    "offload_state_to_cpu": False,
                    "video_height": img.shape[2],
                    "video_width": img.shape[3],
                    "device": self.device,
                    "storage_device": self.device,
                    "point_inputs_per_obj": {},
                    "mask_inputs_per_obj": {},
                    "cached_features": {},
                    "constants": {},
                    "obj_id_to_idx": {},
                    "obj_idx_to_id": {},
                    "obj_ids": [],
                    "output_dict": {"cond_frame_outputs": {}, "non_cond_frame_outputs": {}},
                    "output_dict_per_obj": {},
                    "temp_output_dict_per_obj": {},
                    "consolidated_frame_inds": {"cond_frame_outputs": set(), "non_cond_frame_outputs": set()},
                    "tracking_has_started": False,
                    "frames_already_tracked": {},
                }
                frame_idx = 0
                obj_id = 0
                frame_idx, obj_ids, masks = self.model.add_new_points_or_box(
                    inference_state,
                    frame_idx,
                    obj_id,
                    box=box
                )
                return masks.cpu().numpy()

            # If image is a sequence path, use original logic
            if isinstance(image, str) and ("%04d" in image or "%03d" in image):
                if frame_range is None:
                    raise ValueError("frame_range must be provided for sequence processing")
                # Process sequence
                masks_list = []
                if self.current_state is None:
                    self.current_state, _, _ = self.model.init_state(
                        image,
                        frame_range_min=frame_range[0] if frame_range else None,
                        frame_range_max=frame_range[1] if frame_range else None,
                        original_fps=original_fps,
                        target_fps=target_fps,
                        bits=bits
                    )
                for frame_idx, obj_ids, frame_masks in self.model.propagate_in_video(
                    self.current_state,
                    start_frame_idx=frame_range[0],
                    max_frame_num_to_track=frame_range[1] - frame_range[0] + 1
                ):
                    masks_list.append(frame_masks)
                # Stack all masks
                return torch.cat(masks_list, dim=0).cpu().numpy()
            else:
                # Single frame (torch tensor or other)
                return masks.cpu().numpy()

        except Exception as e:
            logger.error(f"Error in generate_mask: {str(e)}")
            raise

    async def generate_mask_async(self, image, bbox: list, frame_range=None, original_fps=24, target_fps=24, bits=32, progress_callback=None):
        import asyncio
        import inspect

        def sync_progress_update(progress, message=""):
            if progress_callback:
                if inspect.iscoroutinefunction(progress_callback):
                    try:
                        loop = asyncio.get_event_loop()
                        if loop.is_running():
                            asyncio.run_coroutine_threadsafe(progress_callback(progress, message), loop)
                    except RuntimeError:
                        pass  # No event loop, skip
                else:
                    try:
                        progress_callback(progress, message)
                    except Exception as e:
                        logger.error(f"Error in progress callback: {str(e)}")

        async def async_progress_update(progress, message=""):
            if progress_callback:
                if inspect.iscoroutinefunction(progress_callback):
                    await progress_callback(progress, message)
                else:
                    progress_callback(progress, message)

        # Use sync version for thread pool, async for main event loop
        loop = asyncio.get_event_loop()
        if loop.is_running():
            await async_progress_update(0, "Initializing...")
        else:
            sync_progress_update(0, "Initializing...")
        try:
            if isinstance(image, str) and ("%04d" in image or "%03d" in image) and frame_range:
                start_frame, end_frame = frame_range
                total_frames = end_frame - start_frame + 1
                def process_with_progress():
                    result = self.generate_mask(image, bbox, frame_range, original_fps, target_fps, bits)
                    sync_progress_update(100, "Processing complete")
                    return result
                result = await loop.run_in_executor(None, process_with_progress)
            else:
                result = await loop.run_in_executor(
                    None, 
                    self.generate_mask,
                    image, bbox, frame_range, original_fps, target_fps, bits
                )
                sync_progress_update(100, "Processing complete")
            return result
        except Exception as e:
            sync_progress_update(0, f"Error: {str(e)}")
            raise
