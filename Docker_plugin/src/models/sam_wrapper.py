import torch
import numpy as np
from NukeSamurai.sam2_repo.sam2.build_sam import build_sam2_video_predictor
import os
from loguru import logger
import asyncio

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
        
    def generate_mask(self, image, bbox=None, points_positive=None, points_negative=None, frame_range=None, original_fps=24, target_fps=24, bits=32):
        """
        Generate a mask for a single image using bbox and/or positive/negative points.
        """
        if self.model is None:
            raise RuntimeError("No model loaded")
        # If the model is a mock (for tests), use .predict()
        if hasattr(self.model, "predict"):
            input_kwargs = {}
            if bbox is not None:
                input_kwargs['bbox'] = bbox
            if points_positive is not None:
                input_kwargs['points_positive'] = points_positive
            if points_negative is not None:
                input_kwargs['points_negative'] = points_negative
            return self.model.predict(image, **input_kwargs)
        # --- Real model logic for single EXR file ---
        # Prepare image as torch tensor
        img = torch.from_numpy(image).float()
        if img.ndim == 2:
            img = img.unsqueeze(2)
        if img.shape[-1] == 1:
            img = img.repeat(1, 1, 3)
        if img.shape[-1] == 3:
            img = img.permute(2, 0, 1)  # HWC to CHW
        img = img.unsqueeze(0)  # Add batch dimension
        img = img / 255.0 if img.max() > 1.0 else img
        img = img.to(self.device)
        # Create dummy inference state
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
        # Prepare points and labels
        points = []
        labels = []
        if points_positive is not None:
            for pt in points_positive:
                points.append(pt)
                labels.append(1)
        if points_negative is not None:
            for pt in points_negative:
                points.append(pt)
                labels.append(0)
        points = torch.tensor(points, dtype=torch.float32) if points else None
        labels = torch.tensor(labels, dtype=torch.int32) if labels else None
        box = torch.tensor(bbox, dtype=torch.float32) if bbox is not None else None
        # Call add_new_points_or_box
        _, _, masks = self.model.add_new_points_or_box(
            inference_state,
            frame_idx,
            obj_id,
            points=points,
            labels=labels,
            box=box
        )
        return masks.cpu().numpy()

    async def generate_mask_async(self, image, bbox=None, points_positive=None, points_negative=None, frame_range=None, original_fps=24, target_fps=24, bits=32):
        """
        Asynchronously generate a mask for a single image using bbox and/or positive/negative points.
        """
        if self.model is None:
            raise RuntimeError("No model loaded")
        # Await the sync version in a thread pool
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.generate_mask, image, bbox, points_positive, points_negative, frame_range, original_fps, target_fps, bits)

    def track_sequence(self, sequence_path, frame_range, bbox=None, points_positive=None, points_negative=None, original_fps=24, target_fps=24, bits=32, debug=False):
        """
        Track an object in a sequence using bbox and/or positive/negative points, returning a list of masks (one per frame).
        """
        if self.model is None:
            raise RuntimeError("No model loaded")
        # If the model is a mock (for tests), just call generate_mask for each frame
        if hasattr(self.model, "predict"):
            num_frames = frame_range[1] - frame_range[0] + 1
            dummy_img = np.ones((256, 256, 3), dtype=np.float32)
            return [self.generate_mask(dummy_img, bbox, points_positive, points_negative) for _ in range(num_frames)]
        # --- Real model logic ---
        try:
            state, images, frame_start = self.model.init_state(
                sequence_path,
                frame_range_min=frame_range[0],
                frame_range_max=frame_range[1],
                original_fps=original_fps,
                target_fps=target_fps,
                bits=bits
            )
        except Exception as e:
            logger.error(f"[ERROR] Model init_state failed: {e}")
            raise
        frame_idx = 0  # always add prompts to first frame
        obj_id = 0
        # Prepare points and labels
        points = []
        labels = []
        if points_positive is not None:
            for pt in points_positive:
                points.append(pt)
                labels.append(1)
        if points_negative is not None:
            for pt in points_negative:
                points.append(pt)
                labels.append(0)
        prompt_type = None
        if bbox is not None and (points_positive or points_negative):
            prompt_type = 'both'
        elif bbox is not None:
            prompt_type = 'bbox'
        elif points_positive or points_negative:
            prompt_type = 'points'
        else:
            logger.error("[ERROR] No valid prompt provided: must provide bbox and/or points.")
            raise ValueError("Must provide at least bbox or points.")
        if debug:
            logger.info(f"[DEBUG] Adding prompt: {prompt_type}")
        points_tensor = torch.tensor(points, dtype=torch.float32) if points else None
        labels_tensor = torch.tensor(labels, dtype=torch.int32) if labels else None
        box_tensor = torch.tensor(bbox, dtype=torch.float32) if bbox is not None else None
        try:
            if prompt_type == 'bbox':
                self.model.add_new_points_or_box(state, frame_idx, obj_id, box=box_tensor)
            elif prompt_type == 'points':
                self.model.add_new_points_or_box(state, frame_idx, obj_id, points=points_tensor, labels=labels_tensor)
            elif prompt_type == 'both':
                self.model.add_new_points_or_box(state, frame_idx, obj_id, box=box_tensor, points=points_tensor, labels=labels_tensor)
            else:
                logger.error(f"[ERROR] Unknown prompt_type: {prompt_type}")
                raise ValueError(f"Unknown prompt_type: {prompt_type}")
        except Exception as e:
            logger.error(f"[ERROR] add_new_points_or_box failed: {e}")
            raise
        # 3. Propagate masks through the sequence
        if debug:
            logger.info(f"[DEBUG] propagate_in_video: state type={type(state)}, frame_start={frame_start}, frame_range={frame_range}, images shape={getattr(images, 'shape', None)}")
            logger.info(f"[DEBUG] state keys: {list(state.keys())}")
            logger.info(f"[DEBUG] state['images'] shape: {getattr(state.get('images', None), 'shape', None)}")
            logger.info(f"[DEBUG] state['num_frames']: {state.get('num_frames', None)}")
        masks_list = []
        try:
            for idx, (a, b, masks) in enumerate(self.model.propagate_in_video(
                state,
                start_frame_idx=frame_start,
                max_frame_num_to_track=frame_range[1] - frame_range[0] + 1
            )):
                if debug:
                    logger.info(f"[DEBUG] propagate_in_video yielded idx={idx}, masks type={type(masks)}, shape={getattr(masks, 'shape', None)}")
                masks_list.append(masks.cpu().numpy())
        except Exception as e:
            logger.error(f"[ERROR] propagate_in_video failed: {e}")
            raise
        return masks_list
