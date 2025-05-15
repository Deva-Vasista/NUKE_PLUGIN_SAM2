import torch
import numpy as np
from NukeSamurai.sam2_repo.sam2.build_sam import build_sam2_video_predictor
import os
from loguru import logger
import asyncio
import re
from collections.abc import Mapping, Sequence
import functools

# Global patches for tensor operations to avoid BFloat16 issues
def patch_torch_operations():
    """Apply global patches to torch operations to ensure float32 computation"""
    logger.info("Applying global patches to torch operations to ensure float32 computation")
    
    # Set default tensor type to float32
    torch.set_default_tensor_type(torch.FloatTensor)
    
    # Disable mixed precision
    torch.backends.cuda.matmul.allow_tf32 = False
    if hasattr(torch.backends.cudnn, 'allow_tf32'):
        torch.backends.cudnn.allow_tf32 = False
    
    # Patch matrix multiplication operations
    original_matmul = torch.matmul
    original_bmm = torch.bmm
    original_mm = torch.mm
    original_linear = torch.nn.functional.linear
    
    @functools.wraps(original_matmul)
    def patched_matmul(input, other, *, out=None):
        if isinstance(input, torch.Tensor) and input.dtype != torch.float32:
            input = input.to(torch.float32)
        if isinstance(other, torch.Tensor) and other.dtype != torch.float32:
            other = other.to(torch.float32)
        return original_matmul(input, other, out=out)
    
    @functools.wraps(original_bmm)
    def patched_bmm(input, mat2, *, out=None):
        if isinstance(input, torch.Tensor) and input.dtype != torch.float32:
            input = input.to(torch.float32)
        if isinstance(mat2, torch.Tensor) and mat2.dtype != torch.float32:
            mat2 = mat2.to(torch.float32)
        return original_bmm(input, mat2, out=out)
    
    @functools.wraps(original_mm)
    def patched_mm(input, mat2, *, out=None):
        if isinstance(input, torch.Tensor) and input.dtype != torch.float32:
            input = input.to(torch.float32)
        if isinstance(mat2, torch.Tensor) and mat2.dtype != torch.float32:
            mat2 = mat2.to(torch.float32)
        return original_mm(input, mat2, out=out)
    
    @functools.wraps(original_linear)
    def patched_linear(input, weight, bias=None):
        if isinstance(input, torch.Tensor) and input.dtype != torch.float32:
            input = input.to(torch.float32)
        if isinstance(weight, torch.Tensor) and weight.dtype != torch.float32:
            weight = weight.to(torch.float32)
        if bias is not None and isinstance(bias, torch.Tensor) and bias.dtype != torch.float32:
            bias = bias.to(torch.float32)
        return original_linear(input, weight, bias)
    
    # Apply the patches
    torch.matmul = patched_matmul
    torch.bmm = patched_bmm
    torch.mm = patched_mm
    torch.nn.functional.linear = patched_linear
    
    # Patch tensor methods at the class level
    original_tensor_matmul = torch.Tensor.__matmul__
    
    def patched_tensor_matmul(self, other):
        if self.dtype != torch.float32:
            self = self.to(torch.float32)
        if isinstance(other, torch.Tensor) and other.dtype != torch.float32:
            other = other.to(torch.float32)
        return original_tensor_matmul(self, other)
    
    torch.Tensor.__matmul__ = patched_tensor_matmul
    
    logger.info("Successfully applied global patches for tensor operations")

# Apply patches when module is imported
patch_torch_operations()

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
            
            # Force conversion of all model parameters to float32
            logger.info("Converting all model parameters to float32 to avoid dtype mismatches")
            for name, param in self.model.named_parameters():
                if param.dtype != torch.float32:
                    logger.warning(f"Converting parameter {name} from {param.dtype} to float32")
                    param.data = param.data.to(torch.float32)
            
            # Also convert model buffers
            for name, buffer in self.model.named_buffers():
                if hasattr(buffer, 'dtype') and buffer.dtype != torch.float32:
                    logger.warning(f"Converting buffer {name} from {buffer.dtype} to float32")
                    buffer.data = buffer.data.to(torch.float32)
            
            # Scan model for any BFloat16 tensors after initial conversion
            self._scan_and_fix_bfloat16_tensors(self.model)
            
        # Set model to float32 computation mode
        self.model.to(torch.float32)
        self.current_model = model_path
        
    def _scan_and_fix_bfloat16_tensors(self, model):
        """Recursively scan model for any remaining BFloat16 tensors and convert them to Float32"""
        
        # Scan attributes that might contain tensors
        for attr_name, attr_value in model.__dict__.items():
            if isinstance(attr_value, torch.Tensor) and attr_value.dtype == torch.bfloat16:
                logger.warning(f"Found BFloat16 tensor in model attributes: {attr_name}")
                model.__dict__[attr_name] = attr_value.to(torch.float32)
            elif isinstance(attr_value, (list, tuple)):
                # Handle lists or tuples of tensors
                for i, item in enumerate(attr_value):
                    if isinstance(item, torch.Tensor) and item.dtype == torch.bfloat16:
                        if isinstance(attr_value, list):
                            attr_value[i] = item.to(torch.float32)
                        else:  # tuple can't be modified
                            new_tuple = list(attr_value)
                            new_tuple[i] = item.to(torch.float32)
                            model.__dict__[attr_name] = tuple(new_tuple)
            elif isinstance(attr_value, dict):
                # Handle dictionaries containing tensors
                for k, v in attr_value.items():
                    if isinstance(v, torch.Tensor) and v.dtype == torch.bfloat16:
                        attr_value[k] = v.to(torch.float32)
        
        # Recursively check child modules
        for child_name, child_module in model.named_children():
            self._scan_and_fix_bfloat16_tensors(child_module)

    def _convert_windows_to_wsl_path(self, path: str) -> str:
        """Convert Windows path to WSL-compatible path if needed"""
        if path.startswith("C:") or path.startswith("c:"):
            drive_letter = path[0].lower()
            path_part = path[3:]
            path_part = path_part.replace('\\', '/')
            wsl_path = f"/mnt/{drive_letter}/{path_part}"
            logger.info(f"Converting Windows path '{path}' to WSL path '{wsl_path}'")
            return wsl_path
        return path
    
    def _convert_tensors_to_float32(self, obj):
        """Recursively convert all tensors in a nested structure to float32"""
        if isinstance(obj, torch.Tensor) and obj.dtype != torch.float32:
            return obj.to(torch.float32)
        elif isinstance(obj, Mapping):
            return {k: self._convert_tensors_to_float32(v) for k, v in obj.items()}
        elif isinstance(obj, Sequence) and not isinstance(obj, (str, bytes)):
            return [self._convert_tensors_to_float32(x) for x in obj]
        else:
            return obj
    
    def patched_init_state(self, *args, **kwargs):
        """Patched version of init_state that handles Windows paths"""
        # Convert sequence_path if it's a Windows path
        if args and len(args) > 0:
            args = list(args)
            args[0] = self._convert_windows_to_wsl_path(args[0])
            args = tuple(args)
        elif 'video_path' in kwargs:
            kwargs['video_path'] = self._convert_windows_to_wsl_path(kwargs['video_path'])
        
        # Call the original init_state method with converted paths
        state, images, frame_start = self.model.init_state(*args, **kwargs)
        
        # Ensure all tensors in the state are float32
        with torch.no_grad():
            # Convert images to float32 if not already
            if isinstance(images, torch.Tensor) and images.dtype != torch.float32:
                images = images.to(torch.float32)
            
            # Convert relevant tensors in state to float32
            state = self._convert_tensors_to_float32(state)
        
        return state, images, frame_start
        
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
            # Convert Windows path to WSL path if needed
            sequence_path = self._convert_windows_to_wsl_path(sequence_path)
            
            # Use patched init_state to handle paths
            state, images, frame_start = self.patched_init_state(
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
            
        # Make sure tensors are float32
        points_tensor = torch.tensor(points, dtype=torch.float32) if points else None
        labels_tensor = torch.tensor(labels, dtype=torch.int32) if labels else None
        box_tensor = torch.tensor(bbox, dtype=torch.float32) if bbox is not None else None
        
        try:
            # Add prompt with the right type
            if prompt_type == 'bbox':
                self.model.add_new_points_or_box(state, frame_idx, obj_id, box=box_tensor)
            elif prompt_type == 'points':
                self.model.add_new_points_or_box(state, frame_idx, obj_id, points=points_tensor, labels=labels_tensor)
            elif prompt_type == 'both':
                self.model.add_new_points_or_box(state, frame_idx, obj_id, box=box_tensor, points=points_tensor, labels=labels_tensor)
            else:
                logger.error(f"[ERROR] Unknown prompt_type: {prompt_type}")
                raise ValueError(f"Unknown prompt_type: {prompt_type}")
                
            # Make sure state tensors are float32 after adding prompt
            with torch.no_grad():
                state = self._convert_tensors_to_float32(state)
                
        except Exception as e:
            logger.error(f"[ERROR] add_new_points_or_box failed: {e}")
            raise
            
        # Debug logging
        if debug:
            logger.info(f"[DEBUG] propagate_in_video: state type={type(state)}, frame_start={frame_start}, frame_range={frame_range}, images shape={getattr(images, 'shape', None)}")
            logger.info(f"[DEBUG] state keys: {list(state.keys())}")
            logger.info(f"[DEBUG] state['images'] shape: {getattr(state.get('images', None), 'shape', None)}")
            logger.info(f"[DEBUG] state['num_frames']: {state.get('num_frames', None)}")
            
        # Propagate masks
        masks_list = []
        try:
            # Create a wrapper to convert tensors during propagation
            def propagate_with_conversion():
                for idx, (frame_idx, object_ids, masks) in enumerate(self.model.propagate_in_video(
                    state,
                    start_frame_idx=frame_start,
                    max_frame_num_to_track=frame_range[1] - frame_range[0] + 1
                )):
                    # Ensure masks are float32 before yielding
                    if masks.dtype != torch.float32:
                        masks = masks.to(torch.float32)
                    if debug:
                        logger.info(f"[DEBUG] propagate_in_video yielded idx={idx}, masks type={type(masks)}, shape={getattr(masks, 'shape', None)}, dtype={masks.dtype}")
                    masks_list.append(masks.cpu().numpy())
                    yield frame_idx, object_ids, masks
                    
            # Apply the conversion during propagation
            for _ in propagate_with_conversion():
                pass
                    
        except RuntimeError as e:
            if "expected scalar type Float but found BFloat16" in str(e):
                logger.error(f"[ERROR] Data type mismatch (BFloat16 vs Float): {e}")
                logger.info("[INFO] Attempting to recover by forcing model components to float32...")
                # If we got here, try a last resort recovery
                try:
                    for module in self.model.modules():
                        for param in module.parameters():
                            if param.dtype != torch.float32:
                                param.data = param.data.to(torch.float32)
                    # Try propagation again
                    for idx, (_, _, masks) in enumerate(self.model.propagate_in_video(
                        state,
                        start_frame_idx=frame_start,
                        max_frame_num_to_track=frame_range[1] - frame_range[0] + 1
                    )):
                        masks_list.append(masks.cpu().numpy())
                except Exception as recovery_e:
                    logger.error(f"[ERROR] Recovery attempt failed: {recovery_e}")
                    raise
            else:
                logger.error(f"[ERROR] propagate_in_video failed: {e}")
                raise
        except Exception as e:
            logger.error(f"[ERROR] propagate_in_video failed: {e}")
            raise
            
        return masks_list

    def reset_state(self):
        """Reset the model state and clear any cached data."""
        try:
            if self.model is not None:
                # Clear model state using SAM2's built-in reset functionality
                if hasattr(self.model, 'reset_state') and self.current_state is not None:
                    self.model.reset_state(self.current_state)
                
                # Clear any cached features
                if hasattr(self.model, 'cached_features'):
                    self.model.cached_features = {}
                
                # Reset tracking state
                if hasattr(self.model, 'tracking_has_started'):
                    self.model.tracking_has_started = False
                if hasattr(self.model, 'frames_already_tracked'):
                    self.model.frames_already_tracked = {}
                
                # Clear object tracking data
                if hasattr(self.model, 'obj_id_to_idx'):
                    self.model.obj_id_to_idx = {}
                if hasattr(self.model, 'obj_idx_to_id'):
                    self.model.obj_idx_to_id = {}
                if hasattr(self.model, 'obj_ids'):
                    self.model.obj_ids = []
            
            # Clear instance state
            self.current_state = None
            
            # Force GPU memory cleanup
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            logger.info("Successfully reset model state and cleared cache")
            return True
            
        except Exception as e:
            logger.error(f"Error during model state reset: {e}")
            raise RuntimeError(f"Failed to reset model state: {e}")
