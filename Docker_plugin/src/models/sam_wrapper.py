import torch
import numpy as np
from NukeSamurai.sam2_repo.sam2.build_sam import build_sam2_video_predictor
import os
from loguru import logger
import asyncio
import re
from collections.abc import Mapping, Sequence
import functools
import cv2

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
        
    def generate_mask(self, image, bbox=None, points_positive=None, points_negative=None, frame_range=None, original_fps=24, target_fps=24, bits=32, dimensions=None):
        """
        Generate a mask for a single image using bbox and/or positive/negative points.
        """
        if self.model is None:
            raise RuntimeError("No model loaded")
            
        # Log input dimensions
        height, width = image.shape[:2]
        logger.info(f"Input image dimensions: {width}x{height}")
        
        # Check and handle dimension mismatch if expected dimensions are provided
        if dimensions is not None:
            expected_width, expected_height = dimensions
            logger.info(f"Expected dimensions: {expected_width}x{expected_height}")
            if (width, height) != (expected_width, expected_height):
                logger.warning(f"Dimension mismatch: Expected {expected_width}x{expected_height}, got {width}x{height}")
                # Resize image to match expected dimensions
                image = cv2.resize(image, (expected_width, expected_height), interpolation=cv2.INTER_LINEAR)
                logger.info(f"Resized image to {expected_width}x{expected_height}")
        
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
        img = img.permute(2, 0, 1).unsqueeze(0)
        img = img.to(self.device)
        
        # Generate mask
        with torch.inference_mode():
            mask = self.model.predict(img, bbox, points_positive, points_negative)
            
        # Convert mask to numpy and ensure it matches input dimensions
        mask = mask.squeeze().cpu().numpy()
        if dimensions is not None:
            expected_width, expected_height = dimensions
            if mask.shape != (expected_height, expected_width):
                logger.warning(f"Resizing output mask to match expected dimensions {expected_width}x{expected_height}")
                mask = cv2.resize(mask, (expected_width, expected_height), interpolation=cv2.INTER_LINEAR)
        
        return mask

    async def generate_mask_async(self, image, bbox=None, points_positive=None, points_negative=None, frame_range=None, original_fps=24, target_fps=24, bits=32, dimensions=None):
        """Async wrapper for generate_mask"""
        return await asyncio.to_thread(
            self.generate_mask,
            image,
            bbox,
            points_positive,
            points_negative,
            frame_range,
            original_fps,
            target_fps,
            bits,
            dimensions
        )

    def track_sequence(self, sequence_path, frame_range, bbox=None, points_positive=None, points_negative=None, original_fps=24, target_fps=24, bits=32, debug=False, dimensions=None):
        """
        Track objects in a sequence of frames using SAM2.
        """
        if self.model is None:
            raise RuntimeError("No model loaded")
            
        # Convert Windows path to WSL path if needed
        sequence_path = self._convert_windows_to_wsl_path(sequence_path)
        
        # Initialize state and get first frame
        state, images, frame_start = self.patched_init_state(sequence_path, frame_range[0], frame_range[1])
        
        # Log input dimensions
        height, width = images.shape[-2:]
        logger.info(f"Input sequence dimensions: {width}x{height}")
        
        # Check and handle dimension mismatch if expected dimensions are provided
        if dimensions is not None:
            expected_width, expected_height = dimensions
            logger.info(f"Expected dimensions: {expected_width}x{expected_height}")
            if (width, height) != (expected_width, expected_height):
                logger.warning(f"Dimension mismatch: Expected {expected_width}x{expected_height}, got {width}x{height}")
                # Resize images to match expected dimensions
                images = torch.nn.functional.interpolate(
                    images,
                    size=(expected_height, expected_width),
                    mode='bilinear',
                    align_corners=False
                )
                logger.info(f"Resized images to {expected_width}x{expected_height}")
        
        # Track objects
        with torch.inference_mode():
            masks = self.model.track(state, images, bbox, points_positive, points_negative)
            
        # Convert masks to numpy and ensure they match expected dimensions
        masks = masks.cpu().numpy()
        if dimensions is not None:
            expected_width, expected_height = dimensions
            if masks.shape[-2:] != (expected_height, expected_width):
                logger.warning(f"Resizing output masks to match expected dimensions {expected_width}x{expected_height}")
                masks = np.stack([
                    cv2.resize(mask, (expected_width, expected_height), interpolation=cv2.INTER_LINEAR)
                    for mask in masks
                ])
        
        return masks

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
