import sys
import types
import shutil

# Monkeypatch nuke before any other imports
class DummyProgressTask:
    def __init__(self, *args, **kwargs): pass
    def setProgress(self, *args, **kwargs): pass
    def setMessage(self, *args, **kwargs): pass
    def isCancelled(self): return False

nuke = types.SimpleNamespace()
nuke.tprint = print
nuke.ProgressTask = DummyProgressTask
sys.modules['nuke'] = nuke

import os
import numpy as np
import cv2
import torch
import gc
from NukeSamurai.sam2_repo.sam2.build_sam import build_sam2_video_predictor
from NukeSamurai.sam2_repo.sam2.utils.misc import ImgSequences

# ---- HARDCODED PARAMETERS ----
# Set these to your test files and model
EXR_SEQUENCE_PATH = "/home/mappinga/projects/NPP/Test_files/frames/frame_%04d.exr"  # e.g. "/home/user/frames/myseq_%04d.exr"
FRAME_RANGE = [1, 50]  # [start, end+1]
BIT_DEPTH = "32-bit float"  # or "16-bit fixed", etc.
MODEL_PATH = "NukeSamurai/sam2_repo/checkpoints/sam2.1_hiera_tiny.pt"
MODEL_CFG = "configs/sam2.1/sam2.1_hiera_t.yaml"
BBOX = (468,108,956,967)  # (x, y, x+w, y+h) - set to your test region
OUTPUT_DIR = "debug_masks_out"
IMAGE_SIZE = 1024  # or 512, 768, etc. (should match model)
ORIGINAL_FPS = 24
TARGET_FPS = 24

# ---- MULTI-OBJECT, MULTI-FRAME PROMPTS ----
OBJECTS = [
    {"frame_index": 0, "object_id": 0, "points_positive": [(702, 602)], "points_negative": [(622, 483)]},  # frame 0, obj 1
    {"frame_index": 19, "object_id": 1, "points_positive": [(1689, 216)]},  # frame 19, obj 1
]

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---- LOAD SEQUENCE ----
print("[DEBUG] Loading EXR sequence...")
seq = ImgSequences(
    path=EXR_SEQUENCE_PATH,
    frame_range_min=FRAME_RANGE[0],
    frame_range_max=FRAME_RANGE[1],
    original_fps=ORIGINAL_FPS,
    target_fps=TARGET_FPS,
    bits=BIT_DEPTH,
    image_size=IMAGE_SIZE
)
images, video_height, video_width, frame_start = seq.ReadSequence()
print(f"[DEBUG] Loaded {images.shape[0]} frames, size: {video_width}x{video_height}")

# ---- LOAD MODEL ----
print("[DEBUG] Loading SAM2 model...")
predictor = build_sam2_video_predictor(MODEL_CFG, MODEL_PATH, device="cuda:0")

# ---- INIT STATE ----
print("[DEBUG] Initializing model state...")
state, images, frame_start = predictor.init_state(
    EXR_SEQUENCE_PATH,
    offload_video_to_cpu=True,
    frame_range_min=FRAME_RANGE[0],
    frame_range_max=FRAME_RANGE[1],
    original_fps=ORIGINAL_FPS,
    target_fps=TARGET_FPS,
    bits=BIT_DEPTH,
)

# ---- ADD PROMPTS FOR ALL OBJECTS AT SPECIFIED FRAMES ----
print(f"[DEBUG] Adding prompts for {len(OBJECTS)} objects at their respective frames")
for obj in OBJECTS:
    obj_id = obj["object_id"]
    frame_idx = obj["frame_index"]
    points_positive = obj.get("points_positive", [])
    points_negative = obj.get("points_negative", [])
    points = points_positive + points_negative
    labels = [1] * len(points_positive) + [0] * len(points_negative)
    bbox = obj.get("bbox")
    print(f"[DEBUG] Adding prompt for obj_id={obj_id} at frame_idx={frame_idx}: points+labels={list(zip(points, labels))} bbox={bbox}")
    predictor.add_new_points_or_box(state, box=bbox, points=points, labels=labels, frame_idx=frame_idx, obj_id=obj_id)

def extract_masks_from_model_output(masks):
    """Handle both 7 and 9 outputs from the model's SAM head."""
    # If masks is a tuple of outputs, extract the mask tensor
    if isinstance(masks, (tuple, list)):
        # Try to find the mask tensor (should be a torch.Tensor with 4 dims)
        for m in masks:
            if isinstance(m, torch.Tensor) and m.ndim == 4:
                return m
        # Fallback: return the last tensor
        for m in reversed(masks):
            if isinstance(m, torch.Tensor):
                return m
        raise ValueError("Could not find mask tensor in model outputs")
    return masks

# ---- PROPAGATE AND COMBINE MASKS ----
start_frame_idx = 0
max_frame_num_to_track = FRAME_RANGE[1] - 1
print(f"[DEBUG] Propagating tracking/segmentation for all objects... start_frame_idx={start_frame_idx}, max_frame_num_to_track={max_frame_num_to_track}")
frame_masks = {}
for frame_idx, object_ids, masks in predictor.propagate_in_video(state, start_frame_idx=start_frame_idx, max_frame_num_to_track=max_frame_num_to_track):
    if frame_idx not in frame_masks:
        frame_masks[frame_idx] = {}
    masks_tensor = extract_masks_from_model_output(masks)
    for obj_id, mask in zip(object_ids, masks_tensor):
        mask_np = mask[0].cpu().numpy() > 0.0
        frame_masks[frame_idx][obj_id] = mask_np

# ---- COMBINE MASKS FROM ALL OBJECTS PER FRAME ----
print("[DEBUG] Combining masks from all objects into single EXR per frame...")
for frame_idx in sorted(frame_masks.keys()):
    combined_mask = np.zeros((video_height, video_width), dtype=np.uint8)
    for obj_id, mask_np in frame_masks[frame_idx].items():
        combined_mask[mask_np] = obj_id + 1
    mask_img = np.stack([combined_mask]*3, axis=-1).astype("float32")
    out_path = os.path.join(OUTPUT_DIR, f"mask_{frame_idx:04d}.exr")
    cv2.imwrite(out_path, mask_img)
    print(f"[DEBUG] Saved combined mask: {out_path}")

print("[DEBUG] Done. Check output masks and console for errors.")
gc.collect()
torch.cuda.empty_cache() 