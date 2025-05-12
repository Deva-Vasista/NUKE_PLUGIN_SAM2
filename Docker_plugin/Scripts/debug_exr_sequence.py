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
FRAME_RANGE = [1, 20]  # [start, end+1]
BIT_DEPTH = "32-bit float"  # or "16-bit fixed", etc.
MODEL_PATH = "NukeSamurai/sam2_repo/checkpoints/sam2.1_hiera_large.pt"
MODEL_CFG = "configs/samurai/sam2.1_hiera_l.yaml"
BBOX = (468,108,956,967)  # (x, y, x+w, y+h) - set to your test region
OUTPUT_DIR = "debug_masks_out"
IMAGE_SIZE = 1024  # or 512, 768, etc. (should match model)
ORIGINAL_FPS = 24
TARGET_FPS = 24

# ---- MULTI-OBJECT PROMPTS ----
# List of objects, each with its own bbox and/or points
OBJECTS = [
    {"obj_id": 0, "bbox": (454, 185, 500, 633), "points_positive": [(610, 728)], "points_negative": [(538, 608)]},
    {"obj_id": 1, "bbox": (100, 200, 300, 400), "points_positive": [(150, 250)], "points_negative": [(120, 220)]},
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

# ---- PROPAGATE FOR EACH OBJECT SEPARATELY ----
print("[DEBUG] Propagating tracking/segmentation for each object separately...")
temp_dirs = []
for obj in OBJECTS:
    obj_id = obj["obj_id"]
    bbox = obj.get("bbox")
    points_positive = obj.get("points_positive", [])
    points_negative = obj.get("points_negative", [])
    points = points_positive + points_negative
    labels = [1] * len(points_positive) + [0] * len(points_negative)
    print(f"[DEBUG] Processing obj_id={obj_id}: bbox={bbox}, points+labels={list(zip(points, labels))}")
    # Re-initialize state for each object
    state, images, frame_start = predictor.init_state(
        EXR_SEQUENCE_PATH,
        offload_video_to_cpu=True,
        frame_range_min=FRAME_RANGE[0],
        frame_range_max=FRAME_RANGE[1],
        original_fps=ORIGINAL_FPS,
        target_fps=TARGET_FPS,
        bits=BIT_DEPTH,
    )
    # Add prompt for this object
    if bbox and points:
        predictor.add_new_points_or_box(state, box=bbox, points=points, labels=labels, frame_idx=0, obj_id=obj_id)
    elif bbox:
        predictor.add_new_points_or_box(state, box=bbox, frame_idx=0, obj_id=obj_id)
    elif points:
        predictor.add_new_points_or_box(state, points=points, labels=labels, frame_idx=0, obj_id=obj_id)
    else:
        raise ValueError(f"No prompt provided for obj_id={obj_id}")
    # Propagate and save masks for this object
    obj_dir = os.path.join(OUTPUT_DIR, f"obj_{obj_id}")
    temp_dirs.append(obj_dir)
    os.makedirs(obj_dir, exist_ok=True)
    for frame_idx, object_ids, masks in predictor.propagate_in_video(state):
        for oid, mask in zip(object_ids, masks):
            if oid != obj_id:
                continue
            mask_np = mask[0].cpu().numpy() > 0.0
            mask_img = np.stack([mask_np]*3, axis=-1).astype("float32")
            out_path = os.path.join(obj_dir, f"mask_{frame_idx:04d}.exr")
            cv2.imwrite(out_path, mask_img)
            print(f"[DEBUG] Saved mask for obj_id={obj_id}, frame={frame_idx}: {out_path}")

# ---- COMBINE MASKS FROM ALL OBJECTS PER FRAME ----
print("[DEBUG] Combining masks from all objects into single EXR per frame...")
for frame_idx in range(FRAME_RANGE[1] - FRAME_RANGE[0]):
    combined_mask = np.zeros((video_height, video_width), dtype=np.uint8)
    for obj in OBJECTS:
        obj_id = obj["obj_id"]
        obj_dir = os.path.join(OUTPUT_DIR, f"obj_{obj_id}")
        mask_path = os.path.join(obj_dir, f"mask_{frame_idx:04d}.exr")
        if os.path.exists(mask_path):
            mask_img = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
            if mask_img is not None:
                mask_bin = mask_img[..., 0] > 0.0
                combined_mask[mask_bin] = obj_id + 1
    mask_img = np.stack([combined_mask]*3, axis=-1).astype("float32")
    out_path = os.path.join(OUTPUT_DIR, f"mask_{frame_idx:04d}.exr")
    cv2.imwrite(out_path, mask_img)
    print(f"[DEBUG] Saved combined mask: {out_path}")

# ---- CLEAN UP TEMP OBJECT MASK DIRS ----
for obj_dir in temp_dirs:
    shutil.rmtree(obj_dir)

print("[DEBUG] Done. Check output masks and console for errors.")
gc.collect()
torch.cuda.empty_cache() 