import sys
import types

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
FRAME_RANGE = [1, 20    ]  # [start, end+1]
BIT_DEPTH = "32-bit float"  # or "16-bit fixed", etc.
MODEL_PATH = "NukeSamurai/sam2_repo/checkpoints/sam2.1_hiera_large.pt"
MODEL_CFG = "configs/sam2.1/sam2.1_hiera_l.yaml"
BBOX = (468,108,956,967)  # (x, y, x+w, y+h) - set to your test region
OUTPUT_DIR = "debug_masks_out"
IMAGE_SIZE = 1024  # or 512, 768, etc. (should match model)
ORIGINAL_FPS = 24
TARGET_FPS = 24

# ---- PROMPT TYPE SELECTION ----
# Set PROMPT_TYPE to 'bbox', 'points', or 'both'
PROMPT_TYPE = 'points'  # options: 'bbox', 'points', 'both'
# For points: list of (x, y) tuples
POINTS_POSITIVE = [(610, 728)]  # Example positive points
POINTS_NEGATIVE = [(538, 608)]  # Example negative points

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
state, _, _ = predictor.init_state(
    EXR_SEQUENCE_PATH,
    offload_video_to_cpu=True,
    frame_range_min=FRAME_RANGE[0],
    frame_range_max=FRAME_RANGE[1],
    original_fps=ORIGINAL_FPS,
    target_fps=TARGET_FPS,
    bits=BIT_DEPTH,
)

# ---- ADD PROMPT ----
print(f"[DEBUG] Adding prompt: {PROMPT_TYPE}")
frame_idx = 0  # usually first frame
obj_id = 0
bbox = BBOX

if PROMPT_TYPE == 'bbox':
    _, _, masks = predictor.add_new_points_or_box(state, box=bbox, frame_idx=frame_idx, obj_id=obj_id)
elif PROMPT_TYPE == 'points':
    points = POINTS_POSITIVE + POINTS_NEGATIVE
    labels = [1] * len(POINTS_POSITIVE) + [0] * len(POINTS_NEGATIVE)
    _, _, masks = predictor.add_new_points_or_box(state, points=points, labels=labels, frame_idx=frame_idx, obj_id=obj_id)
elif PROMPT_TYPE == 'both':
    points = POINTS_POSITIVE + POINTS_NEGATIVE
    labels = [1] * len(POINTS_POSITIVE) + [0] * len(POINTS_NEGATIVE)
    _, _, masks = predictor.add_new_points_or_box(state, box=bbox, points=points, labels=labels, frame_idx=frame_idx, obj_id=obj_id)
else:
    raise ValueError(f"Unknown PROMPT_TYPE: {PROMPT_TYPE}")

# ---- PROPAGATE ----
print("[DEBUG] Propagating tracking/segmentation...")
for frame_idx, object_ids, masks in predictor.propagate_in_video(state):
    for obj_id, mask in zip(object_ids, masks):
        mask = mask[0].cpu().numpy()
        mask = mask > 0.0
        mask_img = np.zeros((video_height, video_width, 3), np.uint8)
        mask_img[mask] = (255, 255, 255)
        out_path = os.path.join(OUTPUT_DIR, f"masks_{frame_idx:04d}.exr")
        cv2.imwrite(out_path, mask_img.astype("float32"))
        print(f"[DEBUG] Saved mask: {out_path}")

print("[DEBUG] Done. Check output masks and console for errors.")
gc.collect()
torch.cuda.empty_cache() 