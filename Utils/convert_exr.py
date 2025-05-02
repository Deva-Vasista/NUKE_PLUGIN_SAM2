import cv2
import numpy as np

# Read the EXR file (float32)
mask = cv2.imread("flamingo_mask.exr", cv2.IMREAD_UNCHANGED)
if mask is None:
    raise Exception("Failed to read EXR file!")

# If mask is 3 channels, take one channel
if mask.ndim == 3 and mask.shape[2] == 3:
    mask = mask[:, :, 0]

# Normalize to 0-255 and convert to uint8
mask_uint8 = np.clip(mask * 255, 0, 255).astype(np.uint8)

# Save as PNG or JPEG
cv2.imwrite("flamingo_mask.png", mask_uint8)