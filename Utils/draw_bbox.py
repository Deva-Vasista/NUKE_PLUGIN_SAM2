import os

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

# import cv2
# import numpy as np

# # Path to your EXR file
# exr_path = "Test_files/frames/frame_0001.exr"  # Change this to your file

# # Read the EXR file (OpenCV reads as float32, shape HxWxC)
# img = cv2.imread(exr_path, cv2.IMREAD_UNCHANGED)
# if img is None:
#     raise Exception("Failed to read EXR file!")

# # If the image is multi-channel, convert to 8-bit for display
# if img.dtype == np.float32 or img.dtype == np.float64:
#     # Normalize for display
#     img_disp = img
#     if img_disp.ndim == 3 and img_disp.shape[2] > 1:
#         img_disp = img_disp[..., :3]  # Use first 3 channels if more
#     img_disp = cv2.normalize(img_disp, None, 0, 255, cv2.NORM_MINMAX)
#     img_disp = img_disp.astype(np.uint8)
# else:
#     img_disp = img

# # For grayscale, convert to BGR for display
# if img_disp.ndim == 2:
#     img_disp = cv2.cvtColor(img_disp, cv2.COLOR_GRAY2BGR)

# # Clone for drawing
# clone = img_disp.copy()
# bbox = []

# def draw_rectangle(event, x, y, flags, param):
#     global bbox, clone, img_disp
#     temp = clone.copy()
#     if event == cv2.EVENT_LBUTTONDOWN:
#         bbox[:] = [(x, y)]
#     elif event == cv2.EVENT_MOUSEMOVE and len(bbox) == 1:
#         cv2.rectangle(temp, bbox[0], (x, y), (0, 255, 0), 2)
#         cv2.imshow("EXR Viewer", temp)
#     elif event == cv2.EVENT_LBUTTONUP and len(bbox) == 1:
#         bbox.append((x, y))
#         cv2.rectangle(temp, bbox[0], bbox[1], (0, 255, 0), 2)
#         cv2.imshow("EXR Viewer", temp)

# cv2.namedWindow("EXR Viewer")
# cv2.setMouseCallback("EXR Viewer", draw_rectangle)
# cv2.imshow("EXR Viewer", img_disp)

# print("Draw a bounding box with your mouse. Press any key to finish.")

# cv2.waitKey(0)
# cv2.destroyAllWindows()

# if len(bbox) == 2:
#     x1, y1 = bbox[0]
#     x2, y2 = bbox[1]
#     print(f"Bounding box coordinates: {x1},{y1},{x2},{y2}")
# else:
#     print("No bounding box drawn.") 

import cv2
import numpy as np
import matplotlib.pyplot as plt

exr_path = "Test_files/frames/frame_0001.exr"
img = cv2.imread(exr_path, cv2.IMREAD_UNCHANGED)
if img is None:
    raise Exception("Failed to read EXR file!")

if img.dtype == np.float32 or img.dtype == np.float64:
    img_disp = img
    if img_disp.ndim == 3 and img_disp.shape[2] > 1:
        img_disp = img_disp[..., :3]
    img_disp = cv2.normalize(img_disp, None, 0, 255, cv2.NORM_MINMAX)
    img_disp = img_disp.astype(np.uint8)
else:
    img_disp = img

if img_disp.ndim == 2:
    img_disp = cv2.cvtColor(img_disp, cv2.COLOR_GRAY2BGR)

# Show with matplotlib
plt.figure(figsize=(10, 8))
plt.imshow(cv2.cvtColor(img_disp, cv2.COLOR_BGR2RGB))
plt.title("Click two corners for bbox (top-left, bottom-right) or one point for a single point. Close window when done.")
print("Click two points for a bounding box, or one point for a single point. Close the window when done.")

# Get up to 2 points
points = plt.ginput(2, timeout=0)
plt.close()

if len(points) == 2:
    x1, y1 = map(int, points[0])
    x2, y2 = map(int, points[1])
    print(f"Bounding box coordinates: {x1},{y1},{x2},{y2}")
elif len(points) == 1:
    x, y = map(int, points[0])
    print(f"Single point coordinates: {x},{y}")
else:
    print("No points selected.")