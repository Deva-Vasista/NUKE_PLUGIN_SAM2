# NukeSamurai EXR Sequence Processing vs. SAM2 Reference Implementation

## 1. NukeSamurai Plugin: EXR Sequence Workflow (Step-by-Step)

### User Workflow (in Nuke)
1. **User loads an EXR sequence** into Nuke and attaches the NukeSamurai node.
2. **User sets frame range, output path, model type, and output type (EXR/MP4)** via the node UI (`CreateSamuraiNode` in `scripts/nuke_samurai.py`).
3. **User draws a bounding box** on the first frame using OpenCV UI (`BoundingBox.getBbox`).
4. **User clicks 'Generate Mask'**. This triggers `GenerateMask`, which:
   - Gathers all parameters (input path, output path, bbox, frame range, fps, bit depth, model type).
   - Launches a new thread to run the main processing pipeline (`main` in `scripts/demo.py`).

### Backend Processing (`scripts/demo.py`)
5. **Model and config are selected** based on user choice (e.g., `sam2.1_hiera_large.pt`).
6. **Input sequence is parsed**:
   - The EXR sequence path is parsed for frame numbers and bit depth.
   - The first frame is loaded to get image size.
7. **Model is loaded** using `build_sam2_video_predictor`.
8. **Frames are loaded and normalized** using `ImgSequences.ReadSequence` (in `sam2_repo/sam2/utils/misc.py`):
   - All EXR files in the folder are found and sorted.
   - Only frames in the user-specified range are loaded.
   - Each frame is read with OpenCV (`cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH`), resized, and normalized according to bit depth.
   - Frames are stacked into a tensor and normalized by mean/std.
9. **Model state is initialized** with the loaded frames.
10. **Bounding box prompt is added** to the first frame (`add_new_points_or_box`).
11. **Tracking/segmentation is propagated** across the sequence (`propagate_in_video`).
12. **Masks are post-processed and saved**:
    - For each frame, the mask is converted to a color image and saved as EXR (or written to MP4).
    - Output EXR files are named according to the output path template.
13. **Nuke Read node is created** to load the output sequence back into the Nuke script.

## 2. Reference SAM2 Implementation (Meta, [github.com/facebookresearch/sam2](https://github.com/facebookresearch/sam2))

- **Core logic is nearly identical** for video/sequence processing:
  1. User (or script) loads a sequence of images (typically JPEG, but EXR is supported if OpenCV is built with OpenEXR).
  2. Frames are loaded, resized, and normalized (see `ImgSequences` and `load_video_frames_from_*` in `sam2/utils/misc.py`).
  3. Model state is initialized (`SAM2VideoPredictor.init_state`).
  4. User provides prompts (box/points) on a frame.
  5. Model propagates segmentation/tracking across the sequence (`propagate_in_video`).
  6. Masks are returned as tensors or saved as images (user handles output format).

- **EXR support**: The reference code supports EXR via OpenCV if the environment variable `OPENCV_IO_ENABLE_OPENEXR=1` is set (see top of `utils/misc.py`).
- **No Nuke integration**: The reference implementation does not handle Nuke nodes, UI, or direct EXR output for VFX pipelines.

## 3. Key Differences & Special Handling in NukeSamurai

- **NukeSamurai is tightly integrated with Nuke**:
  - Handles Nuke node creation, UI, and progress reporting.
  - Automatically creates Read nodes for output.
- **EXR sequence handling is explicit**:
  - Handles bit depth normalization for EXR (8/10/12/14/16/32-bit float).
  - Ensures correct frame range and naming for VFX workflows.
- **User interaction is via Nuke UI and OpenCV window** for bbox selection.
- **Output is always compatible with Nuke** (EXR/MP4, correct frame numbering).
- **Threading and GPU memory management** are handled to avoid blocking Nuke.

## 4. Are We Doing the Same Thing?

- **Yes, the core segmentation/tracking logic is the same**: Both use the same SAM2 model and processing pipeline for video/sequence segmentation.
- **NukeSamurai adds VFX pipeline glue**: The main difference is the integration with Nuke, explicit EXR/bit depth handling, and user-friendly UI for VFX artists.
- **No custom model logic**: The actual segmentation/tracking is not modified; only the data loading, prompt collection, and output handling are adapted for Nuke and EXR workflows.

## 5. References
- NukeSamurai code: `scripts/nuke_samurai.py`, `scripts/demo.py`, `sam2_repo/sam2/utils/misc.py`
- SAM2 reference: [github.com/facebookresearch/sam2](https://github.com/facebookresearch/sam2) 