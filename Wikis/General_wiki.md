# NukeSamurai FastAPI Backend: System Overview & Design

## 1. System Purpose
This backend exposes the core functionality of the NukeSamurai plugin (object segmentation/tracking in EXR sequences) as a modern, testable, and scalable FastAPI web service. It is designed to be used by a UI (e.g., a Nuke plugin) or other clients via HTTP APIs.

---

## 2. Main Components & Code Structure

- **`src/routes/exr.py`**: FastAPI endpoints for EXR file/sequence processing.
- **`src/models/sam_wrapper.py`**: Wrapper for loading and running the SAM2 model (NukeSamurai core logic) in both single-image and sequence (tracking) modes.
- **`src/utils/sequence_handler.py`**: Utilities for handling EXR sequences.
- **`tests/`**: Pytest-based tests for all API endpoints and workflows.

---

## 3. Key Endpoints & Their Logic

### `/process_exr`
- **Purpose**: Segment a single EXR image using a bounding box and/or positive/negative points.
- **Workflow**:
  1. Receives an EXR file and prompts (bbox, points) via HTTP POST.
  2. Reads and preprocesses the EXR image.
  3. Calls `SAMProcessor.generate_mask`, which:
     - For the real model: builds a dummy inference state and calls `add_new_points_or_box` (NukeSamurai logic) to get the mask.
     - For tests: uses a mock `.predict()`.
  4. Returns the mask as JSON or EXR file.

### `/process_sequence`
- **Purpose**: Track and segment an object across a sequence of EXR frames, using bbox/points on the first frame.
- **Workflow**:
  1. Receives a sequence path, frame range, and prompts via HTTP POST.
  2. Validates the sequence and frame range.
  3. Calls `SAMProcessor.track_sequence`, which:
     - Calls `model.init_state` (NukeSamurai) to load the sequence and initialize tracking.
     - Adds prompts to the first frame with `add_new_points_or_box`.
     - Calls `propagate_in_video` to track/propagate the mask across all frames.
     - Returns a list of masks (one per frame).
  4. Returns the masks as a ZIP of EXR files or as JSON.

### Other Endpoints
- **Model loading, health check, GPU info, batch processing, progress tracking** (WebSocket): All designed for robust, scalable, and observable operation.

---

## 4. System Design & Workflow

- **Stateless API**: Each request is independent; model state is not persisted between requests.
- **Async/Threaded**: Uses FastAPI async endpoints and thread pools for model inference.
- **Testable**: Pytest suite with mock models for fast, deterministic testing.
- **Extensible**: Easy to add new endpoints, models, or features.
- **Separation of Concerns**: UI (Nuke plugin) and backend are decoupled; backend can be used by any client.

---

## 5. Comparison: Backend vs. Original NukeSamurai Plugin

| Aspect                | Original NukeSamurai (Nuke Plugin) | FastAPI Backend Version         |
|-----------------------|-------------------------------------|---------------------------------|
| **UI**                | Nuke GUI                            | Any client (e.g., Nuke, web)    |
| **Execution**         | In-process in Nuke                  | Standalone web server           |
| **Model Calls**       | Direct Python calls                 | HTTP API, async/threaded        |
| **State**             | In-memory, session-based            | Stateless per request           |
| **Testing**           | Manual, in-Nuke                     | Automated, pytest, mock support |
| **Extensibility**     | Harder (tied to Nuke)               | Easy (modular Python)           |
| **Deployment**        | Nuke-only                           | Docker, cloud, local, etc.      |

**Key Similarities:**
- Uses the same core SAM2 model and logic for segmentation/tracking.
- Supports bbox and point prompts, and true object tracking across sequences.

**Key Differences:**
- Backend is stateless, API-driven, and testable outside of Nuke.
- UI and backend are decoupled, enabling new workflows and integrations.

---

## 6. System Constraints & Considerations

- **GPU Required**: For real-time/large-scale inference, a CUDA-capable GPU is needed.
- **EXR Format**: Only EXR files/sequences are supported (matching VFX/Nuke workflows).
- **Statelessness**: No persistent session state; all prompts must be provided per request.
- **Error Handling**: API returns clear HTTP errors, but some model errors may be cryptic (e.g., CUDA OOM, file not found).
- **Performance**: For very large sequences, memory and inference time may be significant.
- **Nuke-specific Features**: Some Nuke UI features (interactive correction, progress bars) are not exposed in the API, but can be emulated by clients.

---

## 7. Expected Workflow (End-to-End)

1. **Client (e.g., Nuke plugin) loads the model** via `/models/load`.
2. **User selects a frame or sequence and provides prompts** (bbox/points).
3. **Client sends a request to `/process_exr` or `/process_sequence`**.
4. **Backend runs the model and returns the mask(s)** as JSON or EXR/ZIP.
5. **Client displays or saves the results**.

---

## 8. How to Extend or Debug

- **Add new endpoints** in `src/routes/` for new workflows.
- **Add new model logic** in `src/models/sam_wrapper.py`.
- **Add/modify tests** in `tests/` to cover new features or edge cases.
- **Check logs and debug prints** for error diagnosis.
- **Compare with NukeSamurai code** for any logic differences.

---

## 9. Summary
This backend brings the power of NukeSamurai's segmentation/tracking to any client or workflow, with modern API design, robust testing, and extensibility. It is not a 1:1 replacement for the Nuke plugin UI, but enables new automation and integration possibilities for VFX and beyond. 