# NukeSamurai Plugin Guide

## Overview
NukeSamurai is a Nuke plugin that integrates Meta's SAM2 (Segment Anything Model 2) for automatic rotoscoping in Nuke. The plugin allows users to generate masks for image sequences by specifying a bounding box on the first frame, which is then propagated through the sequence using SAM2's AI capabilities.

## Architecture

### Core Components
1. **Nuke Node Interface** (`nuke_samurai.py`)
   - Creates a custom Nuke node with UI controls
   - Handles user input (frame ranges, model selection, file paths)
   - Manages the bounding box creation interface

2. **SAM2 Integration** (`demo.py`)
   - Handles the core mask generation functionality
   - Integrates with SAM2 model for mask prediction
   - Manages frame sequence processing and output generation

3. **Model Management**
   - Supports multiple SAM2 model variants (Large, Base+, Small, Tiny)
   - Handles model loading and GPU memory management

## How It Works

### 1. Node Creation and Setup
When a user creates a NukeSamurai node:
```python
def CreateSamuraiNode():
    # Creates a NoOp node with custom knobs for:
    # - File path selection
    # - Frame range
    # - Model type selection
    # - Output settings
    # - Bounding box creation
```

### 2. Bounding Box Creation
```python
class BoundingBox:
    # Opens a CV2 window showing the first frame
    # User draws a bounding box
    # Coordinates are stored for mask generation
```

### 3. Mask Generation Process
1. **Input Preparation**
   - Reads input image sequence
   - Validates frame range and file formats
   - Sets up output paths

2. **Model Processing**
   ```python
   def main():
       # 1. Loads the selected SAM2 model
       # 2. Initializes the video predictor
       # 3. Processes first frame with bounding box
       # 4. Propagates mask through sequence
       # 5. Saves output masks
   ```

3. **Output Generation**
   - Supports EXR and MP4 output formats
   - Creates Nuke Read nodes for the generated masks

## Key Features

### 1. Multi-format Support
- Input: Image sequences (EXR, JPG, PNG, TIFF)
- Output: EXR sequences or MP4 video

### 2. GPU Acceleration
- Uses CUDA for model inference
- Includes memory management and cleanup
```python
torch.cuda.empty_cache()
gc.collect()
```

### 3. Progress Tracking
- Shows progress bar during mask generation
- Allows cancellation of ongoing processes

### 4. Flexible Configuration
- Multiple model options (Large, Base+, Small, Tiny)
- Adjustable frame rates and ranges
- Custom output paths and formats

## Threading and Performance

The plugin uses threading to prevent Nuke from freezing during processing:
```python
childThread = threading.Thread(target=main, args=(...))
childThread.start()
```

## Important Classes and Functions

### 1. InputInfos
- Manages input file metadata
- Handles FPS and bit depth detection

### 2. BoundingBox
- Manages the interactive bounding box creation
- Stores coordinates for mask generation

### 3. Main Processing
```python
def main(video_path, video_output_path, bbox_coord, ...):
    # 1. Sets up SAM2 predictor
    # 2. Processes frames
    # 3. Generates and saves masks
```

## Usage Flow

1. Create NukeSamurai node
2. Connect to image sequence
3. Set frame range and model type
4. Draw bounding box on first frame
5. Generate masks
6. Access results in Nuke

## Technical Requirements

- CUDA-capable GPU
- Python 3.x
- PyTorch
- OpenCV
- Nuke 13+

## Common Operations

### 1. Updating File Paths
```python
def UpdatePath():
    # Updates file path when input changes
```

### 2. Generating Masks
```python
def GenerateMask():
    # Validates inputs
    # Sets up processing parameters
    # Starts mask generation in separate thread
```

## Error Handling

The plugin includes checks for:
- Invalid file paths
- Unsupported formats
- Missing frame numbers
- GPU memory issues
- Processing errors

## Future Dockerization Considerations

For dockerizing this plugin:

1. **Split Architecture**
   - Keep UI and file handling in Nuke
   - Move SAM2 processing to Docker container

2. **API Requirements**
   - Need endpoints for:
     - Model selection
     - Frame processing
     - Mask retrieval

3. **Data Flow**
   - Nuke → Docker: Send frames and bbox
   - Docker → Nuke: Return mask data

4. **Performance Considerations**
   - Handle large file transfers
   - Manage GPU resources
   - Consider batch processing 