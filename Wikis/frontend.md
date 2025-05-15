# SAM2 Nuke Plugin Documentation

## Overview
The SAM2 Nuke plugin provides a user-friendly interface to the Segment Anything Model 2 (SAM2) for video segmentation. It allows users to create masks for objects across video sequences using various selection tools.

## Installation
1. Copy the plugin files to your Nuke plugins directory
2. Ensure the SAM2 API server is running (see API.md)
3. The plugin will be available in Nuke under the "Other" menu as "SAM2"

## Node Interface

### Input/Output Settings
- **File Path**: Path to the input image sequence
- **Update Path**: Button to update the file path from the connected Read node
- **Frame Range**: Start and end frames to process
- **Output Frame Rate**: Target frame rate for the output sequence
- **Model Type**: Choose between different SAM2 models:
  - `base`: Standard model balancing quality and speed
  - `large`: Higher quality but slower processing
  - `small`: Faster processing with slightly lower quality
  - `tiny`: Fastest processing, suitable for quick tests

### Object Selection
- **Object ID**: Select which object to work on (0-10)
  - Each object gets its own mask in the output
  - Use this to segment multiple objects in the same sequence

### Selection Tools
- **Create Selection**: Opens the selection window with the following controls:
  - Left click and drag: Draw bounding box
  - 'p' key: Add positive point
  - 'n' key: Add negative point
  - Left/Right arrows: Change frame
  - 'z' key: Undo last action
  - 'r' key: Reset current frame
  - 'q' key: Finish selection
- **Clear All Selections**: Remove all selections for the current object

### Output Controls
- **File Type**: Choose between EXR sequence or MP4 output
- **Output Path**: Path where the mask files will be saved
  - For EXR sequences, use frame padding (e.g., `####` or `###`)
- **Generate Mask**: Process the sequence with current selections

### Advanced Controls
- **Reset Model State**: Clear model state and GPU memory
  - Use this if you encounter any issues with the model
  - Also resets the current object ID to 0

### Status Display
- **Status Message**: Shows current processing status and progress

## Usage Guide

1. **Basic Workflow**:
   ```
   Read Node -> SAM2 Node -> Write Node (optional)
   ```

2. **Creating Masks**:
   a. Connect a Read node to the SAM2 node
   b. Click "Update Path" to load the sequence
   c. Set frame range and output settings
   d. Click "Create Selection" and use the tools to mark the object
   e. Click "Generate Mask" to process the sequence

3. **Working with Multiple Objects**:
   a. Set "Object ID" to 0 for the first object
   b. Make selections for the first object
   c. Increment "Object ID" for each new object
   d. Make selections for additional objects
   e. Generate masks - each object will get its own mask

4. **Selection Tips**:
   - Use bounding boxes for rough selections
   - Add positive points inside the object
   - Add negative points outside the object
   - Make selections on multiple frames for better tracking
   - Use fewer points for faster processing

5. **Troubleshooting**:
   - If the model seems stuck, use "Reset Model State"
   - If selections aren't working, try clearing all and starting over
   - Check the status message for error information
   - Ensure the API server is running (see API.md)

## Performance Tips

1. **Model Selection**:
   - Start with the `tiny` model for quick tests
   - Use `base` model for production quality
   - `large` model for highest quality but slower processing

2. **Memory Management**:
   - Clear selections when switching between objects
   - Use "Reset Model State" when changing sequences
   - Close the selection window when not in use

3. **Processing Speed**:
   - Fewer selections generally means faster processing
   - Bounding boxes are faster than point selections
   - Consider using a lower output frame rate for faster results

## Known Limitations

1. Input Format:
   - Only supports image sequences (no MP4 input)
   - Recommended formats: EXR, JPEG, PNG, TIFF

2. Performance:
   - Processing time increases with:
     - Higher resolution images
     - More frames
     - More objects
     - More selection points

3. Memory:
   - Large sequences may require significant GPU memory
   - Use "Reset Model State" to free up memory

## Error Messages

Common error messages and solutions:

1. "API server not running":
   - Check if the SAM2 API server is running
   - Verify the server URL in the plugin settings

2. "Failed to load model":
   - Check GPU memory availability
   - Try using a smaller model
   - Reset model state and try again

3. "No selections made":
   - Add at least one selection before generating masks
   - Check if selections were cleared accidentally

4. "Invalid frame range":
   - Ensure frame range is within the input sequence
   - Check if frame numbers are valid 