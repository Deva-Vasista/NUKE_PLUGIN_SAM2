# SAM2 API Documentation

## Base URL
```
http://localhost:8000/api/v1
```

## Endpoints

### 1. Load Model
```
POST /models/load
```

Load a specific model type into memory.

Request Body:
```json
{
    "model_type": "base"  // One of: "base", "large", "small", "tiny"
}
```

Response:
```json
{
    "status": "success",
    "message": "Model loaded successfully",
    "model_type": "base"
}
```

### 2. Process Single Frame
```
POST /process_exr
```

Query Parameters:
- `as_file` (boolean): Return mask as EXR file if true, else as list

Request Body:
```json
{
    "image_path": "/path/to/frame.exr",
    "bbox": [x1, y1, x2, y2],  // Optional
    "points_positive": [[x1,y1], [x2,y2]],  // Optional
    "points_negative": [[x1,y1], [x2,y2]],  // Optional
    "bits": "32-bit float"  // Optional, defaults to "32-bit float"
}
```

### 3. Process Sequence
```
POST /process_sequence
```

Query Parameters:
- `as_file` (boolean): Return masks as ZIP if true, else as JSON list

Request Body:
```json
{
    "sequence_path": "/path/to/frame_%04d.exr",
    "frame_range": [start_frame, end_frame],
    "prompts": [
        {
            "frame_index": 0,
            "object_id": 0,
            "bbox": [x1, y1, x2, y2],  // Optional
            "points_positive": [[x1,y1]],  // Optional
            "points_negative": [[x1,y1]]  // Optional
        }
    ],
    "bits": "32-bit float",  // Optional, defaults to "32-bit float"
    "original_fps": 24,  // Optional, defaults to input file fps
    "target_fps": 24  // Optional, defaults to original_fps
}
```

Response:
```json
{
    "status": "success",
    "task_id": "uuid-string",
    "output_dir": "Output/uuid-string",
    "zip_path": "Output/uuid-string/masks_uuid-string.zip",
    "file_count": 15,
    "frames": [0, 1, 2, ...]
}
```

### 4. Monitor Progress
```
WebSocket /ws/{task_id}
```

Connect to this WebSocket endpoint to receive progress updates:
```json
{
    "progress": 85,
    "message": "Processing frame 15/16",
    "status": "running"  // One of: "running", "completed", "failed"
}
```

### 5. Get Task Output
```
GET /output/{task_id}
```

Response:
```json
{
    "files": [
        {
            "filename": "masks_uuid-string.zip",
            "size": 1234567,
            "created": "2024-05-15T12:34:56"
        }
    ]
}
```

### 6. Download Results
```
GET /download/{filename}
```

Returns the requested file (ZIP or EXR) as a binary response.

### 7. Download All Results
```
GET /download_all/{task_id}
```

Returns a ZIP file containing all output files for the task.

### 8. Get Task Status
```
GET /status/{task_id}
```

Response:
```json
{
    "status": "completed",  // One of: "running", "completed", "failed"
    "progress": 100,
    "message": "Processing completed successfully",
    "error": "Error message if status is failed"  // Optional
}
```

### 9. Reset Model State
```
POST /reset
```

Reset the model state, clear GPU memory, and remove any cached data.

Response:
```json
{
    "status": "success",
    "message": "Model state reset successfully"
}
```

### 10. Health Check
```
GET /health
```

Check if the API server is running.

Response:
```json
{
    "status": "healthy",
    "version": "v1"
}
```

## Example Curl Commands

1. Load model:
```bash
curl -X POST "http://localhost:8000/api/v1/models/load" \
  -H "Content-Type: application/json" \
  -d '{
    "model_type": "base"
  }'
```

2. Process single frame with bbox:
```bash
curl -X POST "http://localhost:8000/api/v1/process_exr?as_file=true" \
  -H "Content-Type: application/json" \
  -d '{
    "image_path": "/path/to/frame.exr",
    "bbox": [454, 185, 500, 633],
    "bits": "32-bit float"
  }' \
  --output "mask.exr"
```

3. Process sequence with multiple objects:
```bash
curl -X POST "http://localhost:8000/api/v1/process_sequence?as_file=true" \
  -H "Content-Type: application/json" \
  -d '{
    "sequence_path": "/path/to/frame_%04d.exr",
    "frame_range": [1, 20],
    "prompts": [
      { "frame_index": 0, "object_id": 0, "points_positive": [[610,728]] },
      { "frame_index": 1, "object_id": 0, "points_positive": [[700,800]] },
      { "frame_index": 0, "object_id": 1, "bbox": [454, 185, 500, 633] },
      { "frame_index": 2, "object_id": 1, "points_negative": [[300,400]] }
    ],
    "bits": "32-bit float",
    "original_fps": 24,
    "target_fps": 24
  }' \
  --output "masks.zip"
```

4. Monitor progress with WebSocket:
```javascript
const ws = new WebSocket('ws://localhost:8000/api/v1/ws/task-id-here');
ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    console.log(`Progress: ${data.progress}% - ${data.message}`);
    if (data.status === 'completed') {
        console.log('Processing completed!');
    } else if (data.status === 'failed') {
        console.error('Processing failed:', data.error);
    }
};
```

5. Reset model state:
```bash
curl -X POST "http://localhost:8000/api/v1/reset"
```

6. Check API health:
```bash
curl "http://localhost:8000/api/v1/health"
``` 