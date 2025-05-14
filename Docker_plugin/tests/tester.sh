#!/bin/bash
set -e

OUTDIR="Output"
mkdir -p "$OUTDIR"

#loading the model
curl -X POST "http://localhost:8000/api/v1/models/load" -H "Content-Type: application/json"  -d '{"model_type": "tiny"}' 

# 1. Test /process_exr with bbox only
curl -X POST "http://localhost:8000/api/v1/process_exr?as_file=true" \
  -H "Content-Type: application/json" \
  -d '{
    "image_path": "/home/mappinga/projects/NPP/Test_files/frames/frame_0001.exr",
    "bbox": [454, 185, 500, 633],
    "bits": "32-bit float"
  }' \
  --output "$OUTDIR/mask_bbox.exr"

# 2. Test /process_exr with positive points only
curl -X POST "http://localhost:8000/api/v1/process_exr?as_file=true" \
  -H "Content-Type: application/json" \
  -d '{
    "image_path": "/home/mappinga/projects/NPP/Test_files/frames/frame_0001.exr",
    "points_positive": [[610,728],[500,600]],
    "bits": "32-bit float"
  }' \
  --output "$OUTDIR/mask_pos_points.exr"

# 3. Test /process_exr with negative points only
curl -X POST "http://localhost:8000/api/v1/process_exr?as_file=true" \
  -H "Content-Type: application/json" \
  -d '{
    "image_path": "/home/mappinga/projects/NPP/Test_files/frames/frame_0001.exr",
    "points_negative": [[300,400],[200,100]],
    "bits": "32-bit float"
  }' \
  --output "$OUTDIR/mask_neg_points.exr"

# 4. Test /process_exr with pos+neg+bbox
curl -X POST "http://localhost:8000/api/v1/process_exr?as_file=true" \
  -H "Content-Type: application/json" \
  -d '{
    "image_path": "/home/mappinga/projects/NPP/Test_files/frames/frame_0001.exr",
    "bbox": [454, 185, 500, 633],
    "points_positive": [[610,728]],
    "points_negative": [[300,400]],
    "bits": "32-bit float"
  }' \
  --output "$OUTDIR/mask_pos_neg_bbox.exr"

# 5. Test /process_sequence with multi-frame selection (same object)
curl -X POST "http://localhost:8000/api/v1/process_sequence?as_file=true" \
  -H "Content-Type: application/json" \
  -d '{
    "sequence_path": "/home/mappinga/projects/NPP/Test_files/frames/frame_%04d.exr",
    "frame_range": [1, 5],
    "prompts": [
      { "frame_index": 0, "points_positive": [[610,728]] },
      { "frame_index": 2, "points_positive": [[700,800]] }
    ],
    "bits": "32-bit float"
  }' \
  --output "$OUTDIR/masks_multiframe.zip"

# 6. Test /process_sequence with multi-object, multi-frame selection
curl -X POST "http://localhost:8000/api/v1/process_sequence?as_file=true" \
  -H "Content-Type: application/json" \
  -d '{
    "sequence_path": "/home/mappinga/projects/NPP/Test_files/frames/frame_%04d.exr",
    "frame_range": [1, 20],
    "prompts": [
      { "frame_index": 0, "object_id": 0, "points_positive": [[610,728]] },
      { "frame_index": 1, "object_id": 0, "points_positive": [[700,800]] },
      { "frame_index": 0, "object_id": 1, "bbox": [454, 185, 500, 633] },
      { "frame_index": 2, "object_id": 1, "points_negative": [[300,400]] }
    ],
    "bits": "32-bit float"
  }' \
  --output "$OUTDIR/masks_multiobj_multiframe.zip"

# 7. Test /process_sequence with reverse tracking
curl -X POST "http://localhost:8000/api/v1/process_sequence?as_file=true" \
  -H "Content-Type: application/json" \
  -d '{
    "sequence_path": "/home/mappinga/projects/NPP/Test_files/frames/frame_%04d.exr",
    "frame_range": [1, 5],
    "reverse": true,
    "prompts": [
      { "frame_index": 3, "object_id": 0, "points_positive": [[610,728]] }
    ],
    "bits": "32-bit float"
  }' \
  --output "$OUTDIR/masks_reverse_tracking.zip"

# 8. Test /models/load
curl -X POST "http://localhost:8000/api/v1/models/load" \
  -H "Content-Type: application/json" \
  -d '{ "model_type": "large" }' \
  -o "$OUTDIR/model_load.json"

# 9. Test /models/status
curl -X GET "http://localhost:8000/api/v1/models/status" \
  -o "$OUTDIR/model_status.json"

# 10. Test /models/unload
curl -X POST "http://localhost:8000/api/v1/models/unload" \
  -o "$OUTDIR/model_unload.json"

# 11. Test /health
curl -X GET "http://localhost:8000/api/v1/health" \
  -o "$OUTDIR/health.json"

# 12. Test /gpu/info
curl -X GET "http://localhost:8000/api/v1/gpu/info" \
  -o "$OUTDIR/gpu_info.json"

# 13. Test /gpu/memory
curl -X GET "http://localhost:8000/api/v1/gpu/memory" \
  -o "$OUTDIR/gpu_memory.json"

# 14. Test /gpu/clear-cache
curl -X POST "http://localhost:8000/api/v1/gpu/clear-cache" \
  -o "$OUTDIR/gpu_clear_cache.json"

# 15. Test /batch/process_batch
curl -X POST "http://localhost:8000/api/v1/batch/process_batch" \
  -H "Content-Type: application/json" \
  -d '{
    "sequences": [
      { "path": "/home/mappinga/projects/NPP/Test_files/frames/frame_%04d.exr", "frame_range": [1, 5], "bbox": "454,185,500,633" }
    ]
  }' \
  -o "$OUTDIR/batch_process.json"

# 16. Test /batch/{batch_id}/status (using a placeholder batch_id)
curl -X GET "http://localhost:8000/api/v1/batch/123e4567-e89b-12d3-a456-426614174000/status" \
  -o "$OUTDIR/batch_status.json"

echo "All tests complete. Outputs saved in $OUTDIR"
