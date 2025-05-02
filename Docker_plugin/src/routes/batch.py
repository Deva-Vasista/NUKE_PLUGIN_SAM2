from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List
import uuid
from src.utils.progress_tracker import ProgressTracker
from src.models.sam_wrapper import SAMProcessor

router = APIRouter()
processor = SAMProcessor()
progress_tracker = ProgressTracker()

class SequenceRequest(BaseModel):
    path: str
    frame_range: List[int]
    bbox: str

class BatchRequest(BaseModel):
    sequences: List[SequenceRequest]

@router.post("/process_batch")
async def process_batch(request: BatchRequest):
    """Process multiple sequences in batch."""
    try:
        batch_id = str(uuid.uuid4())
        task_ids = []
        
        # Create a task for each sequence
        for i, seq in enumerate(request.sequences):
            task_id = f"{batch_id}_{i}"
            await progress_tracker.create_task(task_id)
            task_ids.append(task_id)
            
            # Start processing each sequence
            try:
                await progress_tracker.update_progress(
                    task_id, 
                    0, 
                    f"Processing sequence {i+1}/{len(request.sequences)}"
                )
                
                # Process the sequence
                result = await processor.generate_mask_async(
                    seq.path,
                    [float(x) for x in seq.bbox.split(",")],
                    frame_range=seq.frame_range
                )
                
                await progress_tracker.complete_task(task_id, success=True)
                
            except Exception as e:
                await progress_tracker.complete_task(
                    task_id, 
                    success=False, 
                    error=str(e)
                )
                raise
        
        return {
            "batch_id": batch_id,
            "task_ids": task_ids,
            "status": "processing"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/batch/{batch_id}/status")
async def get_batch_status(batch_id: str):
    """Get status of all tasks in a batch."""
    try:
        # Find all tasks for this batch
        tasks = {}
        for task_id in progress_tracker.tasks:
            if task_id.startswith(batch_id):
                status = progress_tracker.get_task_status(task_id)
                if status:
                    tasks[task_id] = status.to_dict()
        
        if not tasks:
            raise HTTPException(status_code=404, detail="Batch not found")
            
        # Calculate overall progress
        completed = sum(1 for t in tasks.values() if t["status"] in ["completed", "failed"])
        total = len(tasks)
        overall_progress = (completed / total) * 100 if total > 0 else 0
        
        return {
            "batch_id": batch_id,
            "tasks": tasks,
            "overall_progress": overall_progress,
            "status": "completed" if completed == total else "processing"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 