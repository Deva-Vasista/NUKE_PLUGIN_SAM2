from fastapi import APIRouter, BackgroundTasks
from src.models.sam_wrapper import SAMProcessor

router = APIRouter()
processor = SAMProcessor()

@router.post("/load_model/{model_type}")
async def load_model(model_type: str, background_tasks: BackgroundTasks):
    valid_models = {
        "large": "sam2.1_hiera_large.pt",
        "base-plus": "sam2.1_hiera_base_plus.pt",
        "small": "sam2.1_hiera_small.pt",
        "tiny": "sam2.1_hiera_tiny.pt"
    }
    
    if model_type not in valid_models:
        return {"error": "Invalid model type"}
    
    def _load_model_async(model_path):
        processor.load_model(model_path)
        
    background_tasks.add_task(
        _load_model_async, 
        f"checkpoints/{valid_models[model_type]}"
    )
    
    return {"status": "Model loading started"}
