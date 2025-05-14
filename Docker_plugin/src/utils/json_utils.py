import numpy as np
import torch
from typing import Any, Dict, List, Union
import logging

logger = logging.getLogger(__name__)

def ensure_serializable(obj: Any) -> Any:
    """
    Recursively convert an object to a JSON-serializable type.
    Handles:
    - torch.Tensor
    - numpy.ndarray
    - custom objects with to_dict() method
    - lists, dicts, and basic types
    
    Args:
        obj: The object to convert
        
    Returns:
        A JSON-serializable version of the object
    """
    if obj is None:
        return None
    
    # Handle torch tensors
    if isinstance(obj, torch.Tensor):
        return ensure_serializable(obj.cpu().detach().numpy())
    
    # Handle numpy arrays and types
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    if isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    
    # Handle custom objects with to_dict()
    if hasattr(obj, 'to_dict') and callable(getattr(obj, 'to_dict')):
        return ensure_serializable(obj.to_dict())
    
    # Handle dictionaries
    if isinstance(obj, dict):
        return {key: ensure_serializable(value) for key, value in obj.items()}
    
    # Handle lists, tuples and sets
    if isinstance(obj, (list, tuple, set)):
        return [ensure_serializable(item) for item in obj]
    
    # Handle basic types
    if isinstance(obj, (str, int, float, bool)):
        return obj
    
    # If we can't handle it, try to convert to string or dict
    try:
        # For objects that might be dictionaries but not checking as isinstance(dict)
        return ensure_serializable(dict(obj))
    except (TypeError, ValueError):
        try:
            # Last resort: convert to string
            return str(obj)
        except Exception as e:
            logger.warning(f"Could not serialize object of type {type(obj)}: {e}")
            return f"<non-serializable: {type(obj).__name__}>"

def convert_response(response: Any) -> Any:
    """Utility function to convert API response to a serializable format"""
    return ensure_serializable(response) 