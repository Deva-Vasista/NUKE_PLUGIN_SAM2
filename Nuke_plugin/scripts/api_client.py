import requests
import os
import tempfile
from pathlib import Path
import OpenEXR
import Imath
import numpy as np
import nuke
import json
from NukeSAM2 import get_api_base_url

class NukeSamuraiClient:
    def __init__(self, api_url=None):
        self.api_url = api_url or get_api_base_url()
        self.session = requests.Session()
        
    def update_api_url(self):
        """Update the API URL from the node configuration"""
        self.api_url = get_api_base_url()
        
    def check_health(self):
        """Check if the API server is healthy"""
        try:
            self.update_api_url()  # Update URL before making request
            response = self.session.get(f"{self.api_url}/health")
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            nuke.message(f"Error connecting to NukeSamurai server: {str(e)}")
            return None
            
    def validate_numeric_data(self, data):
        """Ensure all numeric data is properly typed as float"""
        if isinstance(data, dict):
            return {k: self.validate_numeric_data(v) for k, v in data.items()}
        elif isinstance(data, list) or isinstance(data, tuple):
            return [self.validate_numeric_data(item) for item in data]
        elif isinstance(data, (int, float)) and not isinstance(data, bool):
            return float(data)
        else:
            return data
            
    def process_exr(self, exr_path, bbox=None, points_positive=None, points_negative=None, bits="32-bit float"):
        """Process EXR file through the API with bounding box or point prompts"""
        try:
            # Validate inputs to proper types
            if bbox is not None:
                bbox = [float(x) for x in bbox]
                
            if points_positive is not None:
                points_positive = [[float(x), float(y)] for x, y in points_positive]
                
            if points_negative is not None:
                points_negative = [[float(x), float(y)] for x, y in points_negative]
            
            # Prepare request data
            request_data = {
                "image_path": exr_path,
                "bits": bits
            }
            
            if bbox is not None:
                request_data["bbox"] = bbox
                
            if points_positive:
                request_data["points_positive"] = points_positive
                
            if points_negative:
                request_data["points_negative"] = points_negative
            
            # Log the request data for debugging
            nuke.tprint(f"Sending request data: {json.dumps(request_data, indent=2)}")
                
            # Send request
            response = self.session.post(
                f"{self.api_url}/api/v1/process_exr?as_file=true",
                json=request_data
            )
            response.raise_for_status()
            
            # Save the result to a temporary file
            with tempfile.NamedTemporaryFile(suffix='.exr', delete=False) as tmp:
                tmp.write(response.content)
                return tmp.name
                
        except Exception as e:
            nuke.message(f"Error processing EXR: {str(e)}")
            return None
            
    def process_sequence(self, sequence_path, frame_range, prompts, bits="32-bit float", original_fps=24, target_fps=24):
        """Process an EXR sequence with prompts"""
        try:
            # Validate all values to ensure they're float
            validated_prompts = []
            for prompt in prompts:
                # Validate bbox if it exists
                if "bbox" in prompt and prompt["bbox"] is not None:
                    prompt["bbox"] = [float(x) for x in prompt["bbox"]]
                    
                # Validate points
                if "points_positive" in prompt and prompt["points_positive"]:
                    prompt["points_positive"] = [[float(x), float(y)] for x, y in prompt["points_positive"]]
                    
                if "points_negative" in prompt and prompt["points_negative"]:
                    prompt["points_negative"] = [[float(x), float(y)] for x, y in prompt["points_negative"]]
                    
                validated_prompts.append(prompt)
            
            # Create request data
            request_data = {
                "sequence_path": sequence_path,
                "frame_range": [int(frame_range[0]), int(frame_range[1])],
                "prompts": validated_prompts,
                "bits": bits,
                "original_fps": float(original_fps),
                "target_fps": float(target_fps),
                "chunk_size": 20  # Match backend chunk size
            }
            
            # Log the request for debugging
            nuke.tprint(f"Sending sequence request: {json.dumps(request_data, indent=2)}")
            
            # Send request with increased timeout for large sequences
            response = self.session.post(
                f"{self.api_url}/api/v1/process_sequence?as_file=true",
                json=request_data,
                timeout=300  # 5 minutes timeout
            )
            response.raise_for_status()
            
            # Check for task_id in response
            result = response.json()
            if "task_id" in result:
                nuke.tprint(f"Processing started with task ID: {result['task_id']}")
                return result["task_id"]
            
            # If no task_id, this is an immediate result
            with tempfile.NamedTemporaryFile(suffix='.zip', delete=False) as tmp:
                tmp.write(response.content)
                return tmp.name
                
        except requests.exceptions.Timeout:
            error_msg = "Request timed out. The sequence might be too large or the server is busy."
            nuke.tprint(error_msg)
            raise Exception(error_msg)
        except requests.exceptions.ConnectionError:
            error_msg = "Failed to connect to the server. Please check if the server is running."
            nuke.tprint(error_msg)
            raise Exception(error_msg)
        except Exception as e:
            error_msg = f"Error processing sequence: {str(e)}"
            nuke.tprint(error_msg)
            raise Exception(error_msg)
            
    def get_task_status(self, task_id):
        """Get the current status of a processing task"""
        try:
            response = self.session.get(f"{self.api_url}/api/v1/status/{task_id}")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            nuke.tprint(f"Error getting task status: {str(e)}")
            return None
            
    def download_task_output(self, task_id, output_path):
        """Download the output files for a completed task"""
        try:
            # First try to get all files as a ZIP
            response = self.session.get(
                f"{self.api_url}/api/v1/download_all/{task_id}",
                stream=True,
                timeout=300  # 5 minutes timeout
            )
            response.raise_for_status()
            
            # Create output directory if it doesn't exist
            output_dir = os.path.dirname(output_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)
            
            # Save the ZIP file
            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    
            return output_path
            
        except Exception as e:
            error_msg = f"Error downloading task output: {str(e)}"
            nuke.tprint(error_msg)
            raise Exception(error_msg)
            
    def create_mask_node(self, exr_path):
        """Create a Nuke node with the processed mask"""
        try:
            # Process the EXR
            result_path = self.process_exr(exr_path)
            if not result_path:
                return None
                
            # Create Read node for the mask
            mask_node = nuke.nodes.Read(file=result_path)
            mask_node['first'].setValue(nuke.frame())
            mask_node['last'].setValue(nuke.frame())
            mask_node['origfirst'].setValue(nuke.frame())
            mask_node['origlast'].setValue(nuke.frame())
            
            # Create Shuffle node to extract the mask channel
            shuffle_node = nuke.nodes.Shuffle(inputs=[mask_node])
            shuffle_node['in'].setValue('mask')
            shuffle_node['out'].setValue('rgba')
            
            return shuffle_node
            
        except Exception as e:
            nuke.message(f"Error creating mask node: {str(e)}")
            return None
