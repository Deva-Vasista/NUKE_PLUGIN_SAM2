import requests
import os
import tempfile
from pathlib import Path
import OpenEXR
import Imath
import numpy as np
import nuke

class NukeSamuraiClient:
    def __init__(self, api_url="http://localhost:8000"):
        self.api_url = api_url
        self.session = requests.Session()
        
    def check_health(self):
        """Check if the API server is healthy"""
        try:
            response = self.session.get(f"{self.api_url}/health")
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            nuke.message(f"Error connecting to NukeSamurai server: {str(e)}")
            return None
            
    def process_exr(self, exr_path, start_frame=None, end_frame=None):
        """Process EXR file through the API"""
        try:
            # Read the EXR file
            with open(exr_path, 'rb') as f:
                files = {'file': ('input.exr', f, 'application/octet-stream')}
                
                # Prepare parameters
                params = {}
                if start_frame is not None:
                    params['start_frame'] = start_frame
                if end_frame is not None:
                    params['end_frame'] = end_frame
                    
                # Send request
                response = self.session.post(
                    f"{self.api_url}/process_exr",
                    files=files,
                    params=params
                )
                response.raise_for_status()
                
                # Save the result to a temporary file
                with tempfile.NamedTemporaryFile(suffix='.exr', delete=False) as tmp:
                    tmp.write(response.json()['result'])
                    return tmp.name
                    
        except Exception as e:
            nuke.message(f"Error processing EXR: {str(e)}")
            return None
            
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
