import nuke
import os
import json
import threading
import websockets
import asyncio
import requests
import cv2
from urllib.parse import urljoin
import zipfile
import time

# API Configuration
API_BASE_URL = "http://localhost:8000"
API_VERSION = "v1"

class Prompt:
    def __init__(self):
        self.frame_index = 0
        self.object_id = 0
        self.bbox = None
        self.points_positive = []
        self.points_negative = []

    def validate_data_types(self):
        """Ensure all data is in correct types before sending to API"""
        if self.bbox is not None:
            # Ensure bbox is a list of float values
            self.bbox = [float(x) for x in self.bbox]
            
        # Convert all points to float values
        validated_points_positive = []
        for point in self.points_positive:
            validated_points_positive.append([float(point[0]), float(point[1])])
        self.points_positive = validated_points_positive
        
        validated_points_negative = []
        for point in self.points_negative:
            validated_points_negative.append([float(point[0]), float(point[1])])
        self.points_negative = validated_points_negative
        
        return self

class BoundingBox:
    input_path = None
    prompts = []  # List of Prompt objects
    current_prompt = None
    drawing_box = False
    start_point = None
    current_box = None
    current_frame = None
    min_frame = None
    max_frame = None
    current_object_id = 0

    @classmethod
    def getBbox(cls):
        file_path = InputInfos.path
        input_file_name = str(os.path.splitext(os.path.basename(file_path))[0])
        
        # Initialize frame range
        cls.min_frame = int(nuke.thisNode().knob('FrameRangeMin').value())
        cls.max_frame = int(nuke.thisNode().knob('FrameRangeMax').value())
        cls.current_frame = cls.min_frame
        
        # Initialize with object_id 0 by default
        cls.current_object_id = 0

        def get_frame_path(frame_num):
            if "%04d" in input_file_name:
                return file_path.replace('%04d', str(f"{frame_num:04}"))
            elif "%03d" in input_file_name:
                return file_path.replace('%03d', str(f"{frame_num:03}"))
            return file_path

        cls.input_path = get_frame_path(cls.current_frame)
        
        window_name = "Selection - Left/Right arrows to change frame, Left click and drag for box, 'p' for positive point, 'n' for negative point, '1-9' to set object ID, 'z' to undo, 'r' to reset, 'q' to finish"
        cv2.namedWindow(window_name, 0)
        cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)
        cv2.resizeWindow(window_name, 1280, 720)

        def mouse_callback(event, x, y, flags, param):
            if event == cv2.EVENT_MOUSEMOVE:
                param['current_pos'] = (x, y)
                if cls.drawing_box:
                    cls.current_box = (cls.start_point[0], cls.start_point[1], x - cls.start_point[0], y - cls.start_point[1])
            elif event == cv2.EVENT_LBUTTONDOWN:
                cls.drawing_box = True
                cls.start_point = (x, y)
            elif event == cv2.EVENT_LBUTTONUP:
                if cls.drawing_box:
                    cls.drawing_box = False
                    if cls.current_box[2] > 0 and cls.current_box[3] > 0:
                        # Create new prompt
                        prompt = Prompt()
                        prompt.frame_index = cls.current_frame
                        prompt.object_id = cls.current_object_id
                        prompt.bbox = [cls.current_box[0], cls.current_box[1], 
                                     cls.current_box[0] + cls.current_box[2], 
                                     cls.current_box[1] + cls.current_box[3]]
                        cls.prompts.append(prompt)
                        nuke.tprint(f"Added box on frame {cls.current_frame} for object {cls.current_object_id}: {prompt.bbox}")
                    cls.current_box = None

        # Dictionary to store current mouse position
        mouse_pos = {'current_pos': (0, 0)}
        cv2.setMouseCallback(window_name, mouse_callback, mouse_pos)

        while True:
            # Load current frame
            current_img_path = get_frame_path(cls.current_frame)
            img = cv2.imread(current_img_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
            img_with_boxes = img.copy()

            # Draw existing boxes and points for current frame
            for prompt in cls.prompts:
                if prompt.frame_index == cls.current_frame:
                    # Use different colors for different object IDs
                    color_r = (prompt.object_id * 40) % 255
                    color_g = (prompt.object_id * 80 + 100) % 255
                    color_b = (prompt.object_id * 120 + 50) % 255
                    
                    # Box color based on object ID
                    object_color = (color_b, color_g, color_r)
                    
                    if prompt.bbox:
                        x1, y1, x2, y2 = prompt.bbox
                        cv2.rectangle(img_with_boxes, (x1, y1), (x2, y2), object_color, 2)
                        cv2.putText(img_with_boxes, f"Obj {prompt.object_id}", (x1, y1-5), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, object_color, 2)
                    
                    for point in prompt.points_positive:
                        cv2.circle(img_with_boxes, point, 5, object_color, -1)
                        cv2.putText(img_with_boxes, f"+{prompt.object_id}", (point[0]-5, point[1]-5), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, object_color, 2)
                    
                    for point in prompt.points_negative:
                        cv2.circle(img_with_boxes, point, 5, (0, 0, 255), -1)
                        cv2.putText(img_with_boxes, f"-{prompt.object_id}", (point[0]-5, point[1]-5), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            # Draw current box being drawn
            if cls.drawing_box and cls.current_box:
                x, y, w, h = cls.current_box
                
                # Color for current object ID
                color_r = (cls.current_object_id * 40) % 255
                color_g = (cls.current_object_id * 80 + 100) % 255
                color_b = (cls.current_object_id * 120 + 50) % 255
                object_color = (color_b, color_g, color_r)
                
                cv2.rectangle(img_with_boxes, (x, y), (x + w, y + h), object_color, 2)

            # Create a semi-transparent overlay for better text visibility
            overlay = img_with_boxes.copy()
            cv2.rectangle(overlay, (0, 0), (600, 400), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.7, img_with_boxes, 0.3, 0, img_with_boxes)

            # Show instructions
            cv2.putText(img_with_boxes, "Left/Right arrows to change frame", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(img_with_boxes, "Left click and drag to draw box", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(img_with_boxes, "Press 'p' to add positive point", (10, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(img_with_boxes, "Press 'n' to add negative point", (10, 120), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(img_with_boxes, "Press '1-9' to set object ID", (10, 150), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(img_with_boxes, "Press 'z' to undo last action", (10, 180), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(img_with_boxes, "Press 'r' to reset current frame", (10, 210), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(img_with_boxes, "Press 'q' to finish", (10, 240), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # Show current frame, object ID, and selection counts
            current_prompts = [p for p in cls.prompts if p.frame_index == cls.current_frame]
            boxes_count = sum(1 for p in current_prompts if p.bbox)
            pos_points = sum(len(p.points_positive) for p in current_prompts)
            neg_points = sum(len(p.points_negative) for p in current_prompts)

            cv2.putText(img_with_boxes, f"Current Frame: {cls.current_frame}", (10, 280), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(img_with_boxes, f"Current Object ID: {cls.current_object_id}", (10, 310), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(img_with_boxes, f"Boxes: {boxes_count}", (10, 340), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(img_with_boxes, f"Positive points: {pos_points}", (10, 370), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(img_with_boxes, f"Negative points: {neg_points}", (10, 400), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # Show object ID summary (for current frame)
            unique_object_ids = sorted(set(p.object_id for p in current_prompts))
            if unique_object_ids:
                obj_summary = "Objects in frame: " + ", ".join(f"ID {obj_id}" for obj_id in unique_object_ids)
                cv2.putText(img_with_boxes, obj_summary, (10, 430), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 100), 2)

            cv2.imshow(window_name, img_with_boxes)

            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):  # Press 'q' to finish
                if cls.prompts:
                    nuke.tprint(f"Finishing selection with {len(cls.prompts)} prompts")
                    cls.send_to_api()  # Send prompts to API
                    break
                else:
                    nuke.tprint("Please make at least one selection before finishing")
            
            elif key == ord('r'):  # Press 'r' to reset current frame
                cls.prompts = [p for p in cls.prompts if p.frame_index != cls.current_frame]
                cls.drawing_box = False
                cls.start_point = None
                cls.current_box = None
                nuke.tprint(f"Reset selections for frame {cls.current_frame}")
            
            elif key == ord('z'):  # Press 'z' to undo
                current_prompts = [p for p in cls.prompts if p.frame_index == cls.current_frame]
                if current_prompts:
                    cls.prompts.remove(current_prompts[-1])
                    nuke.tprint(f"Undid last selection on frame {cls.current_frame}")
            
            # Number keys for object ID selection (1-9)
            elif ord('1') <= key <= ord('9'):
                # Set the object_id (0-8) - subtract 1 to start from 0
                new_object_id = key - ord('1')
                nuke.tprint(f"Set current object ID to {new_object_id}")
                cls.current_object_id = new_object_id
            
            elif key == ord('p'):  # Press 'p' to add positive point
                x, y = mouse_pos['current_pos']
                prompt = Prompt()
                prompt.frame_index = cls.current_frame
                prompt.object_id = cls.current_object_id
                prompt.points_positive = [[x, y]]
                cls.prompts.append(prompt)
                nuke.tprint(f"Added positive point on frame {cls.current_frame} for object {cls.current_object_id}: {x = }, {y = }")
            
            elif key == ord('n'):  # Press 'n' to add negative point
                x, y = mouse_pos['current_pos']
                prompt = Prompt()
                prompt.frame_index = cls.current_frame
                prompt.object_id = cls.current_object_id
                prompt.points_negative = [[x, y]]
                cls.prompts.append(prompt)
                nuke.tprint(f"Added negative point on frame {cls.current_frame} for object {cls.current_object_id}: {x = }, {y = }")
            
            elif key == 83 or key == 100:  # Right arrow key (→)
                if cls.current_frame < cls.max_frame:
                    cls.current_frame += 1
                    nuke.tprint(f"Moving to frame {cls.current_frame}")
            
            elif key == 81 or key == 97:  # Left arrow key (←)
                if cls.current_frame > cls.min_frame:
                    cls.current_frame -= 1
                    nuke.tprint(f"Moving to frame {cls.current_frame}")

        cv2.destroyWindow(window_name)

        if not cls.prompts:
            raise ValueError("No selections were made on any frame")

        # Update UI with more robust error handling
        try:
            node = nuke.thisNode()
            if node is None:
                node = nuke.selectedNode()
                nuke.tprint("Using selectedNode() as thisNode() returned None")
            
            if node is None:
                nuke.tprint("ERROR: No node context found - both thisNode() and selectedNode() returned None")
                return cls.prompts
                
            prompts_knob = node.knob('PromptsList')
            if prompts_knob is None:
                nuke.tprint("ERROR: PromptsList knob not found on node")
                return cls.prompts
                
            prompts_json = json.dumps([p.__dict__ for p in cls.prompts])
            prompts_knob.setValue(prompts_json)
            nuke.tprint(f"Successfully updated PromptsList with {len(cls.prompts)} prompts")
            
        except Exception as e:
            nuke.tprint(f"ERROR updating PromptsList: {str(e)}")
            nuke.tprint("Continuing with prompts data but UI update failed")
            
        return cls.prompts

    @classmethod
    def clearPrompts(cls):
        cls.prompts = []
        cls.current_prompt = None
        nuke.thisNode().knob('PromptsList').setValue("[]")

    @classmethod
    def send_to_api(cls):
        # Format prompts for API
        prompts_data = []
        for prompt in cls.prompts:
            # Validate data types before sending
            prompt.validate_data_types()
            prompt_dict = {
                "frame_index": prompt.frame_index,
                "object_id": prompt.object_id,
                "bbox": prompt.bbox,
                "points_positive": prompt.points_positive,
                "points_negative": prompt.points_negative
            }
            prompts_data.append(prompt_dict)

        # Send data to API
        try:
            response = requests.post(
                f"{API_BASE_URL}/{API_VERSION}/segment",
                json={"prompts": prompts_data}
            )
            response.raise_for_status()
            result = response.json()
            nuke.tprint(f"API response: {result}")
            # TODO: Handle the result (e.g., create mask nodes in Nuke)
        except Exception as e:
            nuke.tprint(f"Error sending prompts to API: {str(e)}")

class InputInfos:
    read = None
    path = None
    original_fps = None
    bits = None
    
    @classmethod
    def getInputInfos(cls):
        f = nuke.thisNode().dependencies()

        for i in f:
            cls.read = i

        # Get file path
        try:
            cls.path = cls.read.knob('file').getValue()
        except:
            cls.path = ''

        # Get metadatas
        ## get input fps
        try:
            cls.original_fps = int(cls.read.metadata()['input/frame_rate'])
        except:
            cls.original_fps = nuke.Root()['fps'].value()

        ## Get input bit depth
        try:
            cls.bits = cls.read.metadata()['input/bitsperchannel']
        except:
            nuke.tprint("No Bit Depth information in the input metadatas. Interpreting it from the input extension.")

            if cls.path.endswith("jpg") or cls.path.endswith("jpeg") or cls.path.endswith("png") or cls.path.endswith("tiff"):
                cls.bits = "8-bit fixed"
            elif cls.path.endswith("exr"):
                cls.bits = "32-bit float"
            else:
                TypeError("Cannot interpret bit depth from unsupported input format.")

def UpdatePath():
    InputInfos.getInputInfos()
    FilePath = InputInfos.path
    nuke.thisNode().knob('FilePath').setValue(FilePath)

async def monitor_progress(task_id):
    """Monitor progress via WebSocket"""
    uri = f"ws://localhost:8000/api/{API_VERSION}/ws/{task_id}"
    try:
        nuke.tprint(f"Connecting to WebSocket at {uri}...")
        # Set shorter timeouts for connection and operations
        async with websockets.connect(uri, ping_timeout=20, close_timeout=10) as websocket:
            nuke.tprint("WebSocket connection established, monitoring progress...")
            
            # Keep track of last progress value and message to detect being stuck
            last_progress = 0
            last_message = ""
            stuck_count = 0
            max_stuck_count = 10  # Exit after being stuck for a while
            
            # Add timeout for the entire monitoring process
            start_time = time.time()
            max_websocket_time = 300  # 5 minutes maximum
            
            # Create a task structure to check for files when progress seems stuck
            output_dir = f"Output/{task_id}"
            
            while True:
                # Check for timeout
                if time.time() - start_time > max_websocket_time:
                    nuke.tprint(f"WebSocket monitoring timed out after {max_websocket_time} seconds")
                    # Set progress to 100% when timing out - it might have completed but we didn't get the notification
                    def set_progress_complete():
                        node = nuke.thisNode()
                        node.knob('Progress').setValue(100)
                        if node.knob('StatusLabel'):
                            node.knob('StatusLabel').setValue("Completed (timeout reached)")
                    nuke.executeInMainThread(set_progress_complete)
                    break
                    
                try:
                    # Use timeout to prevent waiting forever
                    data = await asyncio.wait_for(websocket.recv(), timeout=30)
                    progress_data = json.loads(data)
                    
                    # Update the progress bar
                    progress = progress_data.get('progress', 0)
                    message = progress_data.get('message', '')
                    status = progress_data.get('status', 'running')
                    
                    nuke.tprint(f"Progress update: {progress}% - {message} (status: {status})")
                    
                    # Update progress knob in main thread
                    def update_progress_ui(progress_value, message_text):
                        node = nuke.thisNode()
                        node.knob('Progress').setValue(progress_value)
                        # Update status label if it exists
                        if node.knob('StatusLabel'):
                            node.knob('StatusLabel').setValue(message_text)
                    
                    nuke.executeInMainThread(lambda: update_progress_ui(progress, message))
                    
                    # Check if progress is stuck at the same percentage and message
                    if progress == last_progress and message == last_message:
                        stuck_count += 1
                        if stuck_count >= max_stuck_count:
                            nuke.tprint(f"Progress stuck at {progress}% for {max_stuck_count} updates, checking for output files")
                            
                            # Try to check if output files exist already, which would indicate completion
                            try:
                                status_url = f"{API_BASE_URL}/api/{API_VERSION}/output/{task_id}"
                                status_response = requests.get(status_url, timeout=5)
                                if status_response.status_code == 200:
                                    output_data = status_response.json()
                                    files = output_data.get("files", [])
                                    mask_files = [f for f in files if f["filename"].startswith("mask_")]
                                    
                                    # If we have output files but progress is stuck, consider it complete
                                    if mask_files:
                                        nuke.tprint(f"Found {len(mask_files)} mask files while progress is stuck. Considering task complete.")
                                        def set_progress_complete():
                                            node = nuke.thisNode()
                                            node.knob('Progress').setValue(100)
                                            if node.knob('StatusLabel'):
                                                node.knob('StatusLabel').setValue(f"Completed with {len(mask_files)} frames")
                                        nuke.executeInMainThread(set_progress_complete)
                                        break
                            except Exception as e:
                                nuke.tprint(f"Error checking for output files: {str(e)}")
                            
                            # If we still don't have confirmation of completion, exit monitoring loop
                            nuke.tprint("Exiting monitoring loop due to stuck progress")
                            break
                    else:
                        stuck_count = 0
                        last_progress = progress
                        last_message = message
                    
                    # Only display popup for completion or failure, and break out of the monitoring loop
                    if status in ['completed', 'failed']:
                        # Always make sure progress is 100% when completed
                        if status == 'completed' and progress < 100:
                            def set_progress_complete():
                                node = nuke.thisNode()
                                node.knob('Progress').setValue(100)
                            nuke.executeInMainThread(set_progress_complete)
                        
                        if status == 'failed':
                            error = progress_data.get('error', 'Unknown error')
                            nuke.tprint(f"Task failed: {error}")
                            nuke.executeInMainThread(lambda e=error: nuke.message(f"Task failed: {e}"))
                        break
                        
                except asyncio.TimeoutError:
                    # Timeout waiting for messages, check if connection is still alive
                    nuke.tprint("Timeout waiting for progress updates, sending ping...")
                    try:
                        pong_event = await websocket.ping()
                        await asyncio.wait_for(pong_event, timeout=5)
                        nuke.tprint("Server responded to ping, continuing...")
                    except asyncio.TimeoutError:
                        nuke.tprint("Server did not respond to ping, checking for output files...")
                        
                        # Check if output files exist already, which would indicate completion
                        try:
                            status_url = f"{API_BASE_URL}/api/{API_VERSION}/output/{task_id}"
                            status_response = requests.get(status_url, timeout=5)
                            if status_response.status_code == 200:
                                output_data = status_response.json()
                                files = output_data.get("files", [])
                                mask_files = [f for f in files if f["filename"].startswith("mask_")]
                                
                                # If we have output files but lost connection, consider it complete
                                if mask_files:
                                    nuke.tprint(f"Found {len(mask_files)} mask files after connection timeout. Considering task complete.")
                                    def set_progress_complete():
                                        node = nuke.thisNode()
                                        node.knob('Progress').setValue(100)
                                        if node.knob('StatusLabel'):
                                            node.knob('StatusLabel').setValue(f"Completed with {len(mask_files)} frames")
                                    nuke.executeInMainThread(set_progress_complete)
                            else:
                                nuke.tprint("No output files found yet, breaking connection")
                        except Exception as e:
                            nuke.tprint(f"Error checking for output files: {str(e)}")
                        
                        break
                except Exception as e:
                    nuke.tprint(f"Error receiving progress update: {str(e)}")
                    # Try to continue anyway
                    await asyncio.sleep(2)
    except Exception as e:
        nuke.tprint(f"WebSocket connection error: {str(e)}")
        # If we can't connect to WebSocket, fall back to polling the server for status
        try:
            nuke.tprint("Falling back to HTTP polling for progress...")
            status_url = f"{API_BASE_URL}/api/{API_VERSION}/output/{task_id}"
            while True:
                try:
                    status_response = requests.get(status_url, timeout=5)
                    if status_response.status_code == 200:
                        # We have a result, task is likely complete
                        nuke.executeInMainThread(lambda: nuke.thisNode().knob('Progress').setValue(100))
                        nuke.tprint("Task completed according to HTTP polling")
                        break
                    elif status_response.status_code == 404:
                        # Task not complete yet
                        await asyncio.sleep(3)
                    else:
                        nuke.tprint(f"Unexpected status code from server: {status_response.status_code}")
                        break
                except Exception as polling_err:
                    nuke.tprint(f"Error polling status: {polling_err}")
                    break
        except Exception as fallback_err:
            nuke.tprint(f"Error in fallback polling: {fallback_err}")

def check_api_server():
    """Check if the API server is running and accessible"""
    try:
        response = requests.get(f"{API_BASE_URL}/api/{API_VERSION}/health")
        return response.status_code == 200
    except requests.exceptions.ConnectionError:
        return False

def GenerateMask():
    Output_path = nuke.thisNode().knob('OutputPath').getValue()

    # Checks
    if str(os.path.splitext(os.path.basename(Output_path))[0]) == '':
        raise TypeError("You must assign a file name")
        
    if nuke.thisNode()['FilePath'].value().lower().endswith("mp4"):
        raise TypeError('Unsupported input format. Input must be an Image Sequence')
        
    if nuke.thisNode().knob('FileType').value() == "exr":
        if ("%04d" not in Output_path) and ("%03d" not in Output_path):
            raise TypeError("Your file must contains '####' or '###'")

    if not BoundingBox.prompts:
        raise TypeError("No prompts added. Please add at least one bounding box or points.")

    # Check if API server is running
    if not check_api_server():
        nuke.message("Error: API server is not running. Please ensure the server is running at " + API_BASE_URL)
        return

    video_path = nuke.thisNode().knob('FilePath').value()
    video_output_path = nuke.thisNode().knob('OutputPath').getValue()
    save_to_file = nuke.thisNode().knob('FileType').value()
    
    # Get frame range from UI (Nuke uses 1-based frame numbers)
    ui_frame_min = int(nuke.thisNode().knob('FrameRangeMin').value())
    ui_frame_max = int(nuke.thisNode().knob('FrameRangeMax').value())
    
    # Create frame range for API (server uses 0-based indexing)
    # Adjust frame range for server's 0-based indexing
    frame_range = [ui_frame_min - 1, ui_frame_max]  # Convert to 0-based for server, keep max inclusive
    
    original_fps = int(InputInfos.original_fps)
    target_fps = int(nuke.thisNode().knob('FPS').value())
    bits = InputInfos.bits
    model_type = nuke.thisNode().knob('ModelType').value().lower()

    # First, ensure the correct model is loaded
    try:
        nuke.tprint("Attempting to load model...")
        response = requests.post(
            f"{API_BASE_URL}/api/{API_VERSION}/models/load",
            json={"model_type": model_type},
            timeout=30  # Add timeout
        )
        response.raise_for_status()
        nuke.tprint("Model loaded successfully")
    except requests.exceptions.ConnectionError:
        nuke.message(f"Failed to connect to API server at {API_BASE_URL}. Please ensure the server is running.")
        return
    except requests.exceptions.Timeout:
        nuke.message("Request timed out while loading model. Please try again.")
        return
    except Exception as e:
        nuke.message(f"Failed to load model: {str(e)}")
        return

    # Update status to indicate processing is starting
    nuke.thisNode().knob('StatusLabel').setValue("Starting processing...")
    nuke.thisNode().knob('Progress').setValue(0)

    # Process the sequence
    try:
        nuke.tprint("Preparing to process sequence...")
        
        # Validate all prompts to ensure correct data types
        validated_prompts = []
        for prompt in BoundingBox.prompts:
            # Adjust frame_index to be 0-based for the server
            # The prompt.frame_index is UI-based (1-indexed)
            prompt_dict = prompt.validate_data_types().__dict__
            # Convert frame index to 0-based for server
            prompt_dict["frame_index"] = prompt_dict["frame_index"] - ui_frame_min
            validated_prompts.append(prompt_dict)
        
        request_data = {
            "sequence_path": video_path,
            "frame_range": frame_range,
            "prompts": validated_prompts,
            "bits": bits,
            "original_fps": original_fps,
            "target_fps": target_fps
        }

        nuke.tprint("Sending sequence to API for processing...")
        nuke.tprint(f"Request data: {json.dumps(request_data, indent=2)}")
        
        # Increase timeout to 300 seconds (5 minutes) for processing large sequences
        # The server needs time to process frames, especially for longer sequences
        try:
            nuke.tprint("Making API request with extended timeout (5 minutes)...")
            response = requests.post(
                f"{API_BASE_URL}/api/{API_VERSION}/process_sequence?as_file=true",
                json=request_data,
                timeout=300  # Increased timeout to 5 minutes
            )
            response.raise_for_status()
            
            result = response.json()
            nuke.tprint(f"Received response: {json.dumps(result, indent=2)}")
            
            # Check for valid output
            if "result" in result:
                nuke.tprint("Processing completed successfully")
                
                # Handle different response formats
                if save_to_file:
                    # Check if we have direct access to the output files
                    if "output_dir" in result and "zip_path" in result:
                        output_dir = result["output_dir"]
                        zip_path = result["zip_path"]
                        file_count = result.get("file_count", 0)
                        
                        nuke.tprint(f"Server saved {file_count} files to {output_dir}")
                        nuke.tprint(f"ZIP file available at {zip_path}")
                        
                        # Try to download the result
                        try:
                            # Create output directory if needed
                            output_folder = os.path.dirname(video_output_path)
                            if output_folder and not os.path.exists(output_folder):
                                os.makedirs(output_folder, exist_ok=True)
                            
                            # Use the download endpoint regardless of local file existence
                            # This is more reliable with WSL/Windows interaction
                            nuke.tprint("Downloading result file from server...")
                            
                            # Try to download the result
                            download_url = f"{API_BASE_URL}/api/{API_VERSION}/download/{os.path.basename(zip_path)}"
                            
                            # Try regular download first
                            download_response = requests.get(download_url, stream=True)
                            if download_response.status_code != 200:
                                # If that fails, try the bulk download
                                nuke.tprint("Direct file download failed, trying bulk download...")
                                try:
                                    download_url = f"{API_BASE_URL}/api/{API_VERSION}/download_all/{task_id}"
                                    download_response = requests.get(download_url, stream=True)
                                except NameError:
                                    # task_id might not be defined yet in this code path
                                    nuke.tprint("task_id not defined yet, extracting from output_dir in result")
                                    # Try to extract task_id from output_dir path
                                    if "output_dir" in result:
                                        output_dir = result["output_dir"]
                                        import re
                                        task_id_match = re.search(r'Output/([^/]+)', output_dir)
                                        if task_id_match:
                                            task_id = task_id_match.group(1)
                                            download_url = f"{API_BASE_URL}/api/{API_VERSION}/download_all/{task_id}"
                                            download_response = requests.get(download_url, stream=True)
                                        else:
                                            nuke.tprint("Could not extract task_id from output_dir")
                                            download_response = download_response  # Keep the original response
                                
                            if download_response.status_code == 200:
                                # Continue with downloading and processing as before
                                # Save to temp file first to prevent incomplete downloads
                                temp_output_path = video_output_path + ".tmp"
                                with open(temp_output_path, 'wb') as f:
                                    for chunk in download_response.iter_content(chunk_size=8192):
                                        f.write(chunk)
                                
                                # Rename temp file to final output
                                if os.path.exists(video_output_path):
                                    os.remove(video_output_path)
                                os.rename(temp_output_path, video_output_path)
                                
                                # If it's a ZIP and not the final target format, extract it
                                if not video_output_path.lower().endswith('.zip') and video_output_path.endswith('.exr'):
                                    nuke.tprint(f"Extracting masks from {video_output_path} to output location")
                                    extract_dir = os.path.dirname(video_output_path)
                                    if not extract_dir:
                                        extract_dir = '.'
                                    
                                    # Create extraction dir if needed
                                    if extract_dir and not os.path.exists(extract_dir):
                                        os.makedirs(extract_dir, exist_ok=True)
                                    
                                    # Extract zip
                                    with zipfile.ZipFile(video_output_path, 'r') as zip_ref:
                                        zip_ref.extractall(extract_dir)
                                    
                                    # Clean up zip file after extraction
                                    os.remove(video_output_path)
                                    
                                # Create a read node for the mask files
                                if video_output_path.endswith('.exr'):
                                    pattern = video_output_path
                                else:
                                    pattern = os.path.join(extract_dir, "mask_%04d.exr")
                                    
                                # Create read node with proper frame range
                                # The mask files from server use 0-based indexing, so they're numbered from 0
                                # But Nuke expects frames to match the UI range (1-based), so adjust accordingly
                                read_node = nuke.nodes.Read(file=pattern)
                                
                                # Server's files are 0-indexed but Nuke expects frame numbers to match UI
                                # Set first/last to the UI-based frame range values
                                read_node['first'].setValue(ui_frame_min)
                                read_node['last'].setValue(ui_frame_max)
                                
                                # But also tell the node which frames actually exist in the file sequence (0-based)
                                read_node['origfirst'].setValue(0)
                                read_node['origlast'].setValue(frame_range[1] - frame_range[0])
                                
                                # Offset the frames to map from file index to UI frame number
                                read_node['frame_mode'].setValue('offset')
                                read_node['frame'].setValue(str(ui_frame_min))
                                
                                nuke.message(f"Mask generation completed! Files saved and loaded into Nuke.")
                            else:
                                nuke.message(f"Server processing completed but could not download result (HTTP {download_response.status_code}). Files are available on server at {output_dir}")
                        except Exception as e:
                            nuke.message(f"Error downloading output files: {str(e)}\nFiles are available on server at {output_dir}")
                    else:
                        # Old behavior - binary download
                        try:
                            # If the response is a ZIP file, save it directly
                            with open(video_output_path, 'wb') as f:
                                f.write(response.content)
                            nuke.message("Mask generation completed and saved successfully!")
                            
                            # Create read node for the output file
                            read_node = nuke.nodes.Read(file=video_output_path, first=frame_range[0], last=frame_range[1]-1, origfirst=frame_range[0], origlast=frame_range[1]-1)
                        except Exception as e:
                            nuke.message(f"Error saving output: {str(e)}")
                else:
                    # If we got back JSON data
                    try:
                        if "json_path" in result:
                            # Server saved JSON to file
                            json_path = result["json_path"]
                            nuke.tprint(f"Server saved JSON result to {json_path}")
                            
                            # Create a file structure for Nuke to read
                            output_folder = os.path.dirname(video_output_path)
                            if output_folder and not os.path.exists(output_folder):
                                os.makedirs(output_folder, exist_ok=True)
                                
                            # Save the JSON locally too
                            json_output_path = os.path.splitext(video_output_path)[0] + ".json"
                            
                            # Try to download the result
                            try:
                                download_url = f"{API_BASE_URL}/api/{API_VERSION}/download/{os.path.basename(json_path)}"
                                
                                # Try regular download first
                                download_response = requests.get(download_url, stream=True)
                                if download_response.status_code != 200:
                                    # If that fails, try the bulk download
                                    nuke.tprint("Direct file download failed, trying bulk download...")
                                    try:
                                        download_url = f"{API_BASE_URL}/api/{API_VERSION}/download_all/{task_id}"
                                        download_response = requests.get(download_url, stream=True)
                                    except NameError:
                                        # task_id might not be defined yet in this code path
                                        nuke.tprint("task_id not defined yet, extracting from output_dir in result")
                                        # Try to extract task_id from output_dir path
                                        if "output_dir" in result:
                                            output_dir = result["output_dir"]
                                            import re
                                            task_id_match = re.search(r'Output/([^/]+)', output_dir)
                                            if task_id_match:
                                                task_id = task_id_match.group(1)
                                                download_url = f"{API_BASE_URL}/api/{API_VERSION}/download_all/{task_id}"
                                                download_response = requests.get(download_url, stream=True)
                                            else:
                                                nuke.tprint("Could not extract task_id from output_dir")
                                                download_response = download_response  # Keep the original response
                                
                                if download_response.status_code == 200:
                                    # Continue with downloading and processing as before
                                    # Save to temp file first to prevent incomplete downloads
                                    temp_output_path = json_output_path + ".tmp"
                                    with open(temp_output_path, 'wb') as f:
                                        for chunk in download_response.iter_content(chunk_size=8192):
                                            f.write(chunk)
                                    
                                    # Rename temp file to final output
                                    if os.path.exists(json_output_path):
                                        os.remove(json_output_path)
                                    os.rename(temp_output_path, json_output_path)
                                    
                                    # Try to load the JSON to see if it's valid
                                    try:
                                        with open(json_output_path, 'r') as f:
                                            json_data = json.load(f)
                                        nuke.tprint(f"Downloaded and parsed valid JSON data from server")
                                    except json.JSONDecodeError:
                                        # If it's not valid JSON, it might be a binary file
                                        nuke.tprint("Downloaded file is not valid JSON, might be a binary file")
                                        # Try to download as all_files ZIP and extract
                                        try:
                                            all_files_url = f"{API_BASE_URL}/api/{API_VERSION}/download_all/{task_id}"
                                            all_files_response = requests.get(all_files_url, stream=True, timeout=30)
                                            if all_files_response.status_code == 200:
                                                extract_dir = os.path.dirname(json_output_path)
                                                zip_path = extract_dir + "/all_files.zip"
                                                with open(zip_path, 'wb') as f:
                                                    for chunk in all_files_response.iter_content(chunk_size=8192):
                                                        f.write(chunk)
                                                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                                                    zip_ref.extractall(extract_dir)
                                                # Look for JSON files in the extracted content
                                                for file in os.listdir(extract_dir):
                                                    if file.endswith('.json'):
                                                        json_output_path = os.path.join(extract_dir, file)
                                                        break
                                        except Exception as e:
                                            nuke.tprint(f"Error trying to get JSON data from all-files ZIP: {str(e)}")
                                    
                                    nuke.message(f"Mask generation completed! Data saved to {json_output_path}")
                            except Exception as e:
                                # Fall back to using data from the response
                                masks_data = result.get("result", {})
                                if masks_data:
                                    with open(json_output_path, 'w') as f:
                                        json.dump(masks_data, f)
                                    nuke.message(f"Mask generation completed! Data saved to {json_output_path}")
                                else:
                                    nuke.message(f"Error downloading JSON data: {str(e)}\nData is available on server at {json_path}")
                    except Exception as e:
                        nuke.message(f"Error saving JSON data: {str(e)}")
            elif "task_id" in result:
                nuke.tprint("Processing started, monitoring progress...")
                # Start progress monitoring in a separate thread
                task_id = result["task_id"]
                
                # Create a thread for monitoring progress
                def run_progress_monitor():
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    loop.run_until_complete(monitor_progress(task_id))
                    loop.close()
                
                # Start the progress monitoring in a background thread
                progress_thread = threading.Thread(target=run_progress_monitor)
                progress_thread.daemon = True
                progress_thread.start()
                
                # Wait for processing to complete
                output_endpoint = f"{API_BASE_URL}/api/{API_VERSION}/output/{task_id}"
                max_attempts = 60  # 5 minutes with 5-second interval
                attempt = 0
                success = False
                
                # Set a timeout timer
                start_time = time.time()
                timeout_seconds = 300  # 5 minutes timeout
                
                while attempt < max_attempts:
                    # Check if we've exceeded the timeout
                    if time.time() - start_time > timeout_seconds:
                        nuke.tprint(f"Timeout exceeded ({timeout_seconds} seconds). Giving up on waiting for task.")
                        nuke.message("Processing timed out. The task might still complete on the server, try downloading the results manually.")
                        break
                        
                    attempt += 1
                    nuke.tprint(f"Checking task status (attempt {attempt}/{max_attempts})...")
                    try:
                        output_response = requests.get(output_endpoint, timeout=10)
                        if output_response.status_code == 200:
                            output_data = output_response.json()
                            nuke.tprint(f"Got task output data: {json.dumps(output_data, indent=2)}")
                            
                            # Check if files are complete - if all expected frames exist, we can proceed
                            # The server frames are zero-indexed, but our frame range is one-indexed
                            server_frames = output_data.get("frames", [])
                            expected_frames = list(range(frame_range[0]-1, frame_range[1]-1))  # Adjust for zero-indexing
                            
                            # If all frames we expect are in the server frame list, we can consider it complete
                            all_frames_complete = all(frame in server_frames for frame in expected_frames)
                            if all_frames_complete:
                                nuke.tprint(f"All expected frames {expected_frames} are present in server frames {server_frames}")
                                
                            # Count actual mask files
                            mask_files = [f for f in output_data.get("files", []) if f["filename"].startswith("mask_")]
                            nuke.tprint(f"Found {len(mask_files)} mask files out of {len(expected_frames)} expected frames")
                            
                            # If file count matches expected frames or we have confirmed all frames are complete, proceed
                            if len(mask_files) >= len(expected_frames) or all_frames_complete:
                                nuke.tprint("All frames appear to be processed, proceeding with download regardless of progress status")
                            
                            # Find the ZIP file
                            zip_file = None
                            for file_info in output_data.get("files", []):
                                if file_info["filename"].endswith(".zip"):
                                    zip_file = file_info
                                    break
                                    
                            if zip_file:
                                # Download the ZIP file
                                nuke.tprint(f"Found ZIP file: {zip_file['filename']}")
                                download_url = f"{API_BASE_URL}/api/{API_VERSION}/download/{zip_file['filename']}"
                                
                                try:
                                    # Download and save the file
                                    zip_response = requests.get(download_url, stream=True, timeout=30)
                                    zip_response.raise_for_status()
                                    
                                    # Create output directory if needed
                                    output_dir = os.path.dirname(video_output_path)
                                    if output_dir and not os.path.exists(output_dir):
                                        os.makedirs(output_dir, exist_ok=True)
                                    
                                    # Save to a temporary file first
                                    temp_output_path = video_output_path + ".tmp"
                                    with open(temp_output_path, 'wb') as f:
                                        for chunk in zip_response.iter_content(chunk_size=8192):
                                            f.write(chunk)
                                    
                                    # Rename to final path
                                    if os.path.exists(video_output_path):
                                        os.remove(video_output_path)
                                    os.rename(temp_output_path, video_output_path)
                                    
                                    # Extract the ZIP if needed
                                    if not video_output_path.lower().endswith('.zip'):
                                        extract_dir = os.path.dirname(video_output_path)
                                        if not extract_dir:
                                            extract_dir = '.'
                                        
                                        # Create extraction dir if needed
                                        if not os.path.exists(extract_dir):
                                            os.makedirs(extract_dir, exist_ok=True)
                                        
                                        # Extract the ZIP file
                                        with zipfile.ZipFile(video_output_path, 'r') as zip_ref:
                                            zip_ref.extractall(extract_dir)
                                        
                                        # Clean up the ZIP file
                                        os.remove(video_output_path)
                                        
                                        # Load all EXR files - adjust for zero-indexing on server side
                                        mask_pattern = os.path.join(extract_dir, "mask_%04d.exr")
                                        read_node = nuke.nodes.Read(file=mask_pattern, first=0, last=frame_range[1]-frame_range[0]-1)
                                    else:
                                        # Keep the ZIP file
                                        read_node = nuke.nodes.Read(file=video_output_path, first=frame_range[0]-1, last=frame_range[1]-2)
                                    
                                    success = True
                                    nuke.message(f"Mask generation completed! Files saved and loaded into Nuke.")
                                    break
                                    
                                except Exception as download_err:
                                    nuke.tprint(f"Error downloading ZIP file: {str(download_err)}")
                                    # Try falling back to individual files
                                    try:
                                        # Try downloading the all-in-one ZIP
                                        nuke.tprint("Trying to download all files in one batch...")
                                        all_files_url = f"{API_BASE_URL}/api/{API_VERSION}/download_all/{task_id}"
                                        all_files_response = requests.get(all_files_url, stream=True, timeout=30)
                                        if all_files_response.status_code == 200:
                                            with open(video_output_path, 'wb') as f:
                                                for chunk in all_files_response.iter_content(chunk_size=8192):
                                                    f.write(chunk)
                                            
                                            # Extract the ZIP if needed
                                            extract_dir = os.path.dirname(video_output_path)
                                            if not extract_dir:
                                                extract_dir = '.'
                                            
                                            # Extract the ZIP
                                            with zipfile.ZipFile(video_output_path, 'r') as zip_ref:
                                                zip_ref.extractall(extract_dir)
                                            
                                            # Find EXR files - adjust for zero-indexing
                                            mask_pattern = os.path.join(extract_dir, "mask_%04d.exr")
                                            read_node = nuke.nodes.Read(file=mask_pattern, first=0, last=frame_range[1]-frame_range[0]-1)
                                            
                                            success = True
                                            nuke.message(f"Mask generation completed! Files saved and loaded into Nuke.")
                                            break
                                    except Exception as all_files_err:
                                        nuke.tprint(f"Error downloading all files: {str(all_files_err)}")
                            else:
                                # No ZIP file found yet, but files might be there individually
                                nuke.tprint("No ZIP file found, checking for individual mask files...")
                                
                                # Check if all mask files are present
                                mask_files_present = True
                                for frame_idx in range(0, frame_range[1]-frame_range[0]):  # Adjust for zero-indexing
                                    mask_filename = f"mask_{frame_idx:04d}.exr"
                                    found = False
                                    for file_info in output_data.get("files", []):
                                        if file_info["filename"] == mask_filename:
                                            found = True
                                            break
                                    if not found:
                                        mask_files_present = False
                                        break
                                
                                if mask_files_present:
                                    # All mask files are present, create a directory and download them
                                    try:
                                        extract_dir = os.path.dirname(video_output_path)
                                        if not extract_dir:
                                            extract_dir = '.'
                                        
                                        if not os.path.exists(extract_dir):
                                            os.makedirs(extract_dir, exist_ok=True)
                                        
                                        # Download each mask file - adjusted for zero-indexing
                                        for frame_idx in range(0, frame_range[1]-frame_range[0]):
                                            mask_filename = f"mask_{frame_idx:04d}.exr"
                                            download_url = f"{API_BASE_URL}/api/{API_VERSION}/download/{mask_filename}"
                                            mask_response = requests.get(download_url, timeout=10)
                                            if mask_response.status_code == 200:
                                                output_file = os.path.join(extract_dir, mask_filename)
                                                with open(output_file, 'wb') as f:
                                                    f.write(mask_response.content)
                                        
                                        # Create read node
                                        mask_pattern = os.path.join(extract_dir, "mask_%04d.exr")
                                        read_node = nuke.nodes.Read(file=mask_pattern, first=0, last=frame_range[1]-frame_range[0]-1)
                                        
                                        success = True
                                        nuke.message(f"Mask generation completed! Files saved and loaded into Nuke.")
                                        break
                                    except Exception as mask_download_err:
                                        nuke.tprint(f"Error downloading individual mask files: {str(mask_download_err)}")
                                        
                            # If we reach here and haven't broken out of the loop, check if we need to handle a stuck process
                            # Check if progress is stuck
                            progress_value = nuke.thisNode().knob('Progress').value()
                            if attempt > 5 and progress_value >= 80:
                                # If we've been at 80% or higher for a while and have all expected mask files,
                                # we can consider processing complete even if the server doesn't send a "completed" status
                                files = output_data.get("files", [])
                                mask_files = [f for f in files if f["filename"].startswith("mask_")]
                                expected_frame_count = frame_range[1] - frame_range[0]
                                
                                if len(mask_files) >= expected_frame_count:
                                    nuke.tprint(f"Progress stuck at {progress_value}%, but found {len(mask_files)} mask files, which meets or exceeds expected {expected_frame_count} frames")
                                    nuke.tprint("Considering processing complete and breaking out of waiting loop")
                                    
                                    # Use the all_files endpoint to get everything at once
                                    try:
                                        download_url = f"{API_BASE_URL}/api/{API_VERSION}/download_all/{task_id}"
                                        nuke.tprint(f"Downloading all files using: {download_url}")
                                        all_files_response = requests.get(download_url, stream=True, timeout=60)
                                        if all_files_response.status_code == 200:
                                            extract_dir = os.path.dirname(video_output_path)
                                            if not extract_dir:
                                                extract_dir = '.'
                                            
                                            if not os.path.exists(extract_dir):
                                                os.makedirs(extract_dir, exist_ok=True)
                                                
                                            all_files_zip = os.path.join(extract_dir, "all_files.zip")
                                            
                                            # Save the ZIP
                                            with open(all_files_zip, 'wb') as f:
                                                for chunk in all_files_response.iter_content(chunk_size=8192):
                                                    f.write(chunk)
                                                    
                                            # Extract it
                                            with zipfile.ZipFile(all_files_zip, 'r') as zip_ref:
                                                zip_ref.extractall(extract_dir)
                                                
                                            # Create read node with proper indexing
                                            mask_pattern = os.path.join(extract_dir, "mask_%04d.exr")
                                            read_node = nuke.nodes.Read(file=mask_pattern, first=0, last=frame_range[1]-frame_range[0]-1)
                                            
                                            success = True
                                            nuke.message(f"Mask generation completed! Files saved and loaded into Nuke.")
                                            break
                                    except Exception as e:
                                        nuke.tprint(f"Error when trying to force download files: {str(e)}")

                        elif output_response.status_code == 404:
                            # Task still processing
                            nuke.tprint("Task still processing, waiting...")
                    except Exception as e:
                        nuke.tprint(f"Error checking task status: {str(e)}")
                    
                    # Check if progress is stuck
                    progress_value = nuke.thisNode().knob('Progress').value()
                    if attempt > 10 and progress_value < 90:
                        # Try reading progress directly from server
                        try:
                            progress_url = f"{API_BASE_URL}/api/{API_VERSION}/status/{task_id}"
                            progress_response = requests.get(progress_url, timeout=5)
                            if progress_response.status_code == 200:
                                progress_data = progress_response.json()
                                status = progress_data.get('status', '')
                                if status == 'completed':
                                    # Force another attempt to get output
                                    nuke.tprint("Server indicates task is complete, trying to get output...")
                                    continue
                        except Exception:
                            pass
                    
                    # Wait before next attempt
                    time.sleep(5)
                
                if not success:
                    nuke.message("Timed out waiting for mask generation to complete. Please check the server logs.")
            else:
                nuke.message("Failed to get result data from server response.")
        except requests.exceptions.Timeout:
            nuke.message("Request timed out after 5 minutes. The sequence may be too large or complex to process in that time. Try reducing the frame range or using a different model type.")
        except requests.exceptions.ConnectionError as e:
            nuke.message(f"Connection error to API server: {str(e)}\n\nPlease check if the server is running.")
        except requests.exceptions.HTTPError as e:
            nuke.message(f"HTTP error when contacting API server: {str(e)}\n\nPlease check the server logs for details.")
        except Exception as e:
            nuke.message(f"Unexpected error during mask generation: {str(e)}")
            
    except requests.exceptions.ConnectionError:
        nuke.message(f"Lost connection to API server at {API_BASE_URL}. Please ensure the server is running.")
    except requests.exceptions.Timeout:
        nuke.message("Request timed out. Please try again with a smaller frame range or a faster model (like 'tiny').")
    except Exception as e:
        nuke.message(f"Error during mask generation: {str(e)}")

def CreateSAM2Node():
    # Creating node
    nuke.createNode('NoOp')
    s = nuke.selectedNode()

    # Adding knobs
    s.knob('name').setValue('SAM2')
    
    # Single tab with all controls
    s.addKnob(nuke.Tab_Knob('settings_tab', 'SAM2'))
    
    # Input section
    s.addKnob(nuke.Text_Knob('input_section', 'Input'))
    s.addKnob(nuke.File_Knob('FilePath', 'File Path'))
    s.addKnob(nuke.PyScript_Knob('UpdatePath', 'Update Path', 'UpdatePath()'))
    
    # Frame selection
    s.addKnob(nuke.Int_Knob("FrameRangeMin", 'Frame Range'))
    s.addKnob(nuke.Int_Knob("FrameRangeMax", ' '))
    s.addKnob(nuke.Int_Knob("FPS", 'Output Frame Rate'))
    s.addKnob(nuke.Enumeration_Knob('ModelType', 'Model type', ['Base+', 'Large', 'Small', 'Tiny']))
    
    # Selection controls
    s.addKnob(nuke.Text_Knob('selection_section', 'Selection'))
    s.addKnob(nuke.PyScript_Knob('CreateBoundingBox', 'Create Selection', 'BoundingBox.getBbox()'))
    s.addKnob(nuke.PyScript_Knob('ClearPrompts', 'Clear All Selections', 'BoundingBox.clearPrompts()'))
    s.addKnob(nuke.Text_Knob('PromptsList', 'Prompts List'))
    s.knob('PromptsList').setEnabled(False)
    
    # Output controls
    s.addKnob(nuke.Text_Knob('output_section', 'Output'))
    s.addKnob(nuke.Enumeration_Knob('FileType', 'File type', ['exr', 'mp4']))
    s.addKnob(nuke.File_Knob('OutputPath', 'Output Path'))
    s.addKnob(nuke.PyScript_Knob('GenerateMask', 'Generate Mask', 'GenerateMask()'))
    
    # Progress indicators
    s.addKnob(nuke.Text_Knob('progress_section', 'Progress'))
    s.addKnob(nuke.Double_Knob('Progress', 'Processing Progress'))
    s.knob('Progress').setRange(0, 100)
    s.knob('Progress').setValue(0)
    s.knob('Progress').setEnabled(False)
    s.addKnob(nuke.Text_Knob('StatusLabel', 'Status', 'Idle'))
    
    # Setting ranges, default values, tooltips & format
    s['FPS'].setValue(int(nuke.root().knob('fps').getValue()))
    s['FrameRangeMin'].setValue(int(nuke.Root()['first_frame'].value()))
    s['FrameRangeMax'].setValue(int(nuke.Root()['last_frame'].value()))
    s['Progress'].setValue(0)

    s['FPS'].setFlag(nuke.STARTLINE)
    s['FrameRangeMax'].clearFlag(nuke.STARTLINE)
    s['UpdatePath'].setFlag(nuke.STARTLINE)
    s['GenerateMask'].setFlag(nuke.STARTLINE)
    
    # Section dividers should be startline
    s['input_section'].setFlag(nuke.STARTLINE)
    s['selection_section'].setFlag(nuke.STARTLINE)
    s['output_section'].setFlag(nuke.STARTLINE)
    s['progress_section'].setFlag(nuke.STARTLINE)
    
    s['CreateBoundingBox'].setTooltip("Create selection. Use mouse to draw boxes, 'p' for positive point, 'n' for negative point, 1-9 keys to set object ID, arrows to change frame.")
    s['ClearPrompts'].setTooltip("Clear all selections")
    s['FPS'].setTooltip("Target FPS for the output video")
    s['ModelType'].setTooltip("Choose your model type")
    s['OutputPath'].setTooltip("path/to/your/file_####.exr, to create an image sequence add #### or ###")
    s['GenerateMask'].setTooltip("Generate Mask")
    s['Progress'].setTooltip("Processing progress") 