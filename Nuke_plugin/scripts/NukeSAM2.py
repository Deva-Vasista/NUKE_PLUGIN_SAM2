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
API_BASE_URL = "http://192.168.12.166:8000"
API_VERSION = "v1"

class ThreadedTask:
    """Helper class to run API requests in background threads"""
    def __init__(self, on_complete=None, on_progress=None, on_error=None):
        self.on_complete = on_complete or (lambda result: None)
        self.on_progress = on_progress or (lambda progress, message: None)
        self.on_error = on_error or (lambda error: None)
        self.thread = None
        self.running = False
        self.task_id = None
        self._error_reported = False
        
    def run_in_background(self, func, *args, **kwargs):
        """Run the provided function in a background thread"""
        self.thread = threading.Thread(target=self._run_task, args=(func, args, kwargs))
        self.thread.daemon = True
        self.running = True
        self._error_reported = False
        self.thread.start()
        return self
        
    def _run_task(self, func, args, kwargs):
        """Internal method to execute the task and handle callbacks"""
        try:
            result = func(*args, **kwargs)
            if self.running:  # Check if we've been cancelled
                nuke.executeInMainThread(lambda: self.on_complete(result))
        except Exception as e:
            if self.running:  # Check if we've been cancelled
                self._error_reported = True
                error_msg = str(e)
                nuke.executeInMainThread(lambda: self.on_error(error_msg))
                # Also update status to show the error
                try:
                    nuke.executeInMainThread(lambda msg=error_msg: update_status_safely(f"Error: {msg}"))
                except:
                    pass
        finally:
            self.running = False
            
    def cancel(self):
        """Cancel the running task"""
        self.running = False
        if self.task_id:
            # Try to cancel the task on the server too
            try:
                requests.post(f"{API_BASE_URL}/api/{API_VERSION}/cancel/{self.task_id}")
            except:
                pass  # Ignore errors when cancelling
    
    def report_error(self, error_msg):
        """Report an error and set the error flag"""
        self._error_reported = True
        nuke.executeInMainThread(lambda err=error_msg: self.on_error(err))
        # Also update status to show the error
        try:
            nuke.executeInMainThread(lambda msg=error_msg: update_status_safely(f"Error: {msg}"))
        except:
            pass

    def start_progress_monitor(self, task_id):
        """Start monitoring progress for a task"""
        self.task_id = task_id
        monitor_thread = threading.Thread(target=self._monitor_progress_thread, args=(task_id,))
        monitor_thread.daemon = True
        monitor_thread.start()
        
    def _monitor_progress_thread(self, task_id):
        """Background thread to monitor progress via HTTP polling"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self._monitor_progress_async(task_id))
        finally:
            loop.close()
        
    async def _monitor_progress_async(self, task_id):
        """Async method to monitor progress via WebSocket"""
        uri = f"ws://localhost:8000/api/{API_VERSION}/ws/{task_id}"
        completed = False
        connection_attempts = 0
        max_connection_attempts = 3
        
        while connection_attempts < max_connection_attempts and self.running:
            try:
                async with websockets.connect(uri, ping_timeout=20, close_timeout=10) as websocket:
                    connection_attempts = 0  # Reset counter on successful connection
                    nuke.executeInMainThread(lambda: nuke.tprint("WebSocket connection established, monitoring progress..."))
                    
                    while self.running:
                        try:
                            data = await asyncio.wait_for(websocket.recv(), timeout=30)
                            progress_data = json.loads(data)
                            progress = int(progress_data.get('progress', 0))
                            message = progress_data.get('message', '')
                            status = progress_data.get('status', 'running')
                            nuke.executeInMainThread(lambda p=progress, m=message: self.on_progress(p, m))
                            if status in ['completed', 'failed']:
                                if status == 'completed':
                                    completed = True
                                    nuke.executeInMainThread(lambda: self.on_progress(100, "Processing completed successfully"))
                                if status == 'failed':
                                    error = progress_data.get('error', 'Unknown error')
                                    nuke.executeInMainThread(lambda e=error: self.on_error(f"Task failed: {e}"))
                                break
                        except asyncio.TimeoutError:
                            try:
                                pong_event = await websocket.ping()
                                await asyncio.wait_for(pong_event, timeout=5)
                                nuke.executeInMainThread(lambda: nuke.tprint("WebSocket ping successful, connection still active"))
                            except asyncio.TimeoutError:
                                nuke.executeInMainThread(lambda: nuke.tprint("WebSocket ping failed, reconnecting..."))
                                break
                            except Exception as ping_error:
                                nuke.executeInMainThread(lambda e=ping_error: nuke.tprint(f"Error pinging WebSocket: {e}"))
                                break
                        except Exception as e:
                            nuke.executeInMainThread(lambda e=e: nuke.tprint(f"Error receiving WebSocket data: {e}"))
                            await asyncio.sleep(2)
                    if completed:
                        return
            except Exception as connection_error:
                connection_attempts += 1
                nuke.executeInMainThread(lambda e=connection_error: 
                    nuke.tprint(f"WebSocket connection error (attempt {connection_attempts}/{max_connection_attempts}): {e}"))
                if connection_attempts < max_connection_attempts:
                    await asyncio.sleep(2)
                    continue
                else:
                    nuke.executeInMainThread(lambda: nuke.tprint("WebSocket connection failed, falling back to HTTP polling"))
                    break
        # HTTP polling fallback
        try:
            status_url = f"{API_BASE_URL}/api/{API_VERSION}/status/{task_id}"
            polling_start_time = time.time()
            max_polling_time = 300  # 5 minutes maximum
            while self.running and time.time() - polling_start_time < max_polling_time:
                try:
                    status_response = requests.get(status_url, timeout=5)
                    if status_response.status_code == 200:
                        progress_data = status_response.json()
                        progress = int(progress_data.get('progress', 0))
                        message = progress_data.get('message', '')
                        status = progress_data.get('status', 'running')
                        nuke.executeInMainThread(lambda p=progress, m=message: self.on_progress(p, m))
                        if status in ['completed', 'failed']:
                            if status == 'completed':
                                nuke.executeInMainThread(lambda: self.on_progress(100, "Processing completed successfully"))
                            if status == 'failed':
                                error = progress_data.get('error', 'Unknown error')
                                nuke.executeInMainThread(lambda e=error: self.on_error(f"Task failed: {e}"))
                            break
                    time.sleep(3)
                except Exception as e:
                    nuke.executeInMainThread(lambda e=e: nuke.tprint(f"HTTP polling error: {e}"))
                    time.sleep(3)
        except Exception as e:
            nuke.executeInMainThread(lambda e=e: nuke.tprint(f"Error checking task output: {e}"))

class Prompt:
    def __init__(self):
        self.frame_index = 0
        self.object_id = 0
        self.bboxes = []  # List of bounding boxes instead of single bbox
        self.points_positive = []
        self.points_negative = []
        self.input_path = None  # Store input path for dimension lookup

    def validate_data_types(self):
        """Ensure all data is in correct types and normalize coordinates"""
        # Get image dimensions
        if self.input_path:
            img = cv2.imread(self.input_path, cv2.IMREAD_UNCHANGED)
            if img is not None:
                height, width = img.shape[:2]
                
                # # Normalize all bboxes
                # normalized_bboxes = []
                # for bbox in self.bboxes:
                #     normalized_bbox = [
                #         float(bbox[0]) / width,   # normalized x1
                #         float(bbox[1]) / height,  # normalized y1
                #         float(bbox[2]) / width,   # normalized x2
                #         float(bbox[3]) / height   # normalized y2
                #     ]
                #     normalized_bboxes.append(normalized_bbox)
                # self.bboxes = normalized_bboxes
                
                # # Normalize points
                # self.points_positive = [[float(x)/width, float(y)/height] for x,y in self.points_positive]
                # self.points_negative = [[float(x)/width, float(y)/height] for x,y in self.points_negative]
            else:
                nuke.tprint(f"Warning: Could not read image at {self.input_path} for normalization")
        
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
    current_object_id = 0  # Track current object ID

    @classmethod
    def getBbox(cls):
        file_path = InputInfos.path
        input_file_name = str(os.path.splitext(os.path.basename(file_path))[0])
        
        # Initialize frame range
        cls.min_frame = int(nuke.thisNode().knob('FrameRangeMin').value())
        cls.max_frame = int(nuke.thisNode().knob('FrameRangeMax').value())
        cls.current_frame = cls.min_frame
        cls.current_object_id = 0  # Initialize current object ID

        def get_frame_path(frame_num):
            if "%04d" in input_file_name:
                return file_path.replace('%04d', str(f"{frame_num:04}"))
            elif "%03d" in input_file_name:
                return file_path.replace('%03d', str(f"{frame_num:03}"))
            return file_path

        cls.input_path = get_frame_path(cls.current_frame)
        
        window_name = "Selection - Left/Right arrows to change frame, Left click and drag for box, 'p' for positive point, 'n' for negative point, 'z' to undo, 'r' to reset, 'q' to finish, '1-9' to select object ID"
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
                        # Create new prompt or find existing one
                        prompt = None
                        for p in cls.prompts:
                            if p.frame_index == cls.current_frame and p.object_id == cls.current_object_id:
                                prompt = p
                                break
                        if prompt is None:
                            prompt = Prompt()
                            prompt.frame_index = cls.current_frame
                            prompt.object_id = cls.current_object_id
                            prompt.input_path = get_frame_path(cls.current_frame)  # Store input path
                            cls.prompts.append(prompt)
                        # Add bbox to prompt's list of bboxes
                        new_bbox = [cls.current_box[0], cls.current_box[1], 
                                  cls.current_box[0] + cls.current_box[2], 
                                  cls.current_box[1] + cls.current_box[3]]
                        prompt.bboxes.append(new_bbox)
                        nuke.tprint(f"Added box {len(prompt.bboxes)} for object {cls.current_object_id} on frame {cls.current_frame}: {new_bbox}")
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
                    # Draw all bboxes for this prompt
                    for i, bbox in enumerate(prompt.bboxes):
                        x1, y1, x2, y2 = bbox
                        # Use different colors for different object IDs
                        color = (0, 255, 0)  # Default green
                        if prompt.object_id == 1:
                            color = (255, 0, 0)  # Blue
                        elif prompt.object_id == 2:
                            color = (0, 0, 255)  # Red
                        elif prompt.object_id == 3:
                            color = (255, 255, 0)  # Cyan
                        elif prompt.object_id == 4:
                            color = (255, 0, 255)  # Magenta
                        elif prompt.object_id == 5:
                            color = (0, 255, 255)  # Yellow
                        elif prompt.object_id == 6:
                            color = (128, 0, 0)  # Dark Blue
                        elif prompt.object_id == 7:
                            color = (0, 128, 0)  # Dark Green
                        elif prompt.object_id == 8:
                            color = (0, 0, 128)  # Dark Red
                        elif prompt.object_id == 9:
                            color = (128, 128, 0)  # Dark Yellow
                        
                        cv2.rectangle(img_with_boxes, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(img_with_boxes, f"{prompt.object_id}.{i+1}", (x1, y1-5), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    
                    for point in prompt.points_positive:
                        cv2.circle(img_with_boxes, point, 5, (0, 255, 0), -1)
                        cv2.putText(img_with_boxes, "+", (point[0]-5, point[1]-5), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    
                    for point in prompt.points_negative:
                        cv2.circle(img_with_boxes, point, 5, (0, 0, 255), -1)
                        cv2.putText(img_with_boxes, "-", (point[0]-5, point[1]-5), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            # Draw current box being drawn
            if cls.drawing_box and cls.current_box:
                x, y, w, h = cls.current_box
                # Use color based on current object ID
                color = (0, 255, 0)  # Default green
                if cls.current_object_id == 1:
                    color = (255, 0, 0)  # Blue
                elif cls.current_object_id == 2:
                    color = (0, 0, 255)  # Red
                elif cls.current_object_id == 3:
                    color = (255, 255, 0)  # Cyan
                elif cls.current_object_id == 4:
                    color = (255, 0, 255)  # Magenta
                elif cls.current_object_id == 5:
                    color = (0, 255, 255)  # Yellow
                elif cls.current_object_id == 6:
                    color = (128, 0, 0)  # Dark Blue
                elif cls.current_object_id == 7:
                    color = (0, 128, 0)  # Dark Green
                elif cls.current_object_id == 8:
                    color = (0, 0, 128)  # Dark Red
                elif cls.current_object_id == 9:
                    color = (128, 128, 0)  # Dark Yellow
                cv2.rectangle(img_with_boxes, (x, y), (x + w, y + h), color, 2)

            # Create a semi-transparent overlay for better text visibility
            overlay = img_with_boxes.copy()
            cv2.rectangle(overlay, (0, 0), (400, 400), (0, 0, 0), -1)
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
            cv2.putText(img_with_boxes, "Press 'z' to undo last action", (10, 150), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(img_with_boxes, "Press 'r' to reset current frame", (10, 180), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(img_with_boxes, "Press 'q' to finish", (10, 210), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(img_with_boxes, "Press '1-9' to select object ID", (10, 240), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # Show current frame and selection counts
            current_prompts = [p for p in cls.prompts if p.frame_index == cls.current_frame]
            boxes_count = sum(len(p.bboxes) for p in current_prompts)
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
                    # Find the last prompt with boxes
                    last_prompt = None
                    for p in reversed(current_prompts):
                        if p.bboxes:
                            last_prompt = p
                            break
                    
                    if last_prompt:
                        # Remove the last box from the last prompt
                        last_prompt.bboxes.pop()
                        nuke.tprint(f"Undid last box for object {last_prompt.object_id} on frame {cls.current_frame}")
                        # If no boxes left, remove the prompt
                        if not last_prompt.bboxes and not last_prompt.points_positive and not last_prompt.points_negative:
                            cls.prompts.remove(last_prompt)
                    else:
                        # If no boxes, remove the last prompt
                        cls.prompts.remove(current_prompts[-1])
                        nuke.tprint(f"Undid last selection on frame {cls.current_frame}")
            
            elif key == ord('p'):  # Press 'p' to add positive point
                x, y = mouse_pos['current_pos']
                # Create new prompt or find existing one
                prompt = None
                for p in cls.prompts:
                    if p.frame_index == cls.current_frame and p.object_id == cls.current_object_id:
                        prompt = p
                        break
                if prompt is None:
                    prompt = Prompt()
                    prompt.frame_index = cls.current_frame
                    prompt.object_id = cls.current_object_id
                    cls.prompts.append(prompt)
                # Add point to prompt
                prompt.points_positive.append([x, y])
                nuke.tprint(f"Added positive point for object {cls.current_object_id} on frame {cls.current_frame}: {x = }, {y = }")
            
            elif key == ord('n'):  # Press 'n' to add negative point
                x, y = mouse_pos['current_pos']
                # Create new prompt or find existing one
                prompt = None
                for p in cls.prompts:
                    if p.frame_index == cls.current_frame and p.object_id == cls.current_object_id:
                        prompt = p
                        break
                if prompt is None:
                    prompt = Prompt()
                    prompt.frame_index = cls.current_frame
                    prompt.object_id = cls.current_object_id
                    cls.prompts.append(prompt)
                # Add point to prompt
                prompt.points_negative.append([x, y])
                nuke.tprint(f"Added negative point for object {cls.current_object_id} on frame {cls.current_frame}: {x = }, {y = }")
            
            elif key == 83 or key == 100:  # Right arrow key (→)
                if cls.current_frame < cls.max_frame:
                    cls.current_frame += 1
                    nuke.tprint(f"Moving to frame {cls.current_frame}")
            
            elif key == 81 or key == 97:  # Left arrow key (←)
                if cls.current_frame > cls.min_frame:
                    cls.current_frame -= 1
                    nuke.tprint(f"Moving to frame {cls.current_frame}")
            
            elif ord('1') <= key <= ord('9'):  # Number keys 1-9
                cls.current_object_id = key - ord('0')  # Convert key to 1-9
                nuke.tprint(f"Selected object ID: {cls.current_object_id}")
                # Update the node's CurrentObjectID knob
                try:
                    nuke.thisNode().knob('CurrentObjectID').setValue(cls.current_object_id)
                except:
                    pass

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
                
            prompts_json = json.dumps([
                {
                    "frame_index": p.frame_index,
                    "object_id": p.object_id,
                    "bbox": p.bboxes[0] if p.bboxes else None,
                    "points_positive": p.points_positive,
                    "points_negative": p.points_negative,
                    "input_path": p.input_path
                }
                for p in cls.prompts
            ])
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
                "bbox": prompt.bboxes[0] if prompt.bboxes else None,  # Send first bbox if exists
                "points_positive": prompt.points_positive,
                "points_negative": prompt.points_negative
            }
            prompts_data.append(prompt_dict)

        # Convert paths for API
        try:
            video_path = convert_windows_to_linux_path(nuke.thisNode().knob('FilePath').value())
            video_output_path = convert_windows_to_linux_path(nuke.thisNode().knob('OutputPath').getValue())
            
            # Send data to API
            response = requests.post(
                f"{API_BASE_URL}/api/{API_VERSION}/process_exr",
                json={
                    "prompts": prompts_data,
                    "sequence_path": video_path,
                    "output_path": video_output_path
                }
            )
            response.raise_for_status()
            result = response.json()
            nuke.tprint(f"API response: {result}")
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

def check_api_server():
    """Check if the API server is running and accessible"""
    try:
        response = requests.get(f"{API_BASE_URL}/api/{API_VERSION}/health")
        return response.status_code == 200
    except requests.exceptions.ConnectionError:
        return False

def upgrade_node_if_needed():
    """Add any missing knobs to existing nodes created with older versions"""
    try:
        node = nuke.thisNode()
        if node is None:
            nuke.tprint("Warning: thisNode() returned None, unable to upgrade node")
            return False
        
        # Track if we made any changes
        updated = False
        
        # Check if StatusMessage knob exists
        if not node.knob('StatusMessage'):
            nuke.tprint("Upgrading SAM2 node with new status message display")
            status_knob = nuke.Text_Knob('StatusMessage', 'Status')
            node.addKnob(status_knob)
            status_knob.setEnabled(False)
            status_knob.setValue("Idle - Ready to process")
            status_knob.setTooltip("Current processing status")
            updated = True
            
            # Position the knob after the Progress knob
            try:
                # Get the index of the Progress knob
                progress_index = node.knobs().keys().index('Progress')
                # Move the new knob to right after Progress
                for i in range(len(node.knobs()) - progress_index - 2):
                    status_knob.setFlag(nuke.INVISIBLE)
                    status_knob.clearFlag(nuke.INVISIBLE)
            except:
                nuke.tprint("Warning: Could not reposition StatusMessage knob")
        
        return updated
    except Exception as e:
        nuke.tprint(f"Error upgrading node: {str(e)}")
        return False

def normalize_path(path):
    """Normalize path separators to forward slashes for Nuke compatibility."""
    # Replace backslashes with forward slashes
    normalized = path.replace('\\', '/')
    # Remove any double slashes
    while '//' in normalized:
        normalized = normalized.replace('//', '/')
    return normalized

def convert_windows_to_linux_path(windows_path):
    """Convert Windows network path to Linux path format"""
    # Remove any @ symbol if present
    path = windows_path.lstrip('@')
    
    # Convert backslashes to forward slashes
    path = path.replace('\\', '/')
    
    # Handle network paths (\\server\share\path)
    if path.startswith('//'):
        # Remove the leading // and split into parts
        parts = path[2:].split('/')
        if len(parts) >= 2:
            server = parts[0]
            share = parts[1]
            # Convert to /mnt/share format
            return f"/mnt/shared/{'/'.join(parts[2:])}"
    
    # Handle local Windows paths (e.g., D:/path)
    if ':' in path:
        # Convert drive letter path to network share path
        drive, rest = path.split(':', 1)
        # Remove leading slash if present
        rest = rest.lstrip('/')
        # Convert to network share format
        return f"/mnt/shared/{rest}"
    
    return path

def convert_linux_to_windows_path(linux_path):
    """Convert Linux path to Windows network path format"""
    # Handle /mnt/share paths
    if linux_path.startswith('/mnt/'):
        parts = linux_path.split('/')
        if len(parts) >= 3:
            share = parts[2]
            # Convert to \\server\share format using raw string for backslashes
            remaining_path = '\\'.join(parts[3:])
            return f"\\\\192.168.12.166\\SharedData\\{remaining_path}"
    
    return linux_path

def GenerateMask():
    # Upgrade the node if needed (for backwards compatibility)
    upgrade_node_if_needed()
    
    Output_path = nuke.thisNode().knob('OutputPath').getValue()

    # Checks
    if str(os.path.splitext(os.path.basename(Output_path))[0]) == '':
        nuke.message("You must assign a file name")
        return
        
    if nuke.thisNode()['FilePath'].value().lower().endswith("mp4"):
        nuke.message('Unsupported input format. Input must be an Image Sequence')
        return
        
    if nuke.thisNode().knob('FileType').value() == "exr":
        if ("%04d" not in Output_path) and ("%03d" not in Output_path):
            nuke.message("Your file must contains '####' or '###'")
            return

    if not BoundingBox.prompts:
        nuke.message("No prompts added. Please add at least one bounding box or points.")
        return

    # Check if API server is running
    if not check_api_server():
        nuke.message("Error: API server is not running. Please ensure the server is running at " + API_BASE_URL)
        return

    # Reset progress bar and status message
    update_status_safely("Starting processing...")
    
    # Get all required parameters
    video_path = convert_windows_to_linux_path(nuke.thisNode().knob('FilePath').value())
    video_output_path = convert_windows_to_linux_path(nuke.thisNode().knob('OutputPath').getValue())
    save_to_file = nuke.thisNode().knob('FileType').value()
    frame_range = [int(nuke.thisNode().knob('FrameRangeMin').value()),
                  int(nuke.thisNode().knob('FrameRangeMax').value() + 1)]
    original_fps = int(InputInfos.original_fps)
    target_fps = int(nuke.thisNode().knob('FPS').value())
    bits = InputInfos.bits
    model_type_ui = nuke.thisNode().knob('ModelType').value().lower()
    model_type = 'base-plus' if model_type_ui == 'base' else model_type_ui

    # Create a threaded task for loading the model
    def load_model():
        nuke.tprint("Loading model in background...")
        response = requests.post(
            f"{API_BASE_URL}/api/{API_VERSION}/models/load",
            json={"model_type": model_type},
            timeout=30
        )
        response.raise_for_status()
        return model_type
    
    def on_model_loaded(model_type):
        nuke.tprint(f"Model {model_type} loaded successfully, starting processing...")
        # Start the sequence processing
        process_task.run_in_background(process_sequence)
    
    def on_model_error(error):
        nuke.message(f"Failed to load model: {error}")
    
    # Function to process the sequence after model is loaded
    def process_sequence():
        # Validate all prompts to ensure correct data types
        validated_prompts = []
        for prompt in BoundingBox.prompts:
            # Validate data types before sending
            prompt.validate_data_types()
            prompt_dict = {
                "frame_index": prompt.frame_index,
                "object_id": prompt.object_id,
                "bbox": prompt.bboxes[0] if prompt.bboxes else None,  # Send first bbox if exists
                "points_positive": prompt.points_positive,
                "points_negative": prompt.points_negative
            }
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
        nuke.tprint(f"Using paths - Input: {video_path}")
        nuke.tprint(f"Request data: {json.dumps(request_data, indent=2)}")  # Debug print
        
        # Send request to process_sequence endpoint
        response = requests.post(
            f"{API_BASE_URL}/api/{API_VERSION}/process_sequence?as_file=true",
            json=request_data,
            timeout=300  # 5 minutes timeout
        )
        response.raise_for_status()
        
        result = response.json()
        nuke.tprint(f"Received response: {json.dumps(result, indent=2)}")
        
        # If we get a task_id, start monitoring progress
        if "task_id" in result:
            task_id = result["task_id"]
            nuke.tprint(f"Processing started with task ID: {task_id}")
            process_task.start_progress_monitor(task_id)
            
            # Poll for completion and download results
            output_endpoint = f"{API_BASE_URL}/api/{API_VERSION}/output/{task_id}"
            max_attempts = 60  # 5 minutes at 5-second intervals
            attempt = 0
            error_count = 0
            max_error_count = 10
            while attempt < max_attempts and process_task.running:
                attempt += 1
                try:
                    output_response = requests.get(output_endpoint, timeout=5)
                    if output_response.status_code == 200:
                        error_count = 0  # Reset error count on success
                        output_data = output_response.json()
                        
                        # Find the ZIP file
                        zip_file = None
                        for file_info in output_data.get("files", []):
                            if file_info["filename"].endswith(".zip"):
                                zip_file = file_info
                                break
                                
                        if zip_file:
                            # Found ZIP file, download it
                            download_url = f"{API_BASE_URL}/api/{API_VERSION}/download/{zip_file['filename']}"
                            nuke.executeInMainThread(lambda: update_status_safely("Preparing to download result files..."))
                            
                            # Get file size if available for progress tracking
                            file_size = zip_file.get('size', 0)
                            
                            # Download with progress tracking
                            nuke.executeInMainThread(lambda: update_status_safely("Downloading result files..."))
                            download_response = requests.get(download_url, stream=True, timeout=60)
                            download_response.raise_for_status()
                            
                            # Create output directory
                            output_dir = os.path.dirname(video_output_path)
                            if output_dir and not os.path.exists(output_dir):
                                os.makedirs(output_dir, exist_ok=True)
                            
                            # Save to file with progress tracking
                            total_downloaded = 0
                            with open(video_output_path, 'wb') as f:
                                for i, chunk in enumerate(download_response.iter_content(chunk_size=8192)):
                                    f.write(chunk)
                                    total_downloaded += len(chunk)
                                    # Update progress every 20 chunks
                                    if i % 20 == 0:
                                        if file_size > 0:
                                            # Calculate download progress (90-95%)
                                            download_progress = 90 + (total_downloaded / file_size) * 5
                                            nuke.executeInMainThread(lambda p=download_progress: update_status_safely(f"Download progress: {int(p)}%"))
                                        else:
                                            # If file size unknown, just show intermediate progress
                                            nuke.executeInMainThread(lambda: update_status_safely("Downloading..."))
            
                            # Always make sure we're at 95% after download
                            nuke.executeInMainThread(lambda: update_status_safely("Download complete, extracting files..."))
                            
                            # Extract if needed
                            if not video_output_path.lower().endswith('.zip'):
                                extract_dir = os.path.dirname(video_output_path)
                                with zipfile.ZipFile(video_output_path, 'r') as zip_ref:
                                    # Get total number of files for progress tracking
                                    file_count = len(zip_ref.infolist())
                                    
                                    # Extract files with progress tracking
                                    for i, file in enumerate(zip_ref.infolist()):
                                        zip_ref.extract(file, extract_dir)
                                        # Update extraction progress (95-98%)
                                        if i % max(1, file_count // 10) == 0:  # Update 10 times total
                                            extraction_progress = 95 + (i / file_count) * 3
                                            nuke.executeInMainThread(lambda p=extraction_progress: update_status_safely(f"Extraction progress: {int(p)}%"))
                                        
                                        # Show filename being extracted every 10 files
                                        if i % 10 == 0:
                                            nuke.executeInMainThread(lambda f=file.filename: 
                                                update_status_safely(f"Extracting: {f}"))
                                
                                # Delete the ZIP file
                                os.remove(video_output_path)
                                nuke.executeInMainThread(lambda: update_status_safely("Extraction complete"))
                            
                            # Always make sure we're at 98% after extraction
                            nuke.executeInMainThread(lambda: update_status_safely("Setting up read node..."))
                            
                            # Create a read node in the main thread
                            def create_read_node():
                                if video_output_path.endswith('.zip'):
                                    normalized_path = normalize_path(video_output_path)
                                    read_node = nuke.nodes.Read(file=normalized_path, first=frame_range[0], last=frame_range[1]-1)
                                else:
                                    # Find mask pattern
                                    extract_dir = os.path.dirname(video_output_path)
                                    mask_pattern = os.path.join(extract_dir, "mask_%04d.exr")
                                    # Normalize the path
                                    normalized_pattern = normalize_path(mask_pattern)
                                    read_node = nuke.nodes.Read(file=normalized_pattern, first=0, last=frame_range[1]-frame_range[0]-1)
                                
                                # Mark as 100% complete
                                update_status_safely("Processing completed successfully")
                                
                                nuke.message("Mask generation completed! Files saved and loaded into Nuke.")
                                
                            nuke.executeInMainThread(create_read_node)
                            return
                        else:
                            # No ZIP file yet, keep waiting
                            time.sleep(5)
                    elif output_response.status_code in [404, 503]:
                        error_count += 1
                        if error_count >= max_error_count:
                            nuke.executeInMainThread(lambda: nuke.message(f"Task output unavailable (status {output_response.status_code}) for {error_count} attempts. Stopping."))
                            break
                        time.sleep(5)
                    else:
                        time.sleep(5)
                except Exception as e:
                    nuke.tprint(f"Error checking task output: {str(e)}")
                    time.sleep(5)
            
            # If we get here, we haven't found a ZIP file yet
            # Try the download_all endpoint as a last resort
            try:
                download_url = f"{API_BASE_URL}/api/{API_VERSION}/download_all/{task_id}"
                nuke.executeInMainThread(lambda: update_status_safely("Trying direct download of all result files..."))
                
                download_response = requests.get(download_url, stream=True, timeout=60)
                
                if download_response.status_code == 200:
                    # Create output directory
                    output_dir = os.path.dirname(video_output_path)
                    if output_dir and not os.path.exists(output_dir):
                        os.makedirs(output_dir, exist_ok=True)
                    
                    # Save all files as ZIP with progress tracking
                    total_size = int(download_response.headers.get('content-length', 0))
                    total_downloaded = 0
                    
                    with open(video_output_path, 'wb') as f:
                        for i, chunk in enumerate(download_response.iter_content(chunk_size=8192)):
                            f.write(chunk)
                            total_downloaded += len(chunk)
                            # Update progress every 20 chunks
                            if i % 20 == 0:
                                if total_size > 0:
                                    # Calculate download progress (90-95%)
                                    download_progress = 90 + (total_downloaded / total_size) * 5
                                    nuke.executeInMainThread(lambda p=download_progress: update_status_safely(f"Download progress: {int(p)}%"))
                                else:
                                    # If file size unknown, just show intermediate progress
                                    nuke.executeInMainThread(lambda: update_status_safely("Downloading..."))
                    
                    nuke.executeInMainThread(lambda: update_status_safely("Direct download complete, extracting files..."))
                    
                    # Extract if needed
                    if not video_output_path.lower().endswith('.zip'):
                        extract_dir = os.path.dirname(video_output_path)
                        with zipfile.ZipFile(video_output_path, 'r') as zip_ref:
                            # Get total number of files for progress tracking
                            file_count = len(zip_ref.infolist())
                            
                            # Extract files with progress tracking
                            for i, file in enumerate(zip_ref.infolist()):
                                zip_ref.extract(file, extract_dir)
                                # Update extraction progress (95-98%)
                                if i % max(1, file_count // 10) == 0:  # Update 10 times total
                                    extraction_progress = 95 + (i / file_count) * 3
                                    nuke.executeInMainThread(lambda p=extraction_progress: update_status_safely(f"Extraction progress: {int(p)}%"))
                                
                                # Show filename being extracted every 10 files
                                if i % 10 == 0:
                                    nuke.executeInMainThread(lambda f=file.filename: 
                                        update_status_safely(f"Extracting: {f}"))
                        
                        # Delete the ZIP file
                        os.remove(video_output_path)
                    
                    nuke.executeInMainThread(lambda: update_status_safely("Finalizing..."))
                    
                    # Create read node
                    def create_read_node():
                        if video_output_path.endswith('.zip'):
                            normalized_path = normalize_path(video_output_path)
                            read_node = nuke.nodes.Read(file=normalized_path, first=frame_range[0], last=frame_range[1]-1)
                        else:
                            # Find mask pattern
                            extract_dir = os.path.dirname(video_output_path)
                            mask_pattern = os.path.join(extract_dir, "mask_%04d.exr")
                            # Normalize the path
                            normalized_pattern = normalize_path(mask_pattern)
                            read_node = nuke.nodes.Read(file=normalized_pattern, first=0, last=frame_range[1]-frame_range[0]-1)
                        nuke.message("Mask generation completed! Files saved and loaded into Nuke.")
                    
                    nuke.executeInMainThread(create_read_node)
            except Exception as e:
                nuke.tprint(f"Error downloading all files: {str(e)}")
                nuke.executeInMainThread(lambda e=e: nuke.message(f"Error downloading result: {str(e)}"))
                # Mark error and set the error flag
                process_task.report_error(f"Error downloading result: {str(e)}")
        
        # If we get here with a result but no task_id, it's an immediate result
        elif "result" in result:
            nuke.tprint("Processing completed successfully with immediate result")
            
            # Handle different response formats
            if save_to_file and "output_dir" in result and "zip_path" in result:
                # Server saved files, download the ZIP
                zip_path = result["zip_path"]
                try:
                    download_url = f"{API_BASE_URL}/api/{API_VERSION}/download/{os.path.basename(zip_path)}"
                    download_response = requests.get(download_url, stream=True)
                    
                    # Create output directory
                    output_dir = os.path.dirname(video_output_path)
                    if output_dir and not os.path.exists(output_dir):
                        os.makedirs(output_dir, exist_ok=True)
                    
                    # Save to file
                    with open(video_output_path, 'wb') as f:
                        for chunk in download_response.iter_content(chunk_size=8192):
                            f.write(chunk)
                    
                    # Extract if needed
                    if not video_output_path.lower().endswith('.zip'):
                        extract_dir = os.path.dirname(video_output_path)
                        with zipfile.ZipFile(video_output_path, 'r') as zip_ref:
                            zip_ref.extractall(extract_dir)
                        
                        # Delete the ZIP file
                        os.remove(video_output_path)
                    
                    # Create read node in main thread
                    def create_read_node():
                        if video_output_path.endswith('.zip'):
                            normalized_path = normalize_path(video_output_path)
                            read_node = nuke.nodes.Read(file=normalized_path, first=frame_range[0], last=frame_range[1]-1)
                        else:
                            # Find mask pattern
                            extract_dir = os.path.dirname(video_output_path)
                            mask_pattern = os.path.join(extract_dir, "mask_%04d.exr")
                            # Normalize the path
                            normalized_pattern = normalize_path(mask_pattern)
                            read_node = nuke.nodes.Read(file=normalized_pattern, first=0, last=frame_range[1]-frame_range[0]-1)
                        nuke.message("Mask generation completed! Files saved and loaded into Nuke.")
                    
                    nuke.executeInMainThread(create_read_node)
                except Exception as e:
                    nuke.tprint(f"Error downloading result: {str(e)}")
                    nuke.executeInMainThread(lambda: nuke.message(f"Error downloading result: {str(e)}"))
    
    # Function to handle processing errors
    def on_process_error(error):
        nuke.message(f"Error during processing: {error}")
    
    # Create the threaded task for sequence processing
    process_task = ThreadedTask(
        on_progress=lambda p, m: update_status_safely(f"Progress: {int(p)}% - {m}"),
        on_error=on_process_error
    )
    
    # Create the threaded task for model loading
    model_task = ThreadedTask(
        on_complete=on_model_loaded,
        on_error=on_model_error
    )
    
    # Start loading the model in background
    model_task.run_in_background(load_model)
    
    nuke.tprint("Started background processing. Nuke will remain responsive.")

def reset_model_state():
    """Reset the model state on the server and clear UI state."""
    try:
        # Update status
        update_status_safely("Resetting model state...")
        
        # Make API call to reset endpoint
        response = requests.post(f"{API_BASE_URL}/api/{API_VERSION}/reset")
        try:
            response.raise_for_status()  # Raise exception for non-200 status codes
        except Exception as e:
            # Log the error and show backend response
            error_detail = response.text
            nuke.tprint(f"Reset API call failed: {e}, Response: {error_detail}")
            nuke.message(f"Failed to reset model state. Server response: {error_detail}")
            update_status_safely(f"Error: Failed to reset model state. Server response: {error_detail}")
            return
        
        # Log the successful response for debugging
        nuke.tprint(f"Reset API call succeeded: {response.status_code}, Response: {response.text}")
        
        # Reset local UI state
        update_status_safely("Model state reset successfully")
        BoundingBox.clearPrompts()
        
        # Reset object ID to 0
        try:
            node = nuke.thisNode()
            if node:
                node.knob('CurrentObjectID').setValue(0)
        except:
            pass
            
        nuke.message("Model state reset successfully")
            
    except requests.exceptions.ConnectionError:
        error_msg = "Failed to connect to API server. Please ensure the server is running."
        nuke.tprint(error_msg)
        nuke.message(error_msg)
        update_status_safely(f"Error: {error_msg}")
    except Exception as e:
        error_msg = f"Failed to reset model state: {str(e)}"
        nuke.tprint(error_msg)
        nuke.message(error_msg)
        update_status_safely(f"Error: {error_msg}")

def CreateSAM2Node():
    # Creating node
    nuke.createNode('NoOp')
    s = nuke.selectedNode()

    # Adding knobs
    s.knob('name').setValue('SAM2')
    s.addKnob(nuke.File_Knob('FilePath', 'File Path'))
    s.addKnob(nuke.PyScript_Knob('UpdatePath', 'Update Path', 'UpdatePath()'))
    
    # Frame selection
    s.addKnob(nuke.Int_Knob("FrameRangeMin", 'Frame Range'))
    s.addKnob(nuke.Int_Knob("FrameRangeMax", ' '))
    s.addKnob(nuke.Int_Knob("FPS", 'Output Frame Rate'))
    
    # Model selection - user-friendly names
    s.addKnob(nuke.Enumeration_Knob('ModelType', 'Model type', ['Base', 'Large', 'Small', 'Tiny']))
    
    # Object Selection - improved with dropdown
    s.addKnob(nuke.Double_Knob('CurrentObjectID', 'Object ID'))
    s.knob('CurrentObjectID').setRange(0, 10)  # Allow up to 10 objects
    s.knob('CurrentObjectID').setValue(0)
    s.knob('CurrentObjectID').setTooltip("ID of the object to add selections to (0-10)")
    
    # Status Message
    status_knob = nuke.Text_Knob('StatusMessage', 'Status')
    status_knob.setEnabled(False)
    status_knob.setValue("Idle - Ready to process")
    s.addKnob(status_knob)
    
    # Add divider
    s.addKnob(nuke.Text_Knob('', ''))
    
    # Selection controls
    s.addKnob(nuke.PyScript_Knob('CreateBoundingBox', 'Create Selection', 'BoundingBox.getBbox()'))
    s.addKnob(nuke.PyScript_Knob('ClearPrompts', 'Clear All Selections', 'BoundingBox.clearPrompts()'))
    s.addKnob(nuke.Text_Knob('PromptsList', 'Prompts List'))
    s.knob('PromptsList').setEnabled(False)

    # Add divider
    s.addKnob(nuke.Text_Knob('', ''))
    
    # Output controls
    s.addKnob(nuke.Enumeration_Knob('FileType', 'File type', ['exr', 'mp4']))
    s.addKnob(nuke.File_Knob('OutputPath', 'Output Path'))
    s.addKnob(nuke.PyScript_Knob('GenerateMask', 'Generate Mask', 'GenerateMask()'))
    
    # Add divider
    s.addKnob(nuke.Text_Knob('', ''))
    
    # Reset Model State
    s.addKnob(nuke.PyScript_Knob('ResetState', 'Reset Model State', 'reset_model_state()'))
    
    # Setting default values and tooltips
    s['FPS'].setValue(int(nuke.root().knob('fps').getValue()))
    s['FrameRangeMin'].setValue(int(nuke.Root()['first_frame'].value()))
    s['FrameRangeMax'].setValue(int(nuke.Root()['last_frame'].value()))
    s['StatusMessage'].setValue("Idle - Ready to process")

    s['FPS'].setFlag(nuke.STARTLINE)
    s['FrameRangeMax'].clearFlag(nuke.STARTLINE)
    s['UpdatePath'].setFlag(nuke.STARTLINE)
    s['GenerateMask'].setFlag(nuke.STARTLINE)
    s['ResetState'].setFlag(nuke.STARTLINE)
    
    s['CreateBoundingBox'].setTooltip("Create selection. Use mouse to draw boxes, 'p' for positive points, 'n' for negative points, arrows to change frame.")
    s['ClearPrompts'].setTooltip("Clear all selections")
    s['ResetState'].setTooltip("Reset model state and clear GPU memory")
    s['FPS'].setTooltip("Target FPS for the output video")
    s['ModelType'].setTooltip("Choose your model type: Base, Large, Small, or Tiny")
    s['OutputPath'].setTooltip("path/to/your/file_####.exr, to create an image sequence add #### or ###")
    s['GenerateMask'].setTooltip("Generate Mask")
    s['StatusMessage'].setTooltip("Current processing status")

def update_status_safely(message):
    """Safely update status message, handling cases where knob might be missing"""
    try:
        node = nuke.thisNode()
        if node is None:
            nuke.tprint(f"Status update: {message}")
            return False
        
        status_knob = node.knob('StatusMessage')
        if status_knob is None:
            nuke.tprint(f"Status update: {message}")
            return False
        
        status_knob.setValue(str(message))
        return True
    except Exception as e:
        nuke.tprint(f"Status update: {message}")
        return False

def reset_ui_state():
    """Reset the UI state (progress bar and status message)"""
    update_status_safely("Idle - Ready to process")
    nuke.tprint("Reset UI state")

async def monitor_progress(task_id):
    """This function is kept for backward compatibility but is no longer used.
       Progress monitoring is now handled by the ThreadedTask class."""
    nuke.tprint("Warning: Deprecated monitor_progress function called. Please update your code to use ThreadedTask.")
    pass

# Register functions for Nuke PyScript_Knobs
import __main__
__main__.reset_model_state = reset_model_state
__main__.CreateSAM2Node = CreateSAM2Node
# (Add any other functions you want to call from knobs) 