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
import re
import tempfile
import numpy as np

# API Configuration
API_BASE_URL = "http://localhost:8000"  # Default URL
API_VERSION = "v1"
_current_api_url = API_BASE_URL  # Class variable to store current API URL

def set_api_base_url():
    """Set the API base URL from the node's configuration"""
    global _current_api_url
    try:
        node = nuke.thisNode()
        if node and node.knob('APIBaseURL'):
            url = node.knob('APIBaseURL').value()
            if url and url.strip():
                # Validate URL format
                if not url.startswith(('http://', 'https://')):
                    url = 'http://' + url
                _current_api_url = url.strip()
                nuke.tprint(f"API Base URL set to: {_current_api_url}")
                return
    except:
        pass
    # Reset to default if empty or invalid
    _current_api_url = API_BASE_URL
    nuke.tprint(f"API Base URL reset to default: {_current_api_url}")

def get_api_base_url():
    """Get the current API base URL"""
    global _current_api_url
    try:
        node = nuke.thisNode()
        if node and node.knob('APIBaseURL'):
            url = node.knob('APIBaseURL').value()
            if url and url.strip():
                # Validate URL format
                if not url.startswith(('http://', 'https://')):
                    url = 'http://' + url
                _current_api_url = url.strip()
                return _current_api_url
    except:
        pass
    return _current_api_url

def start_docker_server():
    """Start the Docker container using docker compose"""
    try:
        # Get the directory containing the Nuke plugin script
        plugin_dir = os.path.dirname(os.path.abspath(__file__))
        docker_compose_path = os.path.join(plugin_dir, 'docker-compose.yml')
        
        if not os.path.exists(docker_compose_path):
            nuke.message("Error: docker-compose.yml not found in plugin directory.\nPlease ensure docker-compose.yml is in the same directory as NukeSAM2.py")
            return False
            
        # Check if Docker is running
        try:
            import subprocess
            process = subprocess.run(['docker', 'info'], capture_output=True, text=True)
            if process.returncode != 0:
                nuke.message("Error: Docker is not running. Please start Docker Desktop first.")
                return False
        except Exception as e:
            nuke.message("Error: Docker is not installed or not accessible. Please ensure Docker is installed and running.")
            return False
            
        # Start the container in a background thread
        def run_docker_compose():
            try:
                import subprocess
                # First check if the container is already running
                process = subprocess.run(
                    ['docker', 'compose', 'ps', '--format', 'json'],
                    cwd=plugin_dir,
                    capture_output=True,
                    text=True
                )
                
                if process.returncode == 0 and process.stdout.strip():
                    nuke.executeInMainThread(lambda: nuke.message("Docker server is already running!"))
                    nuke.executeInMainThread(lambda: update_status_safely("Server: Running"))
                    return
                
                # Start the container
                process = subprocess.Popen(
                    ['docker', 'compose', 'up', '-d'],
                    cwd=plugin_dir,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )
                stdout, stderr = process.communicate()
                
                if process.returncode == 0:
                    nuke.executeInMainThread(lambda: nuke.message("Docker server started successfully!"))
                    nuke.executeInMainThread(lambda: update_status_safely("Server: Running"))
                else:
                    error_msg = f"Failed to start Docker server: {stderr}"
                    nuke.executeInMainThread(lambda: nuke.message(error_msg))
                    nuke.executeInMainThread(lambda: update_status_safely("Server: Failed to start"))
            except Exception as e:
                error_msg = f"Error starting Docker server: {str(e)}"
                nuke.executeInMainThread(lambda: nuke.message(error_msg))
                nuke.executeInMainThread(lambda: update_status_safely("Server: Error"))
        
        # Start the thread
        thread = threading.Thread(target=run_docker_compose)
        thread.daemon = True
        thread.start()
        
        return True
    except Exception as e:
        nuke.message(f"Error starting Docker server: {str(e)}")
        return False

def stop_docker_server():
    """Stop the Docker container using docker compose"""
    try:
        # Get the directory containing the Nuke plugin script
        plugin_dir = os.path.dirname(os.path.abspath(__file__))
        docker_compose_path = os.path.join(plugin_dir, 'docker-compose.yml')
        
        if not os.path.exists(docker_compose_path):
            nuke.message("Error: docker-compose.yml not found in plugin directory.\nPlease ensure docker-compose.yml is in the same directory as NukeSAM2.py")
            return False
            
        # Stop the container in a background thread
        def run_docker_compose_down():
            try:
                import subprocess
                # First check if the container is running
                process = subprocess.run(
                    ['docker', 'compose', 'ps', '--format', 'json'],
                    cwd=plugin_dir,
                    capture_output=True,
                    text=True
                )
                
                if process.returncode != 0 or not process.stdout.strip():
                    nuke.executeInMainThread(lambda: nuke.message("Docker server is not running!"))
                    nuke.executeInMainThread(lambda: update_status_safely("Server: Stopped"))
                    return
                
                # Stop the container
                process = subprocess.Popen(
                    ['docker', 'compose', 'down'],
                    cwd=plugin_dir,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )
                stdout, stderr = process.communicate()
                
                if process.returncode == 0:
                    nuke.executeInMainThread(lambda: nuke.message("Docker server stopped successfully!"))
                    nuke.executeInMainThread(lambda: update_status_safely("Server: Stopped"))
                else:
                    error_msg = f"Failed to stop Docker server: {stderr}"
                    nuke.executeInMainThread(lambda: nuke.message(error_msg))
                    nuke.executeInMainThread(lambda: update_status_safely("Server: Failed to stop"))
            except Exception as e:
                error_msg = f"Error stopping Docker server: {str(e)}"
                nuke.executeInMainThread(lambda: nuke.message(error_msg))
                nuke.executeInMainThread(lambda: update_status_safely("Server: Error"))
        
        # Start the thread
        thread = threading.Thread(target=run_docker_compose_down)
        thread.daemon = True
        thread.start()
        
        return True
    except Exception as e:
        nuke.message(f"Error stopping Docker server: {str(e)}")
        return False

def check_server_status():
    """Check if the Docker container is running"""
    try:
        import subprocess
        # Get the directory containing the Nuke plugin script
        plugin_dir = os.path.dirname(os.path.abspath(__file__))
        
        process = subprocess.run(
            ['docker', 'compose', 'ps', '--format', 'json'],
            cwd=plugin_dir,
            capture_output=True,
            text=True
        )
        
        if process.returncode == 0 and process.stdout.strip():
            return True
        return False
    except:
        return False

class PollingMonitor:
    def __init__(self, task_id, on_progress=None, on_error=None, on_complete=None):
        self.task_id = task_id
        self.on_progress = on_progress or (lambda p, m: None)
        self.on_error = on_error or (lambda e: None)
        self.on_complete = on_complete or (lambda: None)
        self.running = False
        self.last_progress = 0
        self.consecutive_errors = 0
        self.max_consecutive_errors = 5
        self.base_polling_interval = 1.0  # Base interval in seconds
        self.min_polling_interval = 0.5   # Minimum interval
        self.max_polling_interval = 5.0   # Maximum interval
        self.progress_threshold = 0.1     # Progress threshold for interval adjustment
        self.last_poll_time = time.time()
        self.start_time = time.time()
        self.max_polling_duration = 3600  # Maximum polling duration (1 hour)
        self.status_check_count = 0
        self.max_status_checks = 60  # Maximum number of status checks after completion
        self.files_ready = False

    def calculate_polling_interval(self, current_progress):
        """Calculate adaptive polling interval based on progress"""
        if current_progress == self.last_progress:
            # If no progress, gradually increase interval up to max
            return min(self.base_polling_interval * (1.1 ** self.consecutive_errors), 
                      self.max_polling_interval)
        else:
            # If progress made, decrease interval down to min
            progress_delta = current_progress - self.last_progress
            if progress_delta > self.progress_threshold:
                return max(self.base_polling_interval * 0.8, self.min_polling_interval)
            return self.base_polling_interval

    def poll_status(self):
        """Poll the task status with adaptive intervals"""
        self.running = True
        self.start_time = time.time()

        while self.running:
            try:
                # Check if we've exceeded maximum polling duration
                if time.time() - self.start_time > self.max_polling_duration:
                    self.on_error("Polling timeout exceeded maximum duration")
                    break

                # Calculate time since last poll
                time_since_last_poll = time.time() - self.last_poll_time
                current_interval = self.calculate_polling_interval(self.last_progress)

                # Only poll if enough time has passed
                if time_since_last_poll >= current_interval:
                    response = requests.get(
                        f"{API_BASE_URL}/api/{API_VERSION}/status/{self.task_id}",
                        timeout=5
                    )
                    
                    if response.status_code == 200:
                        data = response.json()
                        current_progress = int(data.get('progress', 0))
                        message = data.get('message', '')
                        status = data.get('status', 'running')

                        # Update progress
                        self.on_progress(current_progress, message)
                        self.last_progress = current_progress
                        self.consecutive_errors = 0

                        # Check for completion
                        if status in ['completed', 'failed']:
                            if status == 'completed':
                                # Check if files are ready
                                try:
                                    output_response = requests.get(f"{API_BASE_URL}/api/{API_VERSION}/output/{self.task_id}")
                                    if output_response.status_code == 200:
                                        self.files_ready = True
                                        self.on_complete()
                                        break
                                    else:
                                        # Increment status check counter
                                        self.status_check_count += 1
                                        if self.status_check_count >= self.max_status_checks:
                                            self.on_error("Maximum status checks exceeded after completion")
                                            break
                                except Exception as e:
                                    self.on_error(f"Error checking file status: {str(e)}")
                                    break
                            else:
                                error = data.get('error', 'Unknown error')
                                self.on_error(f"Task failed: {error}")
                                break

                    else:
                        self.consecutive_errors += 1
                        if self.consecutive_errors >= self.max_consecutive_errors:
                            self.on_error(f"Too many consecutive errors ({self.consecutive_errors})")
                            break
                        self.on_progress(self.last_progress, f"Error: HTTP {response.status_code}")

                # Sleep for a short time to prevent CPU overuse
                time.sleep(0.1)

            except requests.exceptions.RequestException as e:
                self.consecutive_errors += 1
                if self.consecutive_errors >= self.max_consecutive_errors:
                    self.on_error(f"Connection error: {str(e)}")
                    break
                self.on_progress(self.last_progress, f"Connection error: {str(e)}")
                time.sleep(current_interval)

            except Exception as e:
                self.on_error(f"Unexpected error: {str(e)}")
                break

        self.running = False

    def stop(self):
        """Stop the polling monitor"""
        self.running = False

def start_polling_monitor(task_id, on_progress=None, on_error=None, on_complete=None):
    """Start a new polling monitor in a background thread"""
    monitor = PollingMonitor(task_id, on_progress, on_error, on_complete)
    thread = threading.Thread(target=monitor.poll_status)
    thread.daemon = True
    thread.start()
    return monitor

class WebSocketMonitor:
    def __init__(self, task_id, on_progress=None, on_error=None, on_complete=None):
        self.task_id = task_id
        self.on_progress = on_progress or (lambda p, m: None)
        self.on_error = on_error or (lambda e: None)
        self.on_complete = on_complete or (lambda: None)
        self.connection_lost = False
        self.last_progress = 0
        self.last_message = ""
        self.websocket = None
        self.polling_monitor = None
        self.running = True
        self.completed = False
        self.files_ready = False
        self.retry_count = 0
        self.max_retries = 3
        self.status_check_count = 0
        self.max_status_checks = 60  # Maximum number of status checks after completion
        self.download_attempts = 0
        self.max_download_attempts = 3

    async def start(self):
        """Start monitoring with WebSocket, falling back to HTTP polling if needed"""
        try:
            ws_url = f"ws://{API_BASE_URL.replace('http://', '')}/api/{API_VERSION}/ws/{self.task_id}"
            nuke.executeInMainThread(lambda: nuke.tprint(f"Attempting WebSocket connection to {ws_url}"))
            self.websocket = await connect_with_retry(ws_url)
            if not self.websocket:
                nuke.executeInMainThread(lambda: nuke.tprint("WebSocket connection failed, starting HTTP polling"))
                self.start_polling()
                return

            nuke.executeInMainThread(lambda: nuke.tprint("WebSocket connection established, monitoring progress..."))
            
            while self.running and not self.connection_lost:
                try:
                    data = await asyncio.wait_for(self.websocket.recv(), timeout=30)
                    await self.handle_message(data)
                    # If files are ready, exit the monitoring loop
                    if self.files_ready:
                        break
                except asyncio.TimeoutError:
                    if not await self.check_connection():
                        nuke.executeInMainThread(lambda: nuke.tprint("WebSocket connection lost, falling back to HTTP polling"))
                        self.connection_lost = True
                        self.start_polling()
                        break
                except websockets.exceptions.ConnectionClosed as e:
                    nuke.executeInMainThread(lambda e=e: nuke.tprint(f"WebSocket connection closed: {e}"))
                    if not self.files_ready:
                        if self.retry_count < self.max_retries:
                            self.retry_count += 1
                            nuke.executeInMainThread(lambda: nuke.tprint(f"Attempting to reconnect (attempt {self.retry_count}/{self.max_retries})..."))
                            await asyncio.sleep(2 ** self.retry_count)  # Exponential backoff
                            self.websocket = await connect_with_retry(ws_url)
                            if self.websocket:
                                self.connection_lost = False
                                continue
                        self.connection_lost = True
                        self.start_polling()
                    break
                except Exception as e:
                    nuke.executeInMainThread(lambda e=e: nuke.tprint(f"WebSocket error: {e}"))
                    if not self.files_ready:
                        self.connection_lost = True
                        self.start_polling()
                    break

        except Exception as e:
            nuke.executeInMainThread(lambda e=e: nuke.tprint(f"WebSocket connection error: {e}"))
            if not self.files_ready:
                self.connection_lost = True
                self.start_polling()

    async def handle_message(self, data):
        """Handle incoming WebSocket message"""
        try:
            progress_data = json.loads(data)
            self.last_progress = int(progress_data.get('progress', 0))
            self.last_message = progress_data.get('message', '')
            status = progress_data.get('status', 'running')
            
            # Update progress
            self.on_progress(self.last_progress, self.last_message)
            
            # Handle completion states
            if status == "completed":
                self.completed = True
                # Check if files are ready
                try:
                    response = requests.get(f"{API_BASE_URL}/api/{API_VERSION}/output/{self.task_id}")
                    if response.status_code == 200:
                        self.files_ready = True
                        nuke.executeInMainThread(lambda: nuke.tprint("Files are ready for download"))
                        # Don't call on_complete here, let the download process handle it
                        await self.stop()
                        return  # Exit the monitoring loop
                    else:
                        nuke.executeInMainThread(lambda: nuke.tprint(f"Waiting for files to be ready... (status: {response.status_code})"))
                except requests.exceptions.RequestException as e:
                    nuke.executeInMainThread(lambda e=e: nuke.tprint(f"Error checking file status: {e}"))
                    self.on_error(f"Error checking file status: {str(e)}")
                except Exception as e:
                    nuke.executeInMainThread(lambda e=e: nuke.tprint(f"Unexpected error checking file status: {e}"))
                    self.on_error(f"Unexpected error: {str(e)}")
            elif status == "failed":
                error = progress_data.get('error', 'Unknown error')
                self.on_error(f"Task failed: {error}")
                self.running = False
                await self.stop()
        except json.JSONDecodeError as e:
            nuke.executeInMainThread(lambda e=e: nuke.tprint(f"Error parsing WebSocket message: {e}"))
            self.on_error(f"Error parsing message: {str(e)}")
        except Exception as e:
            nuke.executeInMainThread(lambda e=e: nuke.tprint(f"Error handling WebSocket message: {e}"))
            self.on_error(f"Error handling message: {str(e)}")

    def start_polling(self):
        """Start HTTP polling monitor if not already running"""
        if not self.polling_monitor and not self.files_ready:
            nuke.executeInMainThread(lambda: nuke.tprint("Starting HTTP polling monitor..."))
            self.polling_monitor = start_polling_monitor(
                self.task_id,
                on_progress=self.on_progress,
                on_error=self.on_error,
                on_complete=self.on_complete
            )

    async def stop(self):
        """Stop the WebSocket connection and polling monitor"""
        self.running = False
        if self.websocket:
            try:
                nuke.executeInMainThread(lambda: nuke.tprint("Closing WebSocket connection..."))
                await self.websocket.close()
                nuke.executeInMainThread(lambda: nuke.tprint("WebSocket connection closed successfully"))
            except Exception as e:
                nuke.executeInMainThread(lambda e=e: nuke.tprint(f"Error closing WebSocket: {e}"))
            finally:
                self.websocket = None
        
        # Stop polling monitor if it exists
        if self.polling_monitor:
            nuke.executeInMainThread(lambda: nuke.tprint("Stopping HTTP polling monitor..."))
            self.polling_monitor.stop()
            self.polling_monitor = None
            nuke.executeInMainThread(lambda: nuke.tprint("HTTP polling monitor stopped"))

class ThreadedTask:
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
        uri = f"ws://{API_BASE_URL.replace('http://', '')}/api/{API_VERSION}/ws/{task_id}"
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
                                    nuke.executeInMainThread(lambda: self.on_complete(None))
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
                                nuke.executeInMainThread(lambda: self.on_complete(None))
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
    _image_cache = {}  # Cache for loaded images
    _cache_size = 5  # Number of frames to keep in cache
    _last_loaded_frame = None
    _last_loaded_image = None
    _image_dimensions = None  # Store image dimensions
    _zoom_scale = 1.0  # Current zoom scale
    _zoom_center = None  # Center point for zooming
    _offset = np.array([0, 0], dtype=np.float32)  # Pan offset


    @classmethod
    def get_image_dimensions(cls, frame_path):
        """Get image dimensions and log them"""
        img = cv2.imread(frame_path)
        if img is not None:
            height, width = img.shape[:2]
            nuke.tprint(f"Image dimensions for {frame_path}: {width}x{height}")
            return height, width
        nuke.tprint(f"Failed to get dimensions for {frame_path}")
        return None

    @classmethod
    def _load_image(cls, frame_path):
        """Load image with caching and dimension tracking"""
        # If this frame is already in cache, return it
        if frame_path in cls._image_cache:
            return cls._image_cache[frame_path]

        # Load the image
        img = cv2.imread(frame_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        if img is None:
            nuke.tprint(f"Failed to load image: {frame_path}")
            return None

        # Store dimensions if not already set
        if cls._image_dimensions is None:
            cls._image_dimensions = img.shape[:2]
            nuke.tprint(f"Stored image dimensions: {cls._image_dimensions}")

        # Verify dimensions match
        if img.shape[:2] != cls._image_dimensions:
            nuke.tprint(f"Warning: Image dimensions mismatch. Expected {cls._image_dimensions}, got {img.shape[:2]}")

        # Cache management - remove oldest entry if cache is full
        if len(cls._image_cache) >= cls._cache_size:
            # Remove the oldest entry
            oldest_key = next(iter(cls._image_cache))
            del cls._image_cache[oldest_key]

        # Add to cache
        cls._image_cache[frame_path] = img
        return img

    @classmethod
    def _preload_next_frame(cls, current_frame):
        """Preload the next frame in a background thread"""
        if current_frame < cls.max_frame:
            next_frame = current_frame + 1
            next_frame_path = cls.get_frame_path(next_frame)

            # Start preloading in a background thread
            def preload():
                cls._load_image(next_frame_path)
            
            threading.Thread(target=preload, daemon=True).start()

    @classmethod
    def get_frame_path(cls, frame_num):
        """Get the path for a specific frame number"""
        file_path = InputInfos.path
        input_file_name = str(os.path.splitext(os.path.basename(file_path))[0])
        if "%06d" in input_file_name:
            return file_path.replace('%06d', str(f"{frame_num:06}"))
        if "%05d" in input_file_name:
            return file_path.replace('%05d', str(f"{frame_num:05}"))
        if "%04d" in input_file_name:
            return file_path.replace('%04d', str(f"{frame_num:04}"))
        if "%03d" in input_file_name:
            return file_path.replace('%03d', str(f"{frame_num:03}"))
        if "%02d" in input_file_name:
            return file_path.replace('%02d', str(f"{frame_num:02}"))
        return file_path

    @classmethod
    def get_cover_scale_and_offset(cls, img_w, img_h, win_w=1280, win_h=720):
        """Calculate scale and offset for image display"""
        scale_cover = max(win_w / img_w, win_h / img_h) * cls._zoom_scale
        new_w = int(img_w * scale_cover)
        new_h = int(img_h * scale_cover)
        sx = int((win_w - new_w) / 2 + cls._offset[0])
        sy = int((win_h - new_h) / 2 + cls._offset[1])
        return scale_cover, sx, sy

    @classmethod
    def _zoom_at_point(cls, img, scale, center_point):
        """Zoom image at specific point"""
        if scale == 1.0 or img is None:
            return img.copy() if img is not None else None
            
        height, width = img.shape[:2]
        
        # Calculate new dimensions
        new_width = int(width * scale)
        new_height = int(height * scale)
        
        # Calculate the region to crop
        x1 = max(0, int(center_point[0] - (new_width / 2)))
        y1 = max(0, int(center_point[1] - (new_height / 2)))
        x2 = min(width, x1 + new_width)
        y2 = min(height, y1 + new_height)
        
        # Crop and resize
        cropped = img[y1:y2, x1:x2]
        zoomed = cv2.resize(cropped, (width, height))
        return zoomed

    @classmethod
    def getBbox(cls):
        file_path = InputInfos.path
        input_file_name = str(os.path.splitext(os.path.basename(file_path))[0])
        
        # Initialize frame range
        cls.min_frame = int(nuke.thisNode().knob('FrameRangeMin').value())
        cls.max_frame = int(nuke.thisNode().knob('FrameRangeMax').value())
        cls.current_frame = cls.min_frame
        cls.current_object_id = 0  # Initialize current object ID
        cls._zoom_scale = 1.0  # Reset zoom scale
        cls._zoom_center = None  # Reset zoom center
        cls._offset = np.array([0, 0], dtype=np.float32)  # Reset pan offset

        cls.input_path = cls.get_frame_path(cls.current_frame)
        
        window_name = "Selection - Left/Right arrows to change frame, Left click and drag for box, 'p' for positive point, 'n' for negative point, 'z' to undo, 'r' to reset, 'q' to finish, '1-9' to select object ID"
        cv2.namedWindow(window_name, 0)
        cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)
        cv2.resizeWindow(window_name, 1280, 720)
        
        def mouse_callback(event, x, y, flags, param):
            if event == cv2.EVENT_MOUSEWHEEL:
                # Handle zoom
                if cls._last_loaded_image is not None:
                    # Calculate zoom factor (positive for zoom in, negative for zoom out)
                    zoom_factor = 1.1 if flags > 0 else 0.9
                    
                    # Update zoom scale
                    cls._zoom_scale *= zoom_factor
                    cls._zoom_scale = max(0.1, min(5.0, cls._zoom_scale))  # Limit zoom range
                    
                    # Update zoom center
                    cls._zoom_center = (x, y)
                    
                    # Get zoomed image
                    zoomed_img = cls._zoom_at_point(cls._last_loaded_image, cls._zoom_scale, cls._zoom_center)
                    if zoomed_img is not None:
                        # Draw current state
                        if cls.current_box is not None:
                            cv2.rectangle(zoomed_img, cls.current_box[0], cls.current_box[1], (0, 255, 0), 2)
                        cv2.imshow(window_name, zoomed_img)
                return

            # Original mouse callback functionality
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
                            prompt.input_path = cls.get_frame_path(cls.current_frame)  # Store input path
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
            # Load current frame with caching
            current_img_path = cls.get_frame_path(cls.current_frame)
            img = cls._load_image(current_img_path)
            
            if img is None:
                nuke.tprint(f"Error loading image for frame {cls.current_frame}")
                break
                
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
                    # Preload next frame
                    cls._preload_next_frame(cls.current_frame)
            
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
        """Format prompts for API with dimension information"""
        if cls._image_dimensions is None:
            # Get dimensions from first frame if not set
            first_frame_path = cls.input_path
            cls._image_dimensions = cls.get_image_dimensions(first_frame_path)
            if cls._image_dimensions is None:
                raise ValueError("Could not determine image dimensions")

        # Format prompts for API
        formatted_prompts = []
        for prompt in cls.prompts:
            prompt_dict = {
                "frame_index": prompt.frame_index,
                "object_id": prompt.object_id,
                "bboxes": prompt.bboxes,
                "points_positive": prompt.points_positive,
                "points_negative": prompt.points_negative
            }
            formatted_prompts.append(prompt_dict)

        return {
            "prompts": formatted_prompts,
            "dimensions": {
                "height": cls._image_dimensions[0],
                "width": cls._image_dimensions[1]
            }
        }

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
        response = requests.get(f"{get_api_base_url()}/api/{API_VERSION}/health")
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
            return f"/mnt/PTGData/shared/{'/'.join(parts[2:])}"
    
    # Handle local Windows paths (e.g., D:/path)
    if ':' in path:
        # Convert drive letter path to network share path
        drive, rest = path.split(':', 1)
        # Remove leading slash if present
        rest = rest.lstrip('/')
        # Convert to network share format
        return f"/mnt/PTGData/shared/{rest}"
    
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

def create_read_node(video_output_path, frame_range, is_zip=False, file_path=None):
    """
    Create a read node for the processed mask sequence.
    
    Args:
        video_output_path (str): Path to the output file or directory
        frame_range (list): List containing [start_frame, end_frame]
        is_zip (bool): Whether the output is a zip file
        file_path (str): Original input file path to extract prefix
    
    Returns:
        nuke.Node: The created read node
    """
    try:
        if is_zip:
            normalized_path = normalize_path(video_output_path)
            read_node = nuke.nodes.Read(file=normalized_path, first=frame_range[0], last=frame_range[1]-1)
        else:
            # Verify output dimensions match input
            if BoundingBox._image_dimensions is not None:
                # Get first output frame
                extract_dir = os.path.dirname(video_output_path)
                first_output_frame = os.path.join(extract_dir, f"mask_{frame_range[0]:04d}.exr")
                if os.path.exists(first_output_frame):
                    output_img = cv2.imread(first_output_frame)
                    if output_img is not None:
                        output_dims = output_img.shape[:2]
                        if output_dims != BoundingBox._image_dimensions:
                            nuke.message(f"Warning: Output dimensions ({output_dims}) don't match input dimensions ({BoundingBox._image_dimensions})")
                            nuke.tprint(f"Dimension mismatch - Input: {BoundingBox._image_dimensions}, Output: {output_dims}")
            
            # Extract prefix from input file path
            if file_path:
                # Get the filename without extension
                base_name = os.path.splitext(os.path.basename(file_path))[0]
                # Construct mask pattern using the prefix
                extract_dir = os.path.dirname(video_output_path)
                
                # Try to find a frame padding pattern in the input file path
                match = re.search(r'(#+|%0?(\d+)d)', file_path)
                if match:
                    if '#' in match.group(1):
                        num_digits = len(match.group(1))
                    else:
                        num_digits = int(match.group(2))
                else:
                    num_digits = 4  # Default to 4 if not found

                prefix = base_name.rsplit('_', 1)[0]
                mask_pattern = os.path.join(extract_dir, f"{prefix}_mask_%0{num_digits}d.exr")
            else:
                # Fallback to default pattern if no file_path provided
                extract_dir = os.path.dirname(video_output_path)
                mask_pattern = os.path.join(extract_dir, "mask_%04d.exr")
            
            # Normalize the path
            normalized_pattern = normalize_path(mask_pattern)
            read_node = nuke.nodes.Read(file=normalized_pattern, first=frame_range[0], last=frame_range[1]-1)
            
            # Update status to complete
            nuke.executeInMainThread(lambda: update_status_safely("Processing complete"))
            nuke.executeInMainThread(lambda: reset_ui_state())
            
            nuke.message("Mask generation completed! Files saved and loaded into Nuke.")
        
        return read_node
    except Exception as e:
        error_msg = f"Error creating read node: {str(e)}"
        nuke.tprint(error_msg)
        nuke.message(error_msg)
        raise

def GenerateMask():
    # Upgrade the node if needed (for backwards compatibility)
    upgrade_node_if_needed()
    
    # Set API URL from node configuration
    set_api_base_url()
    
    Output_path = nuke.thisNode().knob('OutputPath').getValue()
    File_path = nuke.thisNode().knob('FilePath').value()

    # Checks
    if str(os.path.splitext(os.path.basename(Output_path))[0]) == '':
        nuke.message("You must assign a file name")
        return
        
    if nuke.thisNode()['FilePath'].value().lower().endswith("mp4"):
        nuke.message('Unsupported input format. Input must be an Image Sequence')
        return
        
    if nuke.thisNode().knob('FileType').value() == "exr":
        # Check for valid frame padding patterns (3 to 10 digits)
        valid_patterns = ['%03d', '%04d', '%05d', '%06d', '%07d', '%08d', '%09d', '%10d']
        has_valid_pattern = any(pattern in Output_path for pattern in valid_patterns)
        if not has_valid_pattern:
            nuke.message("Your file must contain a frame padding pattern (### to ##########)")
            return

    if not BoundingBox.prompts:
        nuke.message("No prompts added. Please add at least one bounding box or points.")
        return

    # Check if API server is running
    if not check_api_server():
        nuke.message("Error: API server is not running. Please ensure the server is running at " + get_api_base_url())
        return

    # Reset progress bar and status message
    update_status_safely("Starting processing...")
    
    # Get all required parameters
    video_path = convert_windows_to_linux_path(nuke.thisNode().knob('FilePath').value())
    video_output_path = nuke.thisNode().knob('OutputPath').getValue()
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
            f"{get_api_base_url()}/api/{API_VERSION}/models/load",
            json={"model_type": model_type},
            timeout=300  # 5 minutes timeout
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
        
        # Get dimensions from BoundingBox class
        if BoundingBox._image_dimensions is None:
            # Get dimensions from first frame if not set
            first_frame_path = get_frame_path(frame_range[0])
            BoundingBox._image_dimensions = BoundingBox.get_image_dimensions(first_frame_path)
            if BoundingBox._image_dimensions is None:
                raise ValueError("Could not determine image dimensions")
        
        request_data = {
            "sequence_path": video_path,
            "frame_range": frame_range,
            "prompts": validated_prompts,
            "bits": bits,
            "original_fps": original_fps,
            "target_fps": target_fps,
            "dimensions": {
                "height": BoundingBox._image_dimensions[0],
                "width": BoundingBox._image_dimensions[1]
            }
        }

        nuke.tprint("Sending sequence to API for processing...")
        nuke.tprint(f"Using paths - Input: {video_path}")
        nuke.tprint(f"Image dimensions: {BoundingBox._image_dimensions}")
        nuke.tprint(f"Request data: {json.dumps(request_data, indent=2)}")  # Debug print
        
        # Send request to process_sequence endpoint
        response = requests.post(
            f"{get_api_base_url()}/api/{API_VERSION}/process_sequence?as_file=true",
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
            output_endpoint = f"{get_api_base_url()}/api/{API_VERSION}/output/{task_id}"
            max_attempts = 60  # 5 minutes at 5-second intervals
            attempt = 0
            error_count = 0
            max_error_count = 10
            while attempt < max_attempts and process_task.running:
                attempt += 1
                try:
                    output_response = requests.get(output_endpoint, timeout=5)
                    if output_response.status_code == 200:
                        # Download the ZIP file
                        download_url = f"{get_api_base_url()}/api/{API_VERSION}/download_all/{task_id}"
                        download_response = requests.get(download_url, stream=True)
                        download_response.raise_for_status()
                        
                        # Get file size for progress tracking
                        file_size = int(download_response.headers.get('content-length', 0))
                        total_downloaded = 0
                        
                        # Save the ZIP file
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
                        
                        # Create read node in main thread
                        try:
                            nuke.executeInMainThread(lambda: create_read_node(video_output_path, frame_range, video_output_path.endswith('.zip'), video_path))
                        except Exception as e:
                            error_msg = f"Error executing in main thread: {str(e)}"
                            nuke.tprint(error_msg)
                            nuke.message(error_msg)
                            process_task.report_error(error_msg)
                        return
                    else:
                        # No ZIP file yet, keep waiting
                        time.sleep(5)
                except Exception as e:
                    error_count += 1
                    if error_count >= max_error_count:
                        error_msg = f"Error downloading result: {str(e)}"
                        nuke.tprint(error_msg)
                        nuke.executeInMainThread(lambda: nuke.message(error_msg))
                        process_task.report_error(error_msg)
                        return
                    time.sleep(5)
    
    # Function to handle processing errors
    def on_process_error(error):
        nuke.message(f"Error during processing: {error}")
    
    # Create the threaded task for sequence processing
    process_task = ThreadedTask(
        on_complete=lambda: None,
        on_progress=lambda p, m: update_status_safely(f"Progress: {int(p)}% - {m}"),
        on_error=on_process_error
    )
    
    # Create the threaded task for model loading
    model_task = ThreadedTask(
        on_complete=on_model_loaded,
        on_progress=lambda p, m: update_status_safely(f"Loading model: {int(p)}% - {m}"),
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
        response = requests.post(
            f"{get_api_base_url()}/api/{API_VERSION}/reset",
            timeout=60  # 1 minute timeout
        )
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
    
    # Server Management
    s.addKnob(nuke.Text_Knob('ServerStatus', 'Server Status'))
    s.knob('ServerStatus').setEnabled(False)
    s.knob('ServerStatus').setValue("Checking...")
    
    s.addKnob(nuke.PyScript_Knob('StartServer', 'Start Server', 'start_docker_server()'))
    s.addKnob(nuke.PyScript_Knob('StopServer', 'Stop Server', 'stop_docker_server()'))
    s.knob('StartServer').setFlag(nuke.STARTLINE)
    
    # API Configuration
    s.addKnob(nuke.String_Knob('APIBaseURL', 'API Base URL'))
    s.knob('APIBaseURL').setValue(API_BASE_URL)
    s.knob('APIBaseURL').setTooltip("Base URL for the SAM2 API server (e.g., http://localhost:8000)")
    s.addKnob(nuke.PyScript_Knob('SetAPIURL', 'Set', 'set_api_base_url()'))
    s.knob('SetAPIURL').setFlag(nuke.STARTLINE)
    
    s.addKnob(nuke.File_Knob('FilePath', 'File Path'))
    s.addKnob(nuke.PyScript_Knob('UpdatePath', 'Update Path', 'UpdatePath()'))
    
    # Frame selection
    s.addKnob(nuke.Int_Knob("FrameRangeMin", 'Frame Range'))
    s.addKnob(nuke.Int_Knob("FrameRangeMax", ' '))
    s.addKnob(nuke.Int_Knob("FPS", 'Output Frame Rate'))
    
    # Add auto-update frame range knob
    s.addKnob(nuke.Boolean_Knob('AutoUpdateFrameRange', 'Auto Update Frame Range'))
    s.knob('AutoUpdateFrameRange').setValue(True)
    s.knob('AutoUpdateFrameRange').setTooltip("Automatically update frame range based on input node")
    
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
    s.addKnob(nuke.Enumeration_Knob('FileType', 'File type', ['exr']))
    s.addKnob(nuke.File_Knob('OutputPath', 'Output Path'))
    s.knob('OutputPath').setTooltip("path/to/your/file_####.exr (supports ### to ########## for frame padding)")
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
    s['OutputPath'].setTooltip("path/to/your/file_####.exr (supports ### to ########## for frame padding)")
    s['GenerateMask'].setTooltip("Generate Mask")
    s['StatusMessage'].setTooltip("Current processing status")
    s['AutoUpdateFrameRange'].setTooltip("Automatically update frame range based on input node")

    # Add callback for auto-updating frame range
    s.addKnob(nuke.PyScript_Knob('updateFrameRange', 'Update Frame Range', 'update_frame_range()'))
    s.knob('updateFrameRange').setVisible(False)
    
    # Add callback for input changes
    def on_knob_changed():
        if nuke.thisKnob().name() == 'inputChange':
            update_frame_range()
    
    s.addKnob(nuke.PyScript_Knob('onKnobChanged', 'On Knob Changed', 'on_knob_changed()'))
    s.knob('onKnobChanged').setVisible(False)
    
    # Register the callback
    nuke.addKnobChanged(on_knob_changed, node=s)

def update_frame_range():
    """Update frame range based on input node"""
    try:
        node = nuke.thisNode()
        if not node:
            return
            
        # Check if the node has the required knobs
        if not node.knob('AutoUpdateFrameRange'):
            return
            
        if not node.knob('AutoUpdateFrameRange').value():
            return
            
        input_node = node.input(0)
        if input_node and input_node.Class() == 'Read':
            node.knob('FrameRangeMin').setValue(int(input_node.knob('first').value()))
            node.knob('FrameRangeMax').setValue(int(input_node.knob('last').value()))
            node.knob('FilePath').setValue(input_node.knob('file').value())
    except Exception as e:
        nuke.tprint(f"Error updating frame range: {str(e)}")

def on_input_changed():
    """Callback when input node changes"""
    try:
        update_frame_range()
    except Exception as e:
        nuke.tprint(f"Error in input changed callback: {str(e)}")

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

async def connect_with_retry(uri, max_retries=5, initial_delay=1):
    retry_count = 0
    delay = initial_delay

    while retry_count < max_retries:
        try:
            nuke.executeInMainThread(lambda: nuke.tprint(f"Attempting WebSocket connection (attempt {retry_count + 1}/{max_retries})..."))
            async with websockets.connect(uri, ping_timeout=20, close_timeout=10) as websocket:
                nuke.executeInMainThread(lambda: nuke.tprint("WebSocket connection established successfully"))
                return websocket
        except Exception as e:
            retry_count += 1
            nuke.executeInMainThread(lambda e=e, retry_count=retry_count, max_retries=max_retries: 
                nuke.tprint(f"WebSocket connection failed (attempt {retry_count}/{max_retries}): {e}"))
            if retry_count < max_retries:
                await asyncio.sleep(delay)
                delay *= 2  # Exponential backoff
            else:
                nuke.executeInMainThread(lambda: nuke.tprint("Max retries reached, falling back to HTTP polling"))
                return None

async def monitor_progress(uri):
    """Monitor progress using WebSocket with HTTP polling fallback"""
    websocket = await connect_with_retry(uri)
    if not websocket:
        # Start HTTP polling monitor
        task_id = uri.split('/')[-1]  # Extract task_id from WebSocket URI
        return start_polling_monitor(
            task_id,
            on_progress=lambda p, m: nuke.executeInMainThread(lambda: nuke.tprint(f"Progress: {p}% - {m}")),
            on_error=lambda e: nuke.executeInMainThread(lambda: nuke.tprint(f"Error: {e}")),
            on_complete=lambda: nuke.executeInMainThread(lambda: nuke.tprint("Processing completed successfully"))
        )

    try:
        while True:
            try:
                data = await asyncio.wait_for(websocket.recv(), timeout=30)
                progress_data = json.loads(data)
                progress = int(progress_data.get('progress', 0))
                message = progress_data.get('message', '')
                status = progress_data.get('status', 'running')
                
                nuke.executeInMainThread(lambda p=progress, m=message: nuke.tprint(f"Progress: {p}% - {m}"))
                
                if status in ['completed', 'failed']:
                    if status == 'completed':
                        nuke.executeInMainThread(lambda: nuke.tprint("Processing completed successfully"))
                    else:
                        error = progress_data.get('error', 'Unknown error')
                        nuke.executeInMainThread(lambda e=error: nuke.tprint(f"Task failed: {e}"))
                    break
                    
            except asyncio.TimeoutError:
                try:
                    pong_event = await websocket.ping()
                    await asyncio.wait_for(pong_event, timeout=5)
                    nuke.executeInMainThread(lambda: nuke.tprint("WebSocket ping successful, connection still active"))
                except (asyncio.TimeoutError, Exception) as e:
                    nuke.executeInMainThread(lambda e=e: nuke.tprint(f"WebSocket connection lost: {e}"))
                    # Extract task_id and start HTTP polling
                    task_id = uri.split('/')[-1]
                    return start_polling_monitor(
                        task_id,
                        on_progress=lambda p, m: nuke.executeInMainThread(lambda: nuke.tprint(f"Progress: {p}% - {m}")),
                        on_error=lambda e: nuke.executeInMainThread(lambda: nuke.tprint(f"Error: {e}")),
                        on_complete=lambda: nuke.executeInMainThread(lambda: nuke.tprint("Processing completed successfully"))
                    )
                    
    except Exception as e:
        nuke.executeInMainThread(lambda e=e: nuke.tprint(f"WebSocket error: {e}"))
        # Extract task_id and start HTTP polling
        task_id = uri.split('/')[-1]
        return start_polling_monitor(
            task_id,
            on_progress=lambda p, m: nuke.executeInMainThread(lambda: nuke.tprint(f"Progress: {p}% - {m}")),
            on_error=lambda e: nuke.executeInMainThread(lambda: nuke.tprint(f"Error: {e}")),
            on_complete=lambda: nuke.executeInMainThread(lambda: nuke.tprint("Processing completed successfully"))
        )
    finally:
        await websocket.close()

# Register functions for Nuke PyScript_Knobs
import __main__
__main__.reset_model_state = reset_model_state
__main__.CreateSAM2Node = CreateSAM2Node
__main__.set_api_base_url = set_api_base_url
__main__.start_docker_server = start_docker_server
__main__.stop_docker_server = stop_docker_server
# (Add any other functions you want to call from knobs) 