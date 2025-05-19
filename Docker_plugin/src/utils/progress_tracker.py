from fastapi import WebSocket
import asyncio
from typing import Dict, Optional
from datetime import datetime
import json
import os

class ProcessingStatus:
    def __init__(self, task_id: str):
        self.task_id = task_id
        self.progress = 0
        self.status = "initializing"  # initializing, running, completed, failed
        self.message = ""
        self.start_time = datetime.now()
        self.end_time: Optional[datetime] = None
        self.error: Optional[str] = None

    def to_dict(self):
        return {
            "task_id": self.task_id,
            "progress": self.progress,
            "status": self.status,
            "message": self.message,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "error": self.error
        }

    @classmethod
    def from_dict(cls, data):
        obj = cls(data["task_id"])
        obj.progress = data.get("progress", 0)
        obj.status = data.get("status", "initializing")
        obj.message = data.get("message", "")
        obj.start_time = datetime.fromisoformat(data["start_time"]) if data.get("start_time") else datetime.now()
        obj.end_time = datetime.fromisoformat(data["end_time"]) if data.get("end_time") else None
        obj.error = data.get("error")
        return obj

class ProgressTracker:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ProgressTracker, cls).__new__(cls)
            cls._instance.tasks: Dict[str, ProcessingStatus] = {}
            cls._instance.connections: Dict[str, set[WebSocket]] = {}
            cls._instance._load_all_statuses()
        return cls._instance

    def _status_file_path(self, task_id):
        return os.path.join("Output", task_id, "status.json")

    def _save_status(self, status: ProcessingStatus):
        os.makedirs(os.path.join("Output", status.task_id), exist_ok=True)
        with open(self._status_file_path(status.task_id), "w") as f:
            json.dump(status.to_dict(), f)

    def _load_status(self, task_id):
        try:
            with open(self._status_file_path(task_id), "r") as f:
                data = json.load(f)
                return ProcessingStatus.from_dict(data)
        except Exception:
            return None

    def _load_all_statuses(self):
        output_dir = "Output"
        if not os.path.exists(output_dir):
            return
        for task_id in os.listdir(output_dir):
            status_path = self._status_file_path(task_id)
            if os.path.exists(status_path):
                status = self._load_status(task_id)
                if status:
                    self.tasks[task_id] = status

    async def create_task(self, task_id: str) -> ProcessingStatus:
        """Create a new task for tracking."""
        status = ProcessingStatus(task_id)
        self.tasks[task_id] = status
        self.connections[task_id] = set()
        self._save_status(status)
        return status

    async def update_progress(self, task_id: str, progress: float, message: str = ""):
        """Update task progress and notify all connected clients."""
        if task_id not in self.tasks:
            return
        status = self.tasks[task_id]
        status.progress = progress
        status.message = message
        status.status = "running"
        self._save_status(status)
        # Notify all connected clients
        if task_id in self.connections:
            dead_connections = set()
            for websocket in self.connections[task_id]:
                try:
                    await websocket.send_json(status.to_dict())
                except:
                    dead_connections.add(websocket)
            # Clean up dead connections
            self.connections[task_id] -= dead_connections

    async def complete_task(self, task_id: str, success: bool = True, error: str = None):
        """Mark a task as completed or failed."""
        if task_id not in self.tasks:
            return
        status = self.tasks[task_id]
        status.end_time = datetime.now()
        if success:
            status.status = "completed"
            status.progress = 100
            status.message = "Processing completed successfully"
        else:
            status.status = "failed"
            status.error = error
            status.message = f"Processing failed: {error}"
        self._save_status(status)
        # Notify all connected clients
        if task_id in self.connections:
            for websocket in self.connections[task_id]:
                try:
                    await websocket.send_json(status.to_dict())
                except:
                    continue

    async def register_client(self, task_id: str, websocket: WebSocket):
        """Register a new WebSocket client for a task."""
        if task_id not in self.connections:
            self.connections[task_id] = set()
        self.connections[task_id].add(websocket)
        # Send initial status if task exists
        if task_id in self.tasks:
            await websocket.send_json(self.tasks[task_id].to_dict())

    async def unregister_client(self, task_id: str, websocket: WebSocket):
        """Unregister a WebSocket client."""
        if task_id in self.connections:
            self.connections[task_id].discard(websocket)

    def get_task_status(self, task_id: str) -> Optional[ProcessingStatus]:
        # Try in-memory first
        status = self.tasks.get(task_id)
        if status:
            return status
        # Try loading from disk if not found
        status = self._load_status(task_id)
        if status:
            self.tasks[task_id] = status
        return status

    def cleanup_old_tasks(self, max_age_hours: int = 24):
        now = datetime.now()
        to_remove = []
        for task_id, status in self.tasks.items():
            if status.end_time and (now - status.end_time).total_seconds() > max_age_hours * 3600:
                to_remove.append(task_id)
        for task_id in to_remove:
            del self.tasks[task_id]
            if task_id in self.connections:
                del self.connections[task_id] 
            # Optionally, remove status.json file
            try:
                os.remove(self._status_file_path(task_id))
            except:
                pass

    def clear_all(self):
        for task_id, connections in self.connections.items():
            for websocket in connections:
                try:
                    asyncio.create_task(websocket.close())
                except:
                    pass
        self.tasks.clear()
        self.connections.clear()
        # Optionally, clear all status.json files
        output_dir = "Output"
        if os.path.exists(output_dir):
            for task_id in os.listdir(output_dir):
                status_path = self._status_file_path(task_id)
                if os.path.exists(status_path):
                    try:
                        os.remove(status_path)
                    except:
                        pass 