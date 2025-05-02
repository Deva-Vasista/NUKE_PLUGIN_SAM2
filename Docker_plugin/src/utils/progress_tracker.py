from fastapi import WebSocket
import asyncio
from typing import Dict, Optional
from datetime import datetime
import json

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

class ProgressTracker:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ProgressTracker, cls).__new__(cls)
            cls._instance.tasks: Dict[str, ProcessingStatus] = {}
            cls._instance.connections: Dict[str, set[WebSocket]] = {}
        return cls._instance

    async def create_task(self, task_id: str) -> ProcessingStatus:
        """Create a new task for tracking."""
        status = ProcessingStatus(task_id)
        self.tasks[task_id] = status
        self.connections[task_id] = set()
        return status

    async def update_progress(self, task_id: str, progress: float, message: str = ""):
        """Update task progress and notify all connected clients."""
        if task_id not in self.tasks:
            return
        
        status = self.tasks[task_id]
        status.progress = progress
        status.message = message
        status.status = "running"

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
        else:
            status.status = "failed"
            status.error = error

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
        """Get the current status of a task."""
        return self.tasks.get(task_id)

    def cleanup_old_tasks(self, max_age_hours: int = 24):
        """Clean up completed tasks older than max_age_hours."""
        now = datetime.now()
        to_remove = []
        for task_id, status in self.tasks.items():
            if status.end_time and (now - status.end_time).total_seconds() > max_age_hours * 3600:
                to_remove.append(task_id)
        
        for task_id in to_remove:
            del self.tasks[task_id]
            if task_id in self.connections:
                del self.connections[task_id] 