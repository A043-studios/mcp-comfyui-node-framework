#!/usr/bin/env python3
"""
Execution Monitor for ComfyUI Framework MCP Server
Provides real-time monitoring and progress tracking for long-running operations
"""

import asyncio
import json
import logging
import time
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, asdict
from enum import Enum

logger = logging.getLogger(__name__)

class ExecutionStatus(Enum):
    """Execution status enumeration"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"

@dataclass
class ExecutionMetrics:
    """Metrics for execution tracking"""
    start_time: datetime
    end_time: Optional[datetime] = None
    duration_seconds: Optional[float] = None
    progress_percentage: float = 0.0
    current_step: str = ""
    total_steps: int = 0
    completed_steps: int = 0
    memory_usage_mb: float = 0.0
    cpu_usage_percent: float = 0.0
    
    def update_progress(self, completed: int, total: int, step_name: str = ""):
        """Update progress metrics"""
        self.completed_steps = completed
        self.total_steps = total
        self.current_step = step_name
        if total > 0:
            self.progress_percentage = (completed / total) * 100.0
    
    def complete(self):
        """Mark execution as completed"""
        self.end_time = datetime.now()
        self.duration_seconds = (self.end_time - self.start_time).total_seconds()
        self.progress_percentage = 100.0

@dataclass
class ExecutionInfo:
    """Complete execution information"""
    execution_id: str
    tool_name: str
    arguments: Dict[str, Any]
    status: ExecutionStatus
    metrics: ExecutionMetrics
    output_directory: Optional[str] = None
    error_message: Optional[str] = None
    artifacts: Dict[str, Any] = None
    logs: List[str] = None
    
    def __post_init__(self):
        if self.artifacts is None:
            self.artifacts = {}
        if self.logs is None:
            self.logs = []
    
    def add_log(self, message: str, level: str = "INFO"):
        """Add a log message"""
        timestamp = datetime.now().isoformat()
        log_entry = f"[{timestamp}] {level}: {message}"
        self.logs.append(log_entry)
        logger.log(getattr(logging, level.upper(), logging.INFO), message)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        result = asdict(self)
        result["status"] = self.status.value
        result["metrics"]["start_time"] = self.metrics.start_time.isoformat()
        if self.metrics.end_time:
            result["metrics"]["end_time"] = self.metrics.end_time.isoformat()
        return result

class ExecutionMonitor:
    """Monitors and tracks execution progress"""
    
    def __init__(self, max_executions: int = 100, cleanup_interval: int = 3600):
        self.executions: Dict[str, ExecutionInfo] = {}
        self.max_executions = max_executions
        self.cleanup_interval = cleanup_interval
        self._lock = threading.RLock()
        self._cleanup_task: Optional[asyncio.Task] = None
        self._progress_callbacks: Dict[str, List[Callable]] = {}
        
        # Start cleanup task
        self._start_cleanup_task()
    
    def _start_cleanup_task(self):
        """Start the cleanup task for old executions"""
        async def cleanup_loop():
            while True:
                try:
                    await asyncio.sleep(self.cleanup_interval)
                    self._cleanup_old_executions()
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error(f"Error in cleanup task: {str(e)}")
        
        try:
            loop = asyncio.get_event_loop()
            self._cleanup_task = loop.create_task(cleanup_loop())
        except RuntimeError:
            # No event loop running, cleanup will be manual
            pass
    
    def _cleanup_old_executions(self):
        """Clean up old completed executions"""
        with self._lock:
            if len(self.executions) <= self.max_executions:
                return
            
            # Sort by start time and keep only the most recent
            sorted_executions = sorted(
                self.executions.items(),
                key=lambda x: x[1].metrics.start_time,
                reverse=True
            )
            
            # Keep only the most recent executions
            to_keep = dict(sorted_executions[:self.max_executions])
            removed_count = len(self.executions) - len(to_keep)
            
            self.executions = to_keep
            
            if removed_count > 0:
                logger.info(f"Cleaned up {removed_count} old executions")
    
    def start_execution(self, execution_id: str, tool_name: str, arguments: Dict[str, Any], 
                       output_directory: Optional[str] = None) -> ExecutionInfo:
        """Start tracking a new execution"""
        with self._lock:
            if execution_id in self.executions:
                raise ValueError(f"Execution {execution_id} already exists")
            
            metrics = ExecutionMetrics(start_time=datetime.now())
            execution = ExecutionInfo(
                execution_id=execution_id,
                tool_name=tool_name,
                arguments=arguments,
                status=ExecutionStatus.PENDING,
                metrics=metrics,
                output_directory=output_directory
            )
            
            self.executions[execution_id] = execution
            execution.add_log(f"Started execution for tool: {tool_name}")
            
            logger.info(f"Started tracking execution {execution_id}")
            return execution
    
    def update_status(self, execution_id: str, status: ExecutionStatus, error_message: Optional[str] = None):
        """Update execution status"""
        with self._lock:
            if execution_id not in self.executions:
                logger.warning(f"Execution {execution_id} not found for status update")
                return
            
            execution = self.executions[execution_id]
            old_status = execution.status
            execution.status = status
            
            if error_message:
                execution.error_message = error_message
                execution.add_log(f"Error: {error_message}", "ERROR")
            
            if status in [ExecutionStatus.COMPLETED, ExecutionStatus.FAILED, ExecutionStatus.CANCELLED, ExecutionStatus.TIMEOUT]:
                execution.metrics.complete()
            
            execution.add_log(f"Status changed from {old_status.value} to {status.value}")
            
            # Trigger progress callbacks
            self._trigger_callbacks(execution_id, execution)
    
    def update_progress(self, execution_id: str, completed: int, total: int, step_name: str = ""):
        """Update execution progress"""
        with self._lock:
            if execution_id not in self.executions:
                logger.warning(f"Execution {execution_id} not found for progress update")
                return
            
            execution = self.executions[execution_id]
            execution.metrics.update_progress(completed, total, step_name)
            
            if step_name:
                execution.add_log(f"Progress: {step_name} ({completed}/{total})")
            
            # Trigger progress callbacks
            self._trigger_callbacks(execution_id, execution)
    
    def add_artifact(self, execution_id: str, artifact_name: str, artifact_data: Any):
        """Add an artifact to the execution"""
        with self._lock:
            if execution_id not in self.executions:
                logger.warning(f"Execution {execution_id} not found for artifact addition")
                return
            
            execution = self.executions[execution_id]
            execution.artifacts[artifact_name] = artifact_data
            execution.add_log(f"Added artifact: {artifact_name}")
    
    def get_execution(self, execution_id: str) -> Optional[ExecutionInfo]:
        """Get execution information"""
        with self._lock:
            return self.executions.get(execution_id)
    
    def get_all_executions(self) -> Dict[str, ExecutionInfo]:
        """Get all executions"""
        with self._lock:
            return self.executions.copy()
    
    def get_active_executions(self) -> Dict[str, ExecutionInfo]:
        """Get currently active executions"""
        with self._lock:
            return {
                eid: execution for eid, execution in self.executions.items()
                if execution.status in [ExecutionStatus.PENDING, ExecutionStatus.RUNNING]
            }
    
    def get_execution_summary(self) -> Dict[str, Any]:
        """Get summary of all executions"""
        with self._lock:
            total = len(self.executions)
            status_counts = {}
            
            for execution in self.executions.values():
                status = execution.status.value
                status_counts[status] = status_counts.get(status, 0) + 1
            
            # Calculate average duration for completed executions
            completed_durations = [
                execution.metrics.duration_seconds
                for execution in self.executions.values()
                if execution.status == ExecutionStatus.COMPLETED and execution.metrics.duration_seconds
            ]
            
            avg_duration = sum(completed_durations) / len(completed_durations) if completed_durations else 0
            
            return {
                "total_executions": total,
                "status_counts": status_counts,
                "active_executions": len(self.get_active_executions()),
                "average_duration_seconds": round(avg_duration, 2),
                "oldest_execution": min(
                    (e.metrics.start_time for e in self.executions.values()),
                    default=None
                ),
                "newest_execution": max(
                    (e.metrics.start_time for e in self.executions.values()),
                    default=None
                )
            }
    
    def register_progress_callback(self, execution_id: str, callback: Callable[[ExecutionInfo], None]):
        """Register a callback for progress updates"""
        with self._lock:
            if execution_id not in self._progress_callbacks:
                self._progress_callbacks[execution_id] = []
            self._progress_callbacks[execution_id].append(callback)
    
    def _trigger_callbacks(self, execution_id: str, execution: ExecutionInfo):
        """Trigger progress callbacks for an execution"""
        callbacks = self._progress_callbacks.get(execution_id, [])
        for callback in callbacks:
            try:
                callback(execution)
            except Exception as e:
                logger.error(f"Error in progress callback: {str(e)}")
    
    def save_execution_log(self, execution_id: str, file_path: Optional[str] = None) -> str:
        """Save execution log to file"""
        with self._lock:
            if execution_id not in self.executions:
                raise ValueError(f"Execution {execution_id} not found")
            
            execution = self.executions[execution_id]
            
            if not file_path:
                output_dir = execution.output_directory or "./logs"
                Path(output_dir).mkdir(parents=True, exist_ok=True)
                file_path = f"{output_dir}/execution_{execution_id}.json"
            
            with open(file_path, 'w') as f:
                json.dump(execution.to_dict(), f, indent=2, default=str)
            
            logger.info(f"Execution log saved to: {file_path}")
            return file_path
    
    def cancel_execution(self, execution_id: str) -> bool:
        """Cancel an execution"""
        with self._lock:
            if execution_id not in self.executions:
                return False
            
            execution = self.executions[execution_id]
            if execution.status in [ExecutionStatus.PENDING, ExecutionStatus.RUNNING]:
                self.update_status(execution_id, ExecutionStatus.CANCELLED)
                return True
            
            return False
    
    def cleanup(self):
        """Clean up the monitor"""
        if self._cleanup_task:
            self._cleanup_task.cancel()
        
        # Save all active executions
        active = self.get_active_executions()
        for execution_id in active:
            try:
                self.save_execution_log(execution_id)
            except Exception as e:
                logger.error(f"Failed to save execution log for {execution_id}: {str(e)}")

# Global execution monitor instance
execution_monitor: Optional[ExecutionMonitor] = None

def get_execution_monitor() -> ExecutionMonitor:
    """Get or create the global execution monitor instance"""
    global execution_monitor
    
    if execution_monitor is None:
        execution_monitor = ExecutionMonitor()
    
    return execution_monitor

def initialize_execution_monitor(max_executions: int = 100, cleanup_interval: int = 3600) -> ExecutionMonitor:
    """Initialize the execution monitor"""
    global execution_monitor
    execution_monitor = ExecutionMonitor(max_executions, cleanup_interval)
    return execution_monitor
