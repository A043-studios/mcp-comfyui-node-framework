"""
Base Agent Class for MCP Multi-Agent Framework
Defines the common interface and functionality for all agents
"""

import time
import asyncio
import logging
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, Any

try:
    from ..utils import create_directory, save_json
except ImportError:
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    from utils import create_directory, save_json

try:
    from ..mcp_client import MCPManager
    MCP_CLIENT_AVAILABLE = True
except ImportError:
    MCP_CLIENT_AVAILABLE = False


class BaseAgent(ABC):
    """
    Abstract base class for all MCP framework agents
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the base agent
        
        Args:
            config: Agent configuration dictionary
        """
        self.config = config
        self.name = self.__class__.__name__
        self.start_time = None
        self.end_time = None

        # MCP integration
        self.mcp_manager = None
        self.mcp_enabled = config.get("mcp_enabled", False)
        if self.mcp_enabled and MCP_CLIENT_AVAILABLE:
            mcp_configs = config.get("mcp_servers", [])
            if mcp_configs:
                self.mcp_manager = MCPManager(mcp_configs)
        
        # Agent capabilities and tools
        self.tools = config.get("tools", [])
        self.model = config.get("model", "anthropic/claude-3.5-sonnet")
        self.temperature = config.get("temperature", 0.1)
        self.max_tokens = config.get("max_tokens", 4000)
        self.quality = config.get("quality", "balanced")

        # Enhanced execution tracking (initialize first)
        self.execution_log = {
            "agent_name": self.name,
            "config": config,
            "status": "initialized",
            "artifacts": [],
            "metrics": {},
            "errors": [],
            "warnings": [],
            "logs": [],
            "performance": {
                "start_time": None,
                "end_time": None,
                "duration_seconds": 0,
                "memory_usage": {},
                "llm_calls": 0,
                "llm_tokens": 0
            }
        }

        # Initialize Python logger for this agent
        self.logger = logging.getLogger(f"agent.{self.name.lower()}")
        self.logger.setLevel(logging.INFO)

        # Initialize LLM manager if not already done by subclass
        if not hasattr(self, 'llm_manager'):
            try:
                from ..llm_client import LLMManager
                self.llm_manager = LLMManager(config)
                self._log_info(f"Initialized LLM manager with model: {self.llm_manager.client.model}")
            except Exception as e:
                self._log_error(f"Failed to initialize LLM manager: {str(e)}")
                self.llm_manager = None
    
    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the agent with the given context
        
        Args:
            context: Execution context from previous agents
            
        Returns:
            Dict containing execution results, artifacts, and metrics
        """
        self.start_time = time.time()
        self.execution_log["start_time"] = datetime.now().isoformat()
        self.execution_log["performance"]["start_time"] = datetime.now().isoformat()

        # Track memory usage if available
        try:
            import psutil
            process = psutil.Process()
            self.execution_log["performance"]["memory_usage"]["start_mb"] = process.memory_info().rss / 1024 / 1024
        except ImportError:
            pass
        self.execution_log["status"] = "running"
        
        try:
            # Pre-execution setup
            self._pre_execute(context)
            
            # Main agent processing
            result = self._process(context)
            
            # Post-execution cleanup
            self._post_execute(context, result)
            
            # Update execution log with performance metrics
            self.end_time = time.time()
            execution_time = self.end_time - self.start_time

            self.execution_log["end_time"] = datetime.now().isoformat()
            self.execution_log["execution_time"] = execution_time
            self.execution_log["status"] = "completed"
            self.execution_log["result"] = result

            # Update performance tracking
            self.execution_log["performance"]["end_time"] = datetime.now().isoformat()
            self.execution_log["performance"]["duration_seconds"] = execution_time

            # Track final memory usage if available
            try:
                import psutil
                process = psutil.Process()
                self.execution_log["performance"]["memory_usage"]["end_mb"] = process.memory_info().rss / 1024 / 1024
                start_mb = self.execution_log["performance"]["memory_usage"].get("start_mb", 0)
                end_mb = self.execution_log["performance"]["memory_usage"]["end_mb"]
                self.execution_log["performance"]["memory_usage"]["delta_mb"] = end_mb - start_mb
            except ImportError:
                pass
            
            return {
                "status": "completed",
                "agent": self.name,
                "artifacts": result.get("artifacts", {}),
                "metrics": result.get("metrics", {}),
                "summary": result.get("summary", ""),
                "execution_time": self.execution_log["execution_time"]
            }
            
        except Exception as e:
            # Handle execution errors
            self.end_time = time.time()
            self.execution_log["end_time"] = datetime.now().isoformat()
            self.execution_log["execution_time"] = self.end_time - self.start_time
            self.execution_log["status"] = "error"
            self.execution_log["error"] = str(e)
            
            # Log error
            self._log_error(str(e))
            
            return {
                "status": "error",
                "agent": self.name,
                "error": str(e),
                "execution_time": self.execution_log["execution_time"],
                "critical_failure": self._is_critical_failure(e)
            }
    
    @abstractmethod
    def _process(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main processing method - must be implemented by subclasses
        
        Args:
            context: Execution context
            
        Returns:
            Dict containing processing results
        """
        pass
    
    def _pre_execute(self, context: Dict[str, Any]):
        """
        Pre-execution setup - can be overridden by subclasses
        
        Args:
            context: Execution context
        """
        # Create agent-specific output directory
        output_dir = context.get("output_directory", "./output")
        self.agent_output_dir = f"{output_dir}/{self.name.lower()}"
        create_directory(self.agent_output_dir)
        
        # Log agent start
        self._log_info(f"Starting {self.name} execution")
    
    def _post_execute(self, context: Dict[str, Any], result: Dict[str, Any]):
        """
        Post-execution cleanup - can be overridden by subclasses
        
        Args:
            context: Execution context
            result: Processing results
        """
        # Save execution log
        log_file = f"{self.agent_output_dir}/execution_log.json"
        save_json(self.execution_log, log_file)

        # Log agent completion with context info
        self._log_info(f"Completed {self.name} execution with {len(context)} context items and {len(result)} result items")
    
    def _log_info(self, message: str):
        """Log informational message"""
        timestamp = datetime.now().isoformat()
        log_entry = {
            "timestamp": timestamp,
            "level": "INFO",
            "message": message
        }
        
        if "logs" not in self.execution_log:
            self.execution_log["logs"] = []
        self.execution_log["logs"].append(log_entry)
        
        # Use Python logger for better integration
        self.logger.info(f"{message}")
    
    def _log_error(self, message: str):
        """Log error message"""
        timestamp = datetime.now().isoformat()
        log_entry = {
            "timestamp": timestamp,
            "level": "ERROR", 
            "message": message
        }
        
        if "logs" not in self.execution_log:
            self.execution_log["logs"] = []
        self.execution_log["logs"].append(log_entry)
        self.execution_log["errors"].append(log_entry)
        
        # Use Python logger for better integration
        self.logger.error(f"{message}")
    
    def _log_warning(self, message: str):
        """Log warning message"""
        timestamp = datetime.now().isoformat()
        log_entry = {
            "timestamp": timestamp,
            "level": "WARNING",
            "message": message
        }
        
        if "logs" not in self.execution_log:
            self.execution_log["logs"] = []
        self.execution_log["logs"].append(log_entry)
        
        # Use Python logger for better integration
        self.logger.warning(f"{message}")

    def _track_llm_usage(self, tokens_used: int = 0, call_count: int = 1):
        """Track LLM usage for performance monitoring"""
        self.execution_log["performance"]["llm_calls"] += call_count
        self.execution_log["performance"]["llm_tokens"] += tokens_used

        self._log_info(f"LLM usage: {call_count} calls, {tokens_used} tokens", {
            "total_calls": self.execution_log["performance"]["llm_calls"],
            "total_tokens": self.execution_log["performance"]["llm_tokens"]
        })

    def _is_critical_failure(self, error: Exception) -> bool:
        """
        Determine if an error is critical and should stop the pipeline
        
        Args:
            error: The exception that occurred
            
        Returns:
            True if the error is critical, False otherwise
        """
        # Define critical error types
        critical_errors = [
            "FileNotFoundError",
            "PermissionError", 
            "ImportError",
            "ModuleNotFoundError"
        ]
        
        error_type = type(error).__name__
        return error_type in critical_errors
    
    def _create_artifact(self, name: str, content: Any, artifact_type: str = "file") -> Dict[str, Any]:
        """
        Create an artifact entry
        
        Args:
            name: Artifact name
            content: Artifact content
            artifact_type: Type of artifact (file, data, etc.)
            
        Returns:
            Artifact dictionary
        """
        artifact = {
            "name": name,
            "type": artifact_type,
            "content": content,
            "created_at": datetime.now().isoformat(),
            "agent": self.name
        }
        
        self.execution_log["artifacts"].append(artifact)
        return artifact
    
    def _update_metrics(self, metrics: Dict[str, Any]):
        """
        Update agent metrics
        
        Args:
            metrics: Dictionary of metrics to update
        """
        self.execution_log["metrics"].update(metrics)
    
    def _get_quality_settings(self) -> Dict[str, Any]:
        """
        Get quality-specific settings for the agent
        
        Returns:
            Dictionary of quality settings
        """
        quality_settings = {
            "fast": {
                "max_iterations": 1,
                "detail_level": "basic",
                "validation_level": "minimal"
            },
            "balanced": {
                "max_iterations": 3,
                "detail_level": "standard", 
                "validation_level": "standard"
            },
            "high": {
                "max_iterations": 5,
                "detail_level": "comprehensive",
                "validation_level": "thorough"
            }
        }
        
        return quality_settings.get(self.quality, quality_settings["balanced"])
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get current agent status
        
        Returns:
            Status dictionary
        """
        return {
            "name": self.name,
            "status": self.execution_log["status"],
            "start_time": self.execution_log.get("start_time"),
            "execution_time": time.time() - self.start_time if self.start_time else 0,
            "artifacts_count": len(self.execution_log["artifacts"]),
            "errors_count": len(self.execution_log["errors"]),
            "mcp_enabled": self.mcp_enabled,
            "mcp_connected": self._is_mcp_connected()
        }

    # MCP Integration Methods

    async def _connect_mcp(self) -> bool:
        """Connect to MCP servers"""
        if not self.mcp_enabled or not self.mcp_manager:
            return False

        try:
            results = await self.mcp_manager.connect_all()
            connected_count = sum(1 for success in results.values() if success)
            total_count = len(results)

            self._log_info(f"Connected to {connected_count}/{total_count} MCP servers")
            return connected_count > 0

        except Exception as e:
            self._log_error(f"Failed to connect to MCP servers: {str(e)}")
            return False

    async def _disconnect_mcp(self):
        """Disconnect from MCP servers"""
        if self.mcp_manager:
            try:
                await self.mcp_manager.disconnect_all()
                self._log_info("Disconnected from MCP servers")
            except Exception as e:
                self._log_error(f"Error disconnecting from MCP servers: {str(e)}")

    def _is_mcp_connected(self) -> bool:
        """Check if any MCP servers are connected"""
        if not self.mcp_manager:
            return False

        return any(
            self.mcp_manager.is_connected(client_name)
            for client_name in self.mcp_manager.clients.keys()
        )

    async def _call_mcp_tool(self, client_name: str, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Call an MCP tool"""
        if not self.mcp_manager:
            return {"success": False, "error": "MCP not enabled"}

        try:
            result = await self.mcp_manager.call_tool(client_name, tool_name, arguments)
            self._log_info(f"Called MCP tool {tool_name} on {client_name}")
            return result
        except Exception as e:
            self._log_error(f"MCP tool call failed: {str(e)}")
            return {"success": False, "error": str(e)}

    async def _get_mcp_tools(self) -> Dict[str, Any]:
        """Get all available MCP tools"""
        if not self.mcp_manager:
            return {}

        try:
            tools = await self.mcp_manager.get_all_tools()
            return {
                client_name: [tool.to_dict() for tool in tool_list]
                for client_name, tool_list in tools.items()
            }
        except Exception as e:
            self._log_error(f"Failed to get MCP tools: {str(e)}")
            return {}

    def _run_async_mcp_operation(self, coro):
        """Run an async MCP operation in sync context"""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If we're already in an async context, we need to handle this differently
                # For now, we'll skip MCP operations in nested async contexts
                self._log_warning("Skipping MCP operation in nested async context")
                return None
            else:
                return loop.run_until_complete(coro)
        except RuntimeError:
            # No event loop, create one
            return asyncio.run(coro)
        except Exception as e:
            self._log_error(f"Error running async MCP operation: {str(e)}")
            return None
