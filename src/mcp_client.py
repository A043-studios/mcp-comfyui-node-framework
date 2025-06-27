"""
MCP (Model Context Protocol) Client for Multi-Agent Framework
Provides access to external tools and capabilities through MCP servers
"""

import asyncio
import json
import logging
from typing import Dict, Any, List, Optional, Union
from abc import ABC, abstractmethod

try:
    import websockets
    WEBSOCKETS_AVAILABLE = True
except ImportError:
    WEBSOCKETS_AVAILABLE = False

try:
    # Try to import MCP if available
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False


class MCPTool:
    """Represents an MCP tool with its capabilities"""
    
    def __init__(self, name: str, description: str, input_schema: Dict[str, Any]):
        self.name = name
        self.description = description
        self.input_schema = input_schema
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema
        }


class MCPResource:
    """Represents an MCP resource"""
    
    def __init__(self, uri: str, name: str, description: str, mime_type: str = None):
        self.uri = uri
        self.name = name
        self.description = description
        self.mime_type = mime_type
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "uri": self.uri,
            "name": self.name,
            "description": self.description,
            "mime_type": self.mime_type
        }


class BaseMCPClient(ABC):
    """Abstract base class for MCP clients"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self.tools: Dict[str, MCPTool] = {}
        self.resources: Dict[str, MCPResource] = {}
        self.connected = False
    
    @abstractmethod
    async def connect(self) -> bool:
        """Connect to the MCP server"""
        pass
    
    @abstractmethod
    async def disconnect(self):
        """Disconnect from the MCP server"""
        pass
    
    @abstractmethod
    async def list_tools(self) -> List[MCPTool]:
        """List available tools"""
        pass
    
    @abstractmethod
    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Call a tool with arguments"""
        pass
    
    @abstractmethod
    async def list_resources(self) -> List[MCPResource]:
        """List available resources"""
        pass
    
    @abstractmethod
    async def read_resource(self, uri: str) -> Dict[str, Any]:
        """Read a resource"""
        pass


class StdioMCPClient(BaseMCPClient):
    """MCP client that communicates via stdio"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.server_params = None
        self.session = None
        
        if not MCP_AVAILABLE:
            raise ImportError("MCP package not available. Install with: pip install mcp")
    
    async def connect(self) -> bool:
        """Connect to stdio MCP server"""
        try:
            server_command = self.config.get("server_command")
            server_args = self.config.get("server_args", [])
            
            if not server_command:
                raise ValueError("server_command required for stdio MCP client")
            
            self.server_params = StdioServerParameters(
                command=server_command,
                args=server_args
            )
            
            # Create client session
            async with stdio_client(self.server_params) as (read, write):
                self.session = ClientSession(read, write)
                
                # Initialize the session
                await self.session.initialize()
                
                # Load available tools and resources
                await self._load_capabilities()
                
                self.connected = True
                self.logger.info(f"Connected to MCP server: {server_command}")
                return True
                
        except Exception as e:
            self.logger.error(f"Failed to connect to MCP server: {str(e)}")
            self.connected = False
            return False
    
    async def disconnect(self):
        """Disconnect from MCP server"""
        if self.session:
            try:
                await self.session.close()
                self.session = None
                self.connected = False
                self.logger.info("Disconnected from MCP server")
            except Exception as e:
                self.logger.error(f"Error disconnecting from MCP server: {str(e)}")
    
    async def _load_capabilities(self):
        """Load tools and resources from the server"""
        try:
            # Load tools
            tools_response = await self.session.list_tools()
            for tool_info in tools_response.tools:
                tool = MCPTool(
                    name=tool_info.name,
                    description=tool_info.description,
                    input_schema=tool_info.inputSchema
                )
                self.tools[tool.name] = tool
            
            # Load resources
            resources_response = await self.session.list_resources()
            for resource_info in resources_response.resources:
                resource = MCPResource(
                    uri=resource_info.uri,
                    name=resource_info.name,
                    description=resource_info.description,
                    mime_type=getattr(resource_info, 'mimeType', None)
                )
                self.resources[resource.uri] = resource
                
            self.logger.info(f"Loaded {len(self.tools)} tools and {len(self.resources)} resources")
            
        except Exception as e:
            self.logger.error(f"Failed to load capabilities: {str(e)}")
    
    async def list_tools(self) -> List[MCPTool]:
        """List available tools"""
        return list(self.tools.values())
    
    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Call a tool with arguments"""
        if not self.connected or not self.session:
            raise RuntimeError("Not connected to MCP server")
        
        if tool_name not in self.tools:
            raise ValueError(f"Tool '{tool_name}' not available")
        
        try:
            response = await self.session.call_tool(tool_name, arguments)
            return {
                "success": True,
                "result": response.content,
                "tool_name": tool_name,
                "arguments": arguments
            }
        except Exception as e:
            self.logger.error(f"Tool call failed for '{tool_name}': {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "tool_name": tool_name,
                "arguments": arguments
            }
    
    async def list_resources(self) -> List[MCPResource]:
        """List available resources"""
        return list(self.resources.values())
    
    async def read_resource(self, uri: str) -> Dict[str, Any]:
        """Read a resource"""
        if not self.connected or not self.session:
            raise RuntimeError("Not connected to MCP server")
        
        if uri not in self.resources:
            raise ValueError(f"Resource '{uri}' not available")
        
        try:
            response = await self.session.read_resource(uri)
            return {
                "success": True,
                "content": response.contents,
                "uri": uri
            }
        except Exception as e:
            self.logger.error(f"Resource read failed for '{uri}': {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "uri": uri
            }


class MockMCPClient(BaseMCPClient):
    """Mock MCP client for testing when MCP is not available"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self._setup_mock_tools()
    
    def _setup_mock_tools(self):
        """Setup mock tools for testing"""
        self.tools = {
            "web_search": MCPTool(
                name="web_search",
                description="Search the web for information",
                input_schema={"type": "object", "properties": {"query": {"type": "string"}}}
            ),
            "file_read": MCPTool(
                name="file_read",
                description="Read file contents",
                input_schema={"type": "object", "properties": {"path": {"type": "string"}}}
            ),
            "code_analysis": MCPTool(
                name="code_analysis",
                description="Analyze code for patterns and issues",
                input_schema={"type": "object", "properties": {"code": {"type": "string"}}}
            )
        }
    
    async def connect(self) -> bool:
        """Mock connection"""
        self.connected = True
        self.logger.info("Connected to mock MCP server")
        return True
    
    async def disconnect(self):
        """Mock disconnection"""
        self.connected = False
        self.logger.info("Disconnected from mock MCP server")
    
    async def list_tools(self) -> List[MCPTool]:
        """List mock tools"""
        return list(self.tools.values())
    
    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Mock tool call"""
        if tool_name not in self.tools:
            return {"success": False, "error": f"Tool '{tool_name}' not available"}
        
        # Return mock responses
        mock_responses = {
            "web_search": {"results": ["Mock search result 1", "Mock search result 2"]},
            "file_read": {"content": "Mock file content"},
            "code_analysis": {"issues": [], "suggestions": ["Mock suggestion"]}
        }
        
        return {
            "success": True,
            "result": mock_responses.get(tool_name, {"mock": "response"}),
            "tool_name": tool_name,
            "arguments": arguments
        }
    
    async def list_resources(self) -> List[MCPResource]:
        """List mock resources"""
        return []
    
    async def read_resource(self, uri: str) -> Dict[str, Any]:
        """Mock resource read"""
        return {
            "success": True,
            "content": "Mock resource content",
            "uri": uri
        }


class MCPClientFactory:
    """Factory for creating MCP clients"""
    
    @classmethod
    def create_client(cls, config: Dict[str, Any]) -> BaseMCPClient:
        """Create appropriate MCP client based on config"""
        
        client_type = config.get("type", "stdio")
        
        if not MCP_AVAILABLE:
            cls._log_warning("MCP not available, using mock client")
            return MockMCPClient(config)
        
        if client_type == "stdio":
            return StdioMCPClient(config)
        else:
            raise ValueError(f"Unsupported MCP client type: {client_type}")
    
    @classmethod
    def _log_warning(cls, message: str):
        """Log warning message"""
        logger = logging.getLogger(cls.__name__)
        logger.warning(message)


class MCPManager:
    """Manager for MCP operations with multiple clients"""
    
    def __init__(self, configs: List[Dict[str, Any]]):
        self.configs = configs
        self.clients: Dict[str, BaseMCPClient] = {}
        self.logger = logging.getLogger(self.__class__.__name__)
    
    async def connect_all(self) -> Dict[str, bool]:
        """Connect to all configured MCP servers"""
        results = {}
        
        for config in self.configs:
            client_name = config.get("name", f"client_{len(self.clients)}")
            
            try:
                client = MCPClientFactory.create_client(config)
                success = await client.connect()
                
                if success:
                    self.clients[client_name] = client
                    results[client_name] = True
                    self.logger.info(f"Connected to MCP client: {client_name}")
                else:
                    results[client_name] = False
                    self.logger.error(f"Failed to connect to MCP client: {client_name}")
                    
            except Exception as e:
                results[client_name] = False
                self.logger.error(f"Error creating MCP client '{client_name}': {str(e)}")
        
        return results
    
    async def disconnect_all(self):
        """Disconnect from all MCP servers"""
        for client_name, client in self.clients.items():
            try:
                await client.disconnect()
                self.logger.info(f"Disconnected from MCP client: {client_name}")
            except Exception as e:
                self.logger.error(f"Error disconnecting from '{client_name}': {str(e)}")
        
        self.clients.clear()
    
    async def get_all_tools(self) -> Dict[str, List[MCPTool]]:
        """Get all tools from all connected clients"""
        all_tools = {}
        
        for client_name, client in self.clients.items():
            try:
                tools = await client.list_tools()
                all_tools[client_name] = tools
            except Exception as e:
                self.logger.error(f"Error getting tools from '{client_name}': {str(e)}")
                all_tools[client_name] = []
        
        return all_tools
    
    async def call_tool(self, client_name: str, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Call a tool on a specific client"""
        if client_name not in self.clients:
            return {"success": False, "error": f"Client '{client_name}' not connected"}
        
        client = self.clients[client_name]
        return await client.call_tool(tool_name, arguments)
    
    def get_client(self, client_name: str) -> Optional[BaseMCPClient]:
        """Get a specific client by name"""
        return self.clients.get(client_name)
    
    def is_connected(self, client_name: str) -> bool:
        """Check if a client is connected"""
        client = self.clients.get(client_name)
        return client is not None and client.connected
