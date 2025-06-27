"""
MCP Multi-Agent Framework for ComfyUI Nodes
Automated ComfyUI node development using AI agents
"""

from .framework import MCPFramework
from .agents import (
    BaseAgent,
    ResearchAgent,
    DevOpsAgent,
    CodingAgent,
    TestingAgent,
    DocumentationAgent
)
from .utils import (
    setup_logging,
    validate_input,
    create_output_structure,
    validate_comfyui_node
)

__version__ = "1.0.0"
__author__ = "A043 Studios"
__email__ = "contact@a043studios.com"
__description__ = "Automated ComfyUI node development using AI agents and MCP"

__all__ = [
    "MCPFramework",
    "BaseAgent",
    "ResearchAgent",
    "DevOpsAgent", 
    "CodingAgent",
    "TestingAgent",
    "DocumentationAgent",
    "setup_logging",
    "validate_input",
    "create_output_structure",
    "validate_comfyui_node"
]
