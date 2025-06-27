"""
MCP Multi-Agent Framework - Agent Implementations
"""

try:
    # Try relative imports first (when imported as a package)
    from .base_agent import BaseAgent
    from .research_agent import ResearchAgent
    from .devops_agent import DevOpsAgent
    from .coding_agent import CodingAgent
    from .testing_agent import TestingAgent
    from .documentation_agent import DocumentationAgent
except ImportError:
    # Fallback to absolute imports (when run directly)
    from base_agent import BaseAgent
    from research_agent import ResearchAgent
    from devops_agent import DevOpsAgent
    from coding_agent import CodingAgent
    from testing_agent import TestingAgent
    from documentation_agent import DocumentationAgent

__all__ = [
    "BaseAgent",
    "ResearchAgent",
    "DevOpsAgent",
    "CodingAgent",
    "TestingAgent",
    "DocumentationAgent"
]
