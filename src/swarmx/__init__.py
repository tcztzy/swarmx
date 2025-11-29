"""SwarmX: A lightweight, stateless multi-agent orchestration framework."""

from ._settings import settings
from .agent import Agent
from .cli import app
from .edge import Edge
from .hook import Hook
from .mcp_manager import MCPManager
from .node import Node
from .tool import Tool
from .version import __version__

__all__ = (
    "__version__",
    "Agent",
    "Edge",
    "Hook",
    "MCPManager",
    "Node",
    "Tool",
    "app",
    "settings",
)
