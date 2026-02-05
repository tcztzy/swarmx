"""SwarmX: A lightweight, stateless multi-agent orchestration framework."""

from ._settings import settings
from .agent import Agent
from .cli import app
from .edge import Edge
from .hook import Hook
from .mcp_manager import MCPManager
from .messages import Messages
from .node import Node
from .swarm import Swarm
from .tool import Tool
from .version import __version__

__all__ = (
    "__version__",
    "Agent",
    "Edge",
    "Hook",
    "Messages",
    "MCPManager",
    "Node",
    "Swarm",
    "Tool",
    "app",
    "settings",
)
