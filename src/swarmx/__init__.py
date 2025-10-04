"""SwarmX: A lightweight, stateless multi-agent orchestration framework."""

from ._settings import settings
from .agent import Agent
from .cli import app
from .edge import Edge
from .hook import Hook
from .mcp_client import ClientRegistry
from .version import __version__

__all__ = (
    "__version__",
    "Agent",
    "Edge",
    "Hook",
    "ClientRegistry",
    "app",
    "settings",
)
