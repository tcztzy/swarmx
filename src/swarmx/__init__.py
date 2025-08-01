"""SwarmX: A lightweight, stateless multi-agent orchestration framework."""

from .agent import Agent
from .cli import app
from .mcp_client import ClientRegistry
from .version import __version__

__all__ = (
    "__version__",
    "Agent",
    "ClientRegistry",
    "app",
)
