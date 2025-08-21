"""SwarmX: A lightweight, stateless multi-agent orchestration framework."""

# Load environment variables from .env file if it exists
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # pragma: no cover
    # python-dotenv not available, skip loading
    pass

from .agent import Agent
from .cli import app
from .hook import Hook, HookOutput
from .mcp_client import ClientRegistry
from .version import __version__

__all__ = (
    "__version__",
    "Agent",
    "Hook",
    "HookOutput",
    "ClientRegistry",
    "app",
)
