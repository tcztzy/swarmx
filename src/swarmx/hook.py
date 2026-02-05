"""Hook module."""

from typing import Literal

from pydantic import BaseModel


class Hook(BaseModel):
    """Hook for agent lifecycle events.

    Each field represents a hook point and contains the name of an MCP Tool
    to execute at that point. This makes the Hook serializable since it only
    stores tool names rather than function references.
    """

    on_start: str | None = None
    """Tool name to execute when agent starts processing"""

    on_end: str | None = None
    """Tool name to execute when agent finishes processing"""

    on_handoff: str | None = None
    """Tool name to execute when agent hands off to another agent"""

    on_chunk: str | None = None
    """Tool name to execute after each stream data chunk"""


HookType = Literal[
    "on_start",
    "on_end",
    "on_handoff",
    "on_chunk",
]
