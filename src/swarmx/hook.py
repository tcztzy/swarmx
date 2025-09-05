"""SwarmX Hook module."""

import logging
from typing import Literal

from pydantic import BaseModel

logger = logging.getLogger(__name__)


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

    on_tool_start: str | None = None
    """Tool name to execute before any tool call"""

    on_tool_end: str | None = None
    """Tool name to execute after any tool call"""

    on_llm_start: str | None = None
    """Tool name to execute before LLM call"""

    on_llm_end: str | None = None
    """Tool name to execute after LLM call"""

    on_chunk: str | None = None
    """Tool name to execute after each stream data chunk"""


HookType = Literal[
    "on_start",
    "on_end",
    "on_handoff",
    "on_tool_start",
    "on_tool_end",
    "on_llm_start",
    "on_llm_end",
    "on_chunk",
]
