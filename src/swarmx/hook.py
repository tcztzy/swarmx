"""SwarmX Hook module."""

import logging
import sys
from typing import Any, Literal, TypeVar

from openai.types.chat import ChatCompletionMessageParam
from pydantic import BaseModel, TypeAdapter

from .mcp_client import CLIENT_REGISTRY

PY312 = sys.version_info >= (3, 12)
if PY312:  # for pydantic.TypeAdapter
    from typing import TypedDict
else:
    from typing_extensions import TypedDict

logger = logging.getLogger(__name__)
T = TypeVar("T", bound=dict | BaseModel)


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

    on_tool_start: str | None = None
    """Tool name to execute before any tool call"""

    on_tool_end: str | None = None
    """Tool name to execute after any tool call"""

    on_llm_start: str | None = None
    """Tool name to execute before LLM call"""

    on_llm_end: str | None = None
    """Tool name to execute after LLM call"""


HookType = Literal[
    "on_start",
    "on_end",
    "on_tool_start",
    "on_tool_end",
    "on_llm_start",
    "on_llm_end",
]


class HookOutput(TypedDict, total=False):
    """Output structure for hook tools.

    Hook tools should return this structure to modify messages and context.
    If no modifications are needed, return the input unchanged.

    """

    messages: list[ChatCompletionMessageParam]
    context: dict[str, Any] | None


async def execute_hooks(
    hooks: list[Hook],
    hook_type: HookType,
    messages: list[ChatCompletionMessageParam],
    context: T | None = None,
) -> tuple[list[ChatCompletionMessageParam], T | None]:
    """Execute hooks of a specific type.

    Args:
        hooks: List of hooks to execute
        hook_type: The type of hook to execute (e.g., 'on_start', 'on_end')
        messages: The current messages to pass to hook tools
        context: The context variables to pass to the hook tools

    Returns:
        Tuple of (modified_messages, modified_context)

    """
    current_messages = messages
    current_context = context

    for hook in hooks:
        tool_name = getattr(hook, hook_type, None)
        if tool_name is not None:
            try:
                hook_input = {
                    "messages": current_messages,
                    "context": (
                        current_context.model_dump()
                        if isinstance(current_context, BaseModel)
                        else current_context
                    ),
                }
                result = await CLIENT_REGISTRY.call_tool(tool_name, hook_input)
                if result.structuredContent is not None:
                    output = TypeAdapter(HookOutput).validate_python(
                        result.structuredContent
                    )
                    if new_context := output.get("context"):
                        if isinstance(context, BaseModel):
                            # For BaseModel, update attributes
                            for k, v in new_context.items():
                                setattr(context, k, v)
                        else:
                            # For dict or None, replace completely
                            context = new_context  # type: ignore
                            current_context = context
                    if new_messages := output.get("messages"):
                        current_messages = new_messages

            except Exception as e:
                logger.warning(f"Hook {hook_type} failed for tool {tool_name}: {e}")

    return current_messages, current_context  # type: ignore
