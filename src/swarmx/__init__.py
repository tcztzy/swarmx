"""SwarmX: A lightweight, stateless multi-agent orchestration framework."""

from .agent import (
    __CTX_VARS_NAME__,
    Agent,
    Result,
    check_instructions,
    does_function_need_context,
    merge_chunk,
    merge_chunks,
    validate_tool,
    validate_tools,
)
from .cli import app
from .mcp_client import ToolRegistry, function_to_json
from .swarm import (
    Swarm,
    _image_content_to_url,
    _mcp_call_tool_result_to_content,
    _resource_to_file,
    content_part_to_str,
    messages_to_chunks,
)
from .version import __version__ as __version__

__all__ = (
    "__CTX_VARS_NAME__",
    "Agent",
    "Swarm",
    "Result",
    "ToolRegistry",
    "_image_content_to_url",
    "_mcp_call_tool_result_to_content",
    "_resource_to_file",
    "app",
    "check_instructions",
    "content_part_to_str",
    "does_function_need_context",
    "function_to_json",
    "merge_chunk",
    "merge_chunks",
    "messages_to_chunks",
    "validate_tool",
    "validate_tools",
)
