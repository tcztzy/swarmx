"""Tool module."""

from datetime import timedelta
from typing import Any, Self

import jsonschema
from mcp import ClientSession
from mcp import Tool as MCPTool
from mcp.shared.session import ProgressFnT
from mcp.types import CallToolResult
from pydantic import PrivateAttr

from .node import Node
from .types import MessagesState


class MCPCallToolResultState(MessagesState):
    """State with CallToolResult."""

    result: CallToolResult


class Tool(Node):
    """Tool node."""

    _session: ClientSession | None = PrivateAttr(default=None)

    @classmethod
    def from_mcp(cls, mcp_tool: MCPTool) -> Self:
        """Create from MCP tool."""
        return cls(
            name=mcp_tool.name,
            description=mcp_tool.description,
            parameters=mcp_tool.inputSchema,
            returns=mcp_tool.outputSchema,
        )

    def model_post_init(self, __context: Any) -> None:
        """Initialize MCP session reference."""
        if self.model_extra is None:
            self._session = None
        else:
            self._session = self.model_extra.pop("session", None)

    async def __call__(
        self,
        arguments: dict[str, Any] | None = None,
        *,
        timeout: float | None = None,
        progress_callback: ProgressFnT | None = None,
        meta: dict[str, Any] | None = None,
        **kwargs,
    ) -> CallToolResult:
        """Execute the targeted MCP tool call."""
        if self._session is None:
            raise RuntimeError(
                f"Client session not found, this tool `{self.name}` could not be called."
            )
        jsonschema.validate(arguments, self.parameters or {})
        result = await self._session.call_tool(
            self.name,
            arguments,
            None if timeout is None else timedelta(seconds=timeout),
            progress_callback,
            meta=meta,
        )
        return result
