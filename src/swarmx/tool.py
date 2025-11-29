"""Tool module."""

from datetime import timedelta
from typing import Annotated, Any, Self

import jsonschema
from mcp import ClientSession
from mcp import Tool as MCPTool
from mcp.shared.session import ProgressFnT
from mcp.types import CallToolResult
from openai.types.shared import FunctionDefinition
from pydantic import Field, PrivateAttr


class Tool(FunctionDefinition):
    """Tool node."""

    name: Annotated[
        str,
        Field(
            pattern=r"^[A-Za-z][A-Za-z0-9_-]{0,63}$",
            frozen=True,
            max_length=64,
        ),
    ]
    returns: dict[str, Any] | None = None

    def __hash__(self):
        """Tool name is unique among all MCP servers and not conflict with any agent's name."""
        return hash(self.name)

    @classmethod
    def from_mcp(cls, mcp_tool: MCPTool) -> Self:
        """Create from MCP tool."""
        return cls(
            name=mcp_tool.name,
            description=mcp_tool.description,
            parameters=mcp_tool.inputSchema,
            returns=mcp_tool.outputSchema,
        )

    _session: ClientSession | None = PrivateAttr(default=None)

    def model_post_init(self, __context: Any) -> None:
        """Initialize MCP session reference."""
        if self.model_extra is None:
            self._session = None
        else:
            self._session = self.model_extra.pop("session", None)

    async def __call__(
        self,
        arguments: dict[str, Any] | None = None,
        read_timeout_seconds: timedelta | None = None,
        progress_callback: ProgressFnT | None = None,
        *,
        meta: dict[str, Any] | None = None,
    ) -> CallToolResult:
        """Execute the targeted MCP tool call."""
        if self._session is None:
            raise RuntimeError(
                f"Client session not found, this tool `{self.name}` could not be called."
            )
        jsonschema.validate(arguments, self.parameters or {})
        return await self._session.call_tool(
            self.name.split("__", maxsplit=2)[2],
            arguments,
            read_timeout_seconds,
            progress_callback,
            meta=meta,
        )
