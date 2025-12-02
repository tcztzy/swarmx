"""MCP client related."""

from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.sse import sse_client
from mcp.client.stdio import stdio_client
from mcp.shared.session import ProgressFnT
from mcp.types import Tool
from openai.types.chat import ChatCompletionFunctionToolParam

from .types import MCPServer


@dataclass
class MCPManager:
    """Registry for tools."""

    exit_stack: AsyncExitStack = field(default_factory=AsyncExitStack)
    mcp_clients: dict[str, ClientSession] = field(default_factory=dict)
    _tools: dict[str, list[Tool]] = field(default_factory=dict)

    @property
    def tools(self) -> list[ChatCompletionFunctionToolParam]:
        """Return all tools, both local and MCP."""
        _tools = []
        for tools in self._tools.values():
            for tool in tools:
                _tool = ChatCompletionFunctionToolParam(
                    type="function",
                    function={
                        "name": tool.name,
                        "parameters": tool.inputSchema,
                    },
                )
                if tool.description:
                    _tool["function"]["description"] = tool.description
                _tools.append(_tool)
        return _tools

    def _parse_name(self, name: str) -> tuple[str, Tool]:
        for server_name, tools in self._tools.items():
            for tool in tools:
                if tool.name == name:
                    return server_name, tool
        raise KeyError()

    def get_tool(self, name: str) -> Tool:
        """Get Tool by name."""
        _, tool = self._parse_name(name)
        return tool

    async def call_tool(
        self,
        name: str,
        arguments: dict[str, Any] | None = None,
        read_timeout_seconds: timedelta | None = None,
        progress_callback: ProgressFnT | None = None,
        *,
        meta: dict[str, Any] | None = None,
    ):
        """Call a tool.

        Args:
            name: The name of the tool
            arguments: The arguments to pass to the tool
            read_timeout_seconds: The read timeout for the tool call
            progress_callback: The progress callback for the tool call
            meta: The meta data for the tool call

        """
        server_name, tool = self._parse_name(name)
        return await self.mcp_clients[server_name].call_tool(
            tool.name, arguments, read_timeout_seconds, progress_callback, meta=meta
        )

    async def add_server(self, name: str, server_params: MCPServer):
        """Add an MCP server to the registry.

        Args:
            name: The name of the server
            server_params: The parameters to connect to the server

        """
        if name in self.mcp_clients:
            return
        read_stream, write_stream = await self.exit_stack.enter_async_context(
            stdio_client(server_params)
            if isinstance(server_params, StdioServerParameters)
            else sse_client(server_params.url, server_params.headers)
        )
        client = await self.exit_stack.enter_async_context(
            ClientSession(read_stream, write_stream)
        )
        await client.initialize()
        self.mcp_clients[name] = client
        self._tools[name] = (await client.list_tools()).tools

    async def close(self):
        """Close all clients."""
        await self.exit_stack.aclose()
        self.mcp_clients = {}
        self._tools = {}
