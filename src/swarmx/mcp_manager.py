"""MCP client related."""

import mimetypes
import os
import re
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any, assert_never

from mcp import ClientSession, StdioServerParameters
from mcp.client.sse import sse_client
from mcp.client.stdio import stdio_client
from mcp.shared.session import ProgressFnT
from mcp.types import (
    BlobResourceContents,
    CallToolResult,
    ContentBlock,
    EmbeddedResource,
    ImageContent,
    ResourceContents,
    TextResourceContents,
)
from mcp.types import Tool as _Tool
from openai.types.chat import (
    ChatCompletionContentPartTextParam,
    ChatCompletionFunctionToolParam,
    ChatCompletionToolMessageParam,
)
from pygments.lexers import get_lexer_for_filename, get_lexer_for_mimetype
from pygments.util import ClassNotFound

from .node import Tool
from .types import MCPServer

mimetypes.add_type("text/markdown", ".md")


@dataclass
class MCPManager:
    """Registry for tools."""

    exit_stack: AsyncExitStack = field(default_factory=AsyncExitStack)
    mcp_clients: dict[str, ClientSession] = field(default_factory=dict)
    _tools: dict[str, list[_Tool]] = field(default_factory=dict)

    @property
    def tools(self) -> list[ChatCompletionFunctionToolParam]:
        """Return all tools, both local and MCP."""
        _tools = []
        for server, tools in self._tools.items():
            for tool in tools:
                _tool = ChatCompletionFunctionToolParam(
                    type="function",
                    function={
                        "name": f"mcp__{server}__{tool.name}",
                        "parameters": tool.inputSchema,
                    },
                )
                if tool.description:
                    _tool["function"]["description"] = tool.description
                _tools.append(_tool)
        return _tools

    def _parse_name(self, name: str) -> tuple[str, _Tool]:
        if (mo := re.match(r"mcp__(?P<server>[^/]+)__(?P<name>[^/]+)", name)) is None:
            raise ValueError("Invalid tool name, expected mcp__<server>__<tool>")
        server_name, tool_name = mo.group("server"), mo.group("name")
        if server_name not in self.mcp_clients:
            raise KeyError(f"Server {server_name} not found")
        for tool in self._tools[server_name]:
            if tool.name == tool_name:
                return server_name, tool
        raise KeyError(f"Tool {tool_name} not found")

    def get_tool(self, name: str) -> _Tool:
        """Get Tool by name."""
        _, tool = self._parse_name(name)
        return tool

    def make_tool_node(self, name: str, tool_call_id: str) -> Tool:
        """Create a graph Tool node bound to this manager."""
        server_name, tool = self._parse_name(name)
        payload = tool.model_dump(mode="python")
        payload["tool_name"] = name
        payload["tool_call_id"] = tool_call_id
        payload["name"] = f"{name}__call__{self._sanitize_call_id(tool_call_id)}"
        payload["mcp_manager"] = self
        payload.setdefault(
            "description",
            f"{tool.description or ''} (from {server_name})".strip(),
        )
        return Tool.model_validate(payload)

    @staticmethod
    def _sanitize_call_id(tool_call_id: str) -> str:
        """Return a filesystem-safe representation of a tool call id."""
        safe = re.sub(r"[^A-Za-z0-9_]", "_", tool_call_id)
        return safe or "call"

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


def _image_to_md(
    content: ImageContent,
) -> str:
    alt = (content.meta or {}).get("alt", "")
    return f"![{alt}](data:{content.mimeType};base64,{content.data})"


def _resource_to_md(
    resource: EmbeddedResource,
) -> str:
    def get_filename(c: ResourceContents) -> str:
        if c.uri.path is not None:
            filename = os.path.basename(c.uri.path)
        elif c.uri.host is not None:
            filename = c.uri.host
        else:
            raise ValueError("Cannot determine filename for resource.")
        return filename

    def get_lang(c: ResourceContents) -> str:
        if c.mimeType is None:
            try:
                lexer = get_lexer_for_filename(get_filename(c))
            except ClassNotFound:
                lexer = None
        else:
            try:
                lexer = get_lexer_for_mimetype(c.mimeType)
            except ClassNotFound:
                lexer = None
        return lexer.aliases[0] if lexer else "text"

    match resource.resource:
        case TextResourceContents() as c:
            # For markdown/html/plain, insert directly with HTML comment delimiters
            if c.mimeType in ("text/markdown", "text/plain"):
                return f"\n<!-- begin {c.uri} -->\n{c.text}\n<!-- end {c.uri} -->\n"
            # For other types, wrap in code block
            lang = get_lang(c)
            return f'\n```{lang} title="{c.uri}"\n{c.text}\n```\n'
        case BlobResourceContents() as c:
            return (
                "<embed"
                f' type="{c.mimeType or "application/octet-stream"}"'
                f' src="data:{c.mimeType or ""};base64,{c.blob}"'
                f' title="{c.uri}"'
                " />"
            )
        case _ as unreachable:
            assert_never(unreachable)


def result_to_content(
    result: CallToolResult,
) -> list[ChatCompletionContentPartTextParam]:
    """Convert MCP tool call result to text params."""

    def _2text(block: ContentBlock):
        match block.type:
            case "text":
                text = block.text
            case "image":
                text = _image_to_md(block)
            case "resource":
                text = _resource_to_md(block)
            case "audio":
                text = f'<audio src="data:{block.mimeType};base64,{block.data}" />'
            case "resource_link":
                text = f"[{block.name}]({block.uri})"
            case _ as unreachable:
                assert_never(unreachable)
        return text

    return [{"text": _2text(block), "type": "text"} for block in result.content]


def result_to_message(
    tool_call_id: str,
    result: CallToolResult | BaseException,
) -> ChatCompletionToolMessageParam:
    """Convert MCP tool call result to OpenAI's message parameter."""
    return {
        "role": "tool",
        "tool_call_id": tool_call_id,
        "content": "".join(p["text"] for p in result_to_content(result))
        if isinstance(result, CallToolResult)
        else str(result),
    }
