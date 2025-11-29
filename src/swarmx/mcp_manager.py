"""MCP client related."""

import mimetypes
import os
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
    Tool,
)
from openai.types.chat import (
    ChatCompletionContentPartTextParam,
    ChatCompletionFunctionToolParam,
    ChatCompletionToolMessageParam,
)
from pygments.lexers import get_lexer_for_filename, get_lexer_for_mimetype
from pygments.util import ClassNotFound

from .types import MCPServer

mimetypes.add_type("text/markdown", ".md")


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
