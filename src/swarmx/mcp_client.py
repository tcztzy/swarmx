"""MCP client related."""

import asyncio
import json
import logging
import mimetypes
import os
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any, Iterable, assert_never

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
    ChatCompletionChunk,
    ChatCompletionContentPartTextParam,
    ChatCompletionMessageParam,
    ChatCompletionMessageToolCallParam,
    ChatCompletionToolMessageParam,
    ChatCompletionToolParam,
)
from pygments.lexers import get_lexer_for_filename, get_lexer_for_mimetype
from pygments.util import ClassNotFound

from .types import MarkdownFlavor, MCPServer
from .utils import now

mimetypes.add_type("text/markdown", ".md")
logging.basicConfig(filename=".swarmx.log", level=logging.INFO)
logger = logging.getLogger(__name__)


# SECTION 4: Tool registry
@dataclass
class ClientRegistry:
    """Registry for tools."""

    exit_stack: AsyncExitStack = field(default_factory=AsyncExitStack)
    mcp_clients: dict[str, ClientSession] = field(default_factory=dict)
    _tools: dict[tuple[str, str], Tool] = field(default_factory=dict)
    _tool_to_server: dict[str, str] = field(default_factory=dict)

    @property
    def tools(self) -> list[ChatCompletionToolParam]:
        """Return all tools, both local and MCP."""
        return [
            ChatCompletionToolParam(
                type="function",
                function={
                    "name": tool.name,
                    "description": tool.description or "",
                    "parameters": tool.inputSchema,
                },
            )
            for tool in self._tools.values()
        ]

    async def call_tool(
        self,
        name: str,
        arguments: dict[str, Any],
        read_timeout_seconds: timedelta | None = None,
        progress_callback: ProgressFnT | None = None,
    ):
        """Call a tool.

        Args:
            name: The name of the tool
            arguments: The arguments to pass to the tool
            read_timeout_seconds: The read timeout for the tool call
            progress_callback: The progress callback for the tool call

        """
        server_name = self._tool_to_server.get(name)
        if server_name is None:
            raise ValueError(f"Tool {name} not found")
        return await self.mcp_clients[server_name].call_tool(
            name, arguments, read_timeout_seconds, progress_callback
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
        for tool in (await client.list_tools()).tools:
            self._tool_to_server[tool.name] = name
            self._tools[name, tool.name] = tool

    async def close(self):
        """Close all clients."""
        await self.exit_stack.aclose()


CLIENT_REGISTRY = ClientRegistry()


def _image_to_md(
    content: ImageContent,
) -> str:
    alt = (content.meta or {}).get("alt", "")
    return f"![{alt}](data:{content.mimeType};base64,{content.data})"


def _resource_to_md(
    resource: EmbeddedResource,
    flavor: MarkdownFlavor = "gfm",
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
            lang = get_lang(c)
            match flavor:
                case "gfm":
                    return f'\n```{lang} title="{c.uri}"\n{c.text}\n```\n'
                case "mystmd":
                    return f"\n```{{code}} {lang}\n:filename: {c.uri}\n{c.text}\n```\n"
                case _:
                    assert_never(flavor)
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
    flavor: MarkdownFlavor = "gfm",
) -> list[ChatCompletionContentPartTextParam]:
    """Convert MCP tool call result to text params."""

    def _2text(block: ContentBlock):
        match block.type:
            case "text":
                text = block.text
            case "image":
                text = _image_to_md(block)
            case "resource":
                text = _resource_to_md(block, flavor)
            case "audio":
                text = f'<audio src="data:{block.mimeType};base64,{block.data}" />'
            case "resource_link":
                text = f"[{block.name}](blob:{block.uri})"
            case _ as unreachable:
                assert_never(unreachable)
        return text

    return [{"text": _2text(block), "type": "text"} for block in result.content]


def result_to_message(
    result: CallToolResult,
    tool_call_id: str,
) -> ChatCompletionToolMessageParam:
    """Convert MCP tool call result to OpenAI's message parameter."""
    return {
        "role": "tool",
        "content": result_to_content(result),
        "tool_call_id": tool_call_id,
    }


def result_to_chunk(
    tool_call: ChatCompletionMessageToolCallParam,
    result: CallToolResult | BaseException,
) -> ChatCompletionChunk:
    """Convert MCP tool call result to OpenAI's chunk parameter."""
    content: str = ""
    if isinstance(result, CallToolResult):
        content = "".join(p["text"] for p in result_to_content(result))
    else:
        content = str(result)

    return ChatCompletionChunk.model_validate(
        {
            "id": tool_call["id"],
            "object": "chat.completion.chunk",
            "created": now(),
            "model": tool_call["function"]["name"],
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": content, "role": "tool"},
                    "finish_reason": "stop",
                }
            ],
        }
    )


async def exec_tool_calls(tool_calls: Iterable[ChatCompletionMessageToolCallParam]):
    """Execute tool calls and return the messages."""
    messages: list[ChatCompletionMessageParam] = []

    async def _inner(
        tool_call: ChatCompletionMessageToolCallParam,
    ) -> tuple[ChatCompletionMessageParam, ChatCompletionChunk]:
        try:
            result = await CLIENT_REGISTRY.call_tool(
                tool_call["function"]["name"],
                json.loads(tool_call["function"]["arguments"]),
            )
            return result_to_message(result, tool_call["id"]), result_to_chunk(
                tool_call, result
            )
        except Exception as e:
            return {
                "role": "tool",
                "content": str(e),
                "tool_call_id": tool_call["id"],
            }, result_to_chunk(tool_call, e)

    async with asyncio.TaskGroup() as tg:
        tasks = [tg.create_task(_inner(tool_call)) for tool_call in tool_calls]
        for task in asyncio.as_completed(tasks):
            message, chunk = await task
            yield chunk
            messages.append(message)
    yield messages
