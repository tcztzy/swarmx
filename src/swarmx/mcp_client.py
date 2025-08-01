"""MCP client related."""

import asyncio
import json
import logging
import mimetypes
import os
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any, Iterable

from mcp import ClientSession, StdioServerParameters
from mcp.client.sse import sse_client
from mcp.client.stdio import stdio_client
from mcp.shared.session import ProgressFnT
from mcp.types import (
    BlobResourceContents,
    CallToolResult,
    EmbeddedResource,
    ImageContent,
    ResourceContents,
    TextResourceContents,
    Tool,
)
from openai.types.chat import (
    ChatCompletionContentPartImageParam,
    ChatCompletionContentPartParam,
    ChatCompletionContentPartTextParam,
    ChatCompletionMessageParam,
    ChatCompletionMessageToolCallParam,
    ChatCompletionToolParam,
)
from openai.types.chat.chat_completion_content_part_param import File
from pygments.lexers import get_lexer_for_filename, get_lexer_for_mimetype
from pygments.util import ClassNotFound

from .types import MCPServer

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


def _image_content_to_url(
    content: ImageContent,
) -> ChatCompletionContentPartImageParam:
    return {
        "type": "image_url",
        "image_url": {
            "url": f"data:{content.mimeType};base64,{content.data}",
        },
    }


def _resource_to_file(
    resource: EmbeddedResource,
) -> ChatCompletionContentPartTextParam | File:
    def get_filename(c: ResourceContents) -> str:
        if c.uri.path is not None:
            filename = os.path.basename(c.uri.path)
        elif c.uri.host is not None:
            filename = c.uri.host
        elif c.mimeType is not None:
            ext = mimetypes.guess_extension(c.mimeType)
            if ext is None:
                raise ValueError(
                    f"Cannot determine filename for resource. mimeType={c.mimeType}"
                )
            filename = f"file{ext}"
        else:
            raise ValueError("Cannot determine filename for resource.")
        return filename

    filename = get_filename(resource.resource)
    match resource.resource:
        case TextResourceContents() as c:
            if c.mimeType is None:
                try:
                    lexer = get_lexer_for_filename(filename)
                except ClassNotFound:
                    lexer = None
            else:
                try:
                    lexer = get_lexer_for_mimetype(c.mimeType)
                except ClassNotFound:
                    lexer = None
            lang = lexer.aliases[0] if lexer else "text"
            return {
                "type": "text",
                "text": f'\n```{lang} title="{c.uri}"\n{c.text}\n```\n',
            }
        case BlobResourceContents() as c:
            return {
                "type": "file",
                "file": {
                    "file_data": c.blob,
                    "filename": filename,
                },
            }
        case _:
            raise ValueError("Unsupported resource type.")


def result_to_message(
    result: CallToolResult,
    tool_call_id: str,
) -> ChatCompletionMessageParam:
    """Convert MCP tool call result to OpenAI's message parameter."""
    content: list[ChatCompletionContentPartParam] = []
    for chunk in result.content:
        match chunk.type:
            case "text":
                content.append({"type": "text", "text": chunk.text})
            case "image":
                content.append(_image_content_to_url(chunk))
            case "resource":
                content.append(_resource_to_file(chunk))
            case "audio":
                match chunk.mimeType:
                    case (
                        "audio/vnd.wav"
                        | "audio/vnd.wave"
                        | "audio/wave"
                        | "audio/x-pn-wav"
                        | "audio/x-wav"
                        | "audio/wav"
                    ):
                        format = "wav"
                    case "audio/mpeg":
                        format = "mp3"
                    case _:
                        raise ValueError(f"Unsupported audio format: {chunk.mimeType}")
                content.append(
                    {
                        "type": "input_audio",
                        "input_audio": {"data": chunk.data, "format": format},
                    }
                )
            case "resource_link":
                content.append({"text": str(chunk.uri), "type": "text"})
    if all(part for part in content):
        return {"role": "tool", "content": content, "tool_call_id": tool_call_id}  # type: ignore
    return {"role": "user", "content": content, "name": tool_call_id}


async def exec_tool_calls(tool_calls: Iterable[ChatCompletionMessageToolCallParam]):
    """Execute tool calls and return the messages."""
    messages: list[ChatCompletionMessageParam] = []
    tasks = [
        asyncio.create_task(
            CLIENT_REGISTRY.call_tool(
                tool_call["function"]["name"],
                json.loads(tool_call["function"]["arguments"]),
            )
        )
        for tool_call in tool_calls
    ]
    results: list[CallToolResult | BaseException] = await asyncio.gather(
        *tasks, return_exceptions=True
    )
    for tool_call, result in zip(tool_calls, results):
        if isinstance(result, BaseException):
            logger.error(f"{tool_call['id']} failed:", exc_info=result)
            messages.append(
                {
                    "role": "tool",
                    "content": str(result),
                    "tool_call_id": tool_call["id"],
                }
            )
        else:
            logger.info(result.model_dump_json(exclude_unset=True))
            messages.append(result_to_message(result, tool_call["id"]))
    return messages
