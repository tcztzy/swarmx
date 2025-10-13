"""SwarmX types."""

from typing import Any, Literal, TypeAlias

from mcp.client.stdio import StdioServerParameters
from openai.types.chat import (
    ChatCompletionAssistantMessageParam,
    ChatCompletionMessageParam,
)
from pydantic import BaseModel


class SSEServer(BaseModel):
    """SSE server configuration.

    This is used to configure the MCP server for the agent.

    """

    type: Literal["sse"]
    url: str
    headers: dict[str, str] | None = None


GraphMode = Literal["locked", "handoff", "expand"]

MCPServer = StdioServerParameters | SSEServer
MarkdownFlavor = Literal["gfm", "mystmd"]


class AssistantMessage(ChatCompletionAssistantMessageParam, total=False):  # type: ignore
    """Extended assistant message."""

    parsed: Any


MessageParam: TypeAlias = ChatCompletionMessageParam | AssistantMessage
