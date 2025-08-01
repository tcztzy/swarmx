"""SwarmX types."""

from typing import Literal

from mcp.client.stdio import StdioServerParameters
from pydantic import BaseModel


class SSEServer(BaseModel):
    """SSE server configuration.

    This is used to configure the MCP server for the agent.

    """

    type: Literal["sse"]
    url: str
    headers: dict[str, str] | None = None


MCPServer = StdioServerParameters | SSEServer
