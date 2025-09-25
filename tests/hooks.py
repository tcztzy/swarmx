"""Test MCP server exposing hook tools."""

from typing import Any

from mcp.server.fastmcp import FastMCP

server = FastMCP(
    name="hooks",
    instructions="Test hook server for SwarmX",
)


@server.tool(
    name="record",
    description="Record hook invocation for tests",
    structured_output=True,
)
async def record_hook(
    messages: list[dict[str, Any]],
    context: dict[str, Any] | None = None,
    agent: dict[str, Any] | None = None,
    tool: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return structured content capturing hook details."""
    context = context or {}
    payload = {
        "messages_count": len(messages),
        "agent_name": (agent or {}).get("name"),
        "tool_name": (tool or {}).get("name") if tool else None,
    }
    counter = context.get("hook_counter", 0)
    return {
        "hook_counter": counter + 1,
        "last_hook": payload,
    }


if __name__ == "__main__":
    server.run()
