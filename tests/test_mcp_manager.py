import json
import sys
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock

import pytest
from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters
from mcp.types import (
    CallToolResult,
    Tool,
)

from swarmx.mcp_manager import MCPManager

pytestmark = pytest.mark.anyio


async def test_client_registry_add_server_already_exists():
    """Test ClientRegistry add_server when server already exists."""
    registry = MCPManager()

    server_params = StdioServerParameters(
        command=sys.executable, args=["-m", "mcp_server_time"]
    )
    await registry.add_server("time", server_params)

    # Should return early without adding
    await registry.add_server(
        "time",
        StdioServerParameters(command=sys.executable, args=["-m", "not_a_server"]),
    )

    # Should still have the original mock
    assert isinstance(registry.mcp_clients["time"], ClientSession)


async def test_client_registry_add_stdio_server():
    """Test ClientRegistry add_server with stdio server."""
    registry = MCPManager()

    server_params = StdioServerParameters(
        command=sys.executable, args=["-m", "mcp_server_time"]
    )
    await registry.add_server("time", server_params)

    assert "time" in registry.mcp_clients
    assert any(tool.name == "get_current_time" for tool in registry._tools["time"])
    assert any(
        tool["function"]["name"] == "get_current_time" for tool in registry.tools
    )


async def test_client_registry_call_tool_success():
    """Test ClientRegistry call_tool success."""
    registry = MCPManager()

    # Mock MCP client
    server_params = StdioServerParameters(
        command=sys.executable, args=["-m", "mcp_server_time"]
    )
    await registry.add_server("time", server_params)
    result = await registry.call_tool("get_current_time", {"timezone": "UTC"})
    assert (text_part := result.content[0]).type == "text"
    obj = json.loads(text_part.text)
    assert (
        obj["timezone"] == "UTC"
        and (dt := datetime.fromisoformat(obj["datetime"])).tzinfo == UTC
    )
    assert datetime.now(tz=UTC) - dt < timedelta(seconds=1)


async def test_client_registry_call_tool_with_timeout_and_callback():
    """Test ClientRegistry call_tool with timeout and progress callback."""
    registry = MCPManager()

    # Mock MCP client
    mock_client = AsyncMock()
    mock_client.call_tool = AsyncMock()
    mock_result = CallToolResult.model_validate(
        {"content": [{"type": "text", "text": "Tool result"}]}
    )
    mock_client.call_tool.return_value = mock_result

    registry.mcp_clients["test_server"] = mock_client
    registry._tools["test_server"] = [
        Tool(name="test_tool", inputSchema={"type": "object", "properties": {}})
    ]

    callback = AsyncMock()

    result = await registry.call_tool(
        "test_tool",
        {"arg": "value"},
        read_timeout_seconds=timedelta(seconds=30),
        progress_callback=callback,
    )

    assert result == mock_result
    mock_client.call_tool.assert_called_once_with(
        "test_tool", {"arg": "value"}, timedelta(seconds=30), callback, meta=None
    )


async def test_client_registry_tools_property():
    """Test ClientRegistry tools property."""
    registry = MCPManager()

    tool1 = Tool(
        name="tool1",
        description="Tool 1",
        inputSchema={"type": "object", "properties": {}},
    )
    tool2 = Tool(
        name="tool2",
        description="Tool 2",
        inputSchema={"type": "object", "properties": {}},
    )

    # Set tools directly on the registry's internal storage with correct key format
    registry._tools["server1"] = [tool1]
    registry._tools["server2"] = [tool2]

    tools = registry.tools

    assert tools[0]["function"]["name"] == "tool1"
    assert tools[1]["function"]["name"] == "tool2"


async def test_client_registry_tools_include_description():
    registry = MCPManager()
    registry._tools["server"] = [
        Tool(
            name="described",
            description="Helpful tool",
            inputSchema={"type": "object", "properties": {}},
        )
    ]
    tools = registry.tools
    assert tools[0]["function"].get("description") == "Helpful tool"


async def test_client_registry_parse_name_errors():
    registry = MCPManager()
    with pytest.raises(KeyError):
        registry._parse_name("invalid")
    registry.mcp_clients["server"] = AsyncMock()
    registry._tools["server"] = []
    with pytest.raises(KeyError):
        registry._parse_name("missing")


async def test_client_registry_call_tool_missing_server():
    registry = MCPManager()
    with pytest.raises(KeyError):
        await registry.call_tool("tool", {})
