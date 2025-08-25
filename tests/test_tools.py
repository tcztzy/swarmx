import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from mcp.client.stdio import StdioServerParameters

from swarmx import ClientRegistry

pytestmark = pytest.mark.anyio


@pytest.fixture
def tool_registry():
    registry = ClientRegistry()
    yield registry
    asyncio.run(registry.close())


async def test_call_tool_not_found(tool_registry):
    # Attempt to call a tool that doesn't exist
    with pytest.raises(ValueError, match="Invalid tool name, expected <server>/<tool>"):
        await tool_registry.call_tool("nonexistent_tool", {})
    with pytest.raises(KeyError, match="Server nonexistent_server not found"):
        await tool_registry.call_tool("nonexistent_server/nonexistent_tool", {})
    tool_registry.mcp_clients["test_server"] = AsyncMock()
    tool_registry._tools["test_server"] = [MagicMock(name="test_tool")]
    with pytest.raises(KeyError, match="Tool nonexistent_tool not found"):
        await tool_registry.call_tool("test_server/nonexistent_tool", {})


async def test_add_mcp_server():
    with (
        patch("swarmx.mcp_client.stdio_client") as mock_stdio_client,
        patch("swarmx.mcp_client.ClientSession") as mock_client_session,
    ):
        # Setup mocks
        mock_client = AsyncMock()
        mock_client.initialize = AsyncMock()
        mock_client.list_tools = AsyncMock()

        tool = MagicMock()
        tool.name = "test_tool"
        tool.model_dump.return_value = {
            "name": "test_tool",
            "inputSchema": {"type": "object", "properties": {}},
        }

        mock_client.list_tools.return_value.tools = [tool]
        mock_client_session.return_value.__aenter__.return_value = mock_client

        mock_read_stream = AsyncMock()
        mock_write_stream = AsyncMock()
        mock_stdio_client.return_value.__aenter__.return_value = (
            mock_read_stream,
            mock_write_stream,
        )

        # Create a registry and add an MCP server
        registry = ClientRegistry()
        await registry.add_server(
            "test_server", StdioServerParameters(command="echo", args=["test"])
        )

        # Verify that the client was initialized and tools were fetched
        mock_client.initialize.assert_called_once()
        mock_client.list_tools.assert_called_once()

        # Verify that the tool was added correctly
        assert "test_server" in registry._tools and any(
            tool.name == "test_tool" for tool in registry._tools["test_server"]
        )

        # Cleanup
        await registry.close()
