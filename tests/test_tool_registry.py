import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from mcp.client.stdio import StdioServerParameters

from swarmx import (
    Agent,
    Result,
    ToolRegistry,
    does_function_need_context,
)

pytestmark = pytest.mark.anyio


def test_does_function_need_context():
    # Test with function that needs context variables
    def func_with_ctx(context_variables: dict):
        return "Hello"

    assert does_function_need_context(func_with_ctx) is True

    # Test with function that doesn't need context variables
    def func_without_ctx(name: str):
        return f"Hello {name}"

    assert does_function_need_context(func_without_ctx) is False


@pytest.fixture
def tool_registry():
    registry = ToolRegistry()
    yield registry
    asyncio.run(registry.close())


def test_tool_registry_properties(tool_registry):
    # Test the tools property
    assert tool_registry.tools == []

    # Add a function
    def test_func():
        return "Test"

    tool_registry.add_function(test_func)

    # Check that the function was added correctly
    assert len(tool_registry.tools) == 1
    assert tool_registry.tools[0]["function"]["name"] == "test_func"


@pytest.mark.anyio
async def test_add_function_and_call_tool(tool_registry):
    # Define a test function
    def test_func(name: str):
        return f"Hello, {name}!"

    # Add the function to the registry
    tool_registry.add_function(test_func)

    # Call the tool
    result = await tool_registry.call_tool("test_func", {"name": "World"})

    # Check the result
    assert isinstance(result, Result)
    assert result.value == "Hello, World!"


@pytest.mark.anyio
async def test_call_tool_with_context(tool_registry):
    # Define a test function that needs context variables
    def test_func_with_ctx(name: str, context_variables: dict):
        return f"Hello, {name}! User ID: {context_variables.get('user_id')}"

    # Add the function to the registry
    tool_registry.add_function(test_func_with_ctx)

    # Call the tool with context variables
    result = await tool_registry.call_tool(
        "test_func_with_ctx", {"name": "World"}, {"user_id": "12345"}
    )

    # Check the result
    assert isinstance(result, Result)
    assert result.value == "Hello, World! User ID: 12345"


@pytest.mark.anyio
async def test_call_tool_async_function(tool_registry):
    # Define an async test function
    async def async_test_func(name: str):
        await asyncio.sleep(0.01)  # Simulate async operation
        return f"Hello async, {name}!"

    # Add the function to the registry
    tool_registry.add_function(async_test_func)

    # Call the tool
    result = await tool_registry.call_tool("async_test_func", {"name": "World"})

    # Check the result
    assert isinstance(result, Result)
    assert result.value == "Hello async, World!"


@pytest.mark.anyio
async def test_call_tool_agent_return(tool_registry):
    # Define a function that returns an Agent
    def agent_return_func():
        return Agent(name="TestAgent", instructions="Test instructions")

    # Add the function to the registry
    tool_registry.add_function(agent_return_func)

    # Call the tool
    result = await tool_registry.call_tool("agent_return_func", {})

    # Check the result
    assert isinstance(result, Result)
    assert isinstance(result.agent, Agent)
    assert result.agent.name == "TestAgent"


async def test_call_tool_not_found(tool_registry):
    # Attempt to call a tool that doesn't exist
    with pytest.raises(ValueError, match="Tool nonexistent_tool not found"):
        await tool_registry.call_tool("nonexistent_tool", {})


async def test_add_mcp_server():
    with (
        patch("swarmx.stdio_client") as mock_stdio_client,
        patch("swarmx.ClientSession") as mock_client_session,
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
        registry = ToolRegistry()
        await registry.add_mcp_server(
            "test_server", StdioServerParameters(command="echo", args=["test"])
        )

        # Verify that the client was initialized and tools were fetched
        mock_client.initialize.assert_called_once()
        mock_client.list_tools.assert_called_once()

        # Verify that the tool was added correctly
        assert ("test_server", "test_tool") in registry.mcp_tools
        assert registry._tools["test_tool"] == "test_server"

        # Cleanup
        await registry.close()
