from datetime import timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from mcp.client.stdio import StdioServerParameters
from mcp.types import CallToolResult, Tool
from openai.types.chat import ChatCompletionMessageToolCallParam

from swarmx.mcp_client import ClientRegistry, exec_tool_calls, result_to_message
from swarmx.types import SSEServer

pytestmark = pytest.mark.anyio


async def test_result_to_message_text():
    """Test converting a text result to a message."""

    result = CallToolResult.model_validate(
        {"content": [{"type": "text", "text": "Hello, world!"}]}
    )
    message = result_to_message(result, "test_tool_call_id")
    assert message["role"] == "tool"
    assert message["tool_call_id"] == "test_tool_call_id"
    assert not isinstance(message["content"], str)
    parts = list(message["content"])
    assert parts[0]["text"] == "Hello, world!"


async def test_result_to_message_image():
    """Test converting an image result to a message."""

    result = CallToolResult.model_validate(
        {"content": [{"type": "image", "data": "base64data", "mimeType": "image/png"}]}
    )
    message = result_to_message(result, "test_tool_call_id")
    assert message["role"] == "tool"
    assert message["tool_call_id"] == "test_tool_call_id"
    assert not isinstance(message["content"], str)
    parts = list(message["content"])
    assert parts[0]["text"] == "![](data:image/png;base64,base64data)"


async def test_result_to_message_resource():
    """Test converting a resource result to a message."""

    result = CallToolResult.model_validate(
        {
            "content": [
                {
                    "type": "resource",
                    "resource": {
                        "type": "text",
                        "uri": "file:///file.txt",
                        "text": "File content",
                        "mimeType": "text/plain",
                    },
                }
            ]
        }
    )
    message = result_to_message(result, "test_tool_call_id")
    assert message["role"] == "tool"
    assert message["tool_call_id"] == "test_tool_call_id"
    assert not isinstance(message["content"], str)
    parts = list(message["content"])
    assert parts[0]["text"] == '\n```text title="file:///file.txt"\nFile content\n```\n'

    result = CallToolResult.model_validate(
        {
            "content": [
                {
                    "type": "resource",
                    "resource": {
                        "type": "text",
                        "uri": "file:///main.rs",
                        "text": 'fn main() {\n    println!("Hello, world!");\n}',
                        "mimeType": "text/rust",
                    },
                }
            ]
        }
    )
    message = result_to_message(result, "test_tool_call_id")
    assert message["role"] == "tool"
    assert message["tool_call_id"] == "test_tool_call_id"
    assert not isinstance(message["content"], str)
    parts = list(message["content"])
    assert (
        parts[0]["text"]
        == '\n```rust title="file:///main.rs"\nfn main() {\n    println!("Hello, world!");\n}\n```\n'
    )


async def test_result_to_message_audio():
    """Test converting an audio result to a message."""

    result = CallToolResult.model_validate(
        {"content": [{"type": "audio", "data": "base64data", "mimeType": "audio/wav"}]}
    )
    message = result_to_message(result, "test_tool_call_id")
    assert message["role"] == "tool"
    assert message["tool_call_id"] == "test_tool_call_id"
    assert not isinstance(message["content"], str)
    parts = list(message["content"])
    assert parts[0]["text"] == '<audio src="data:audio/wav;base64,base64data" />'


async def test_result_to_message_mixed_content():
    """Test converting a result with mixed content types to a message."""
    result = CallToolResult.model_validate(
        {
            "content": [
                {"type": "text", "text": "Hello, world!"},
                {"type": "image", "data": "base64data", "mimeType": "image/png"},
                {
                    "type": "resource",
                    "resource": {"uri": "file://file.txt", "text": "File content"},
                },
            ]
        }
    )
    message = result_to_message(result, "test_tool_call_id")
    assert message["role"] == "tool"
    assert message["tool_call_id"] == "test_tool_call_id"
    assert not isinstance(message["content"], str)
    parts = list(message["content"])
    assert len(parts) == 3
    assert parts[0]["text"] == "Hello, world!"
    assert parts[1]["text"] == "![](data:image/png;base64,base64data)"
    assert "File content" in parts[2]["text"]


async def test_client_registry_call_tool_not_found():
    """Test ClientRegistry call_tool with tool not found."""
    registry = ClientRegistry()

    with pytest.raises(ValueError, match="Tool nonexistent_tool not found"):
        await registry.call_tool("nonexistent_tool", {})


async def test_client_registry_add_server_already_exists():
    """Test ClientRegistry add_server when server already exists."""
    registry = ClientRegistry()

    # Mock that server already exists
    registry.mcp_clients["test_server"] = MagicMock()

    server_params = StdioServerParameters(command="python", args=["-m", "test_server"])

    # Should return early without adding
    await registry.add_server("test_server", server_params)

    # Should still have the original mock
    assert isinstance(registry.mcp_clients["test_server"], MagicMock)


async def test_client_registry_add_stdio_server():
    """Test ClientRegistry add_server with stdio server."""
    registry = ClientRegistry()

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

        server_params = StdioServerParameters(
            command="python", args=["-m", "test_server"]
        )

        await registry.add_server("test_server", server_params)

        assert "test_server" in registry.mcp_clients
        assert "test_tool" in registry._tool_to_server
        assert registry._tool_to_server["test_tool"] == "test_server"


async def test_client_registry_add_sse_server():
    """Test ClientRegistry add_server with SSE server."""
    registry = ClientRegistry()

    with (
        patch("swarmx.mcp_client.sse_client") as mock_sse_client,
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
        mock_sse_client.return_value.__aenter__.return_value = (
            mock_read_stream,
            mock_write_stream,
        )

        server_params = SSEServer(
            type="sse",
            url="http://localhost:8000/sse",
            headers={"Authorization": "Bearer token"},
        )

        await registry.add_server("test_server", server_params)

        assert "test_server" in registry.mcp_clients
        assert "test_tool" in registry._tool_to_server
        assert registry._tool_to_server["test_tool"] == "test_server"


async def test_client_registry_call_tool_success():
    """Test ClientRegistry call_tool success."""
    registry = ClientRegistry()

    # Mock MCP client
    mock_client = AsyncMock()
    mock_client.call_tool = AsyncMock()
    mock_result = CallToolResult.model_validate(
        {"content": [{"type": "text", "text": "Tool result"}]}
    )
    mock_client.call_tool.return_value = mock_result

    registry.mcp_clients["test_server"] = mock_client
    registry._tool_to_server["test_tool"] = "test_server"

    result = await registry.call_tool("test_tool", {"arg": "value"})

    assert result == mock_result
    mock_client.call_tool.assert_called_once_with(
        "test_tool", {"arg": "value"}, None, None
    )


async def test_client_registry_call_tool_with_timeout_and_callback():
    """Test ClientRegistry call_tool with timeout and progress callback."""
    registry = ClientRegistry()

    # Mock MCP client
    mock_client = AsyncMock()
    mock_client.call_tool = AsyncMock()
    mock_result = CallToolResult.model_validate(
        {"content": [{"type": "text", "text": "Tool result"}]}
    )
    mock_client.call_tool.return_value = mock_result

    registry.mcp_clients["test_server"] = mock_client
    registry._tool_to_server["test_tool"] = "test_server"

    callback = AsyncMock()

    result = await registry.call_tool(
        "test_tool",
        {"arg": "value"},
        read_timeout_seconds=timedelta(seconds=30),
        progress_callback=callback,
    )

    assert result == mock_result
    mock_client.call_tool.assert_called_once_with(
        "test_tool", {"arg": "value"}, timedelta(seconds=30), callback
    )


async def test_client_registry_tools_property():
    """Test ClientRegistry tools property."""
    registry = ClientRegistry()

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
    registry._tools[("server1", "tool1")] = tool1
    registry._tools[("server2", "tool2")] = tool2

    tools = registry.tools

    assert len(tools) == 2
    assert any(tool["function"]["name"] == "tool1" for tool in tools)
    assert any(tool["function"]["name"] == "tool2" for tool in tools)


async def test_client_registry_close():
    """Test ClientRegistry close method."""
    registry = ClientRegistry()

    # Mock exit stack
    mock_exit_stack = AsyncMock()
    registry.exit_stack = mock_exit_stack

    await registry.close()

    mock_exit_stack.aclose.assert_called_once()


async def test_exec_tool_calls():
    """Test exec_tool_calls function."""
    tool_calls: list[ChatCompletionMessageToolCallParam] = [
        {
            "id": "call_123",
            "type": "function",
            "function": {"name": "test_tool", "arguments": '{"arg": "value"}'},
        }
    ]

    with patch("swarmx.mcp_client.CLIENT_REGISTRY.call_tool") as mock_call_tool:
        mock_result = CallToolResult.model_validate(
            {"content": [{"type": "text", "text": "Tool result"}]}
        )
        mock_call_tool.return_value = mock_result

        results = []
        async for chunk in exec_tool_calls(tool_calls):
            results.append(chunk)

        # exec_tool_calls yields both chunks and messages
        assert len(results) == 2
        # Last result should be the list of messages
        assert isinstance(results[-1], list)
        assert len(results[-1]) == 1
        assert results[-1][0]["role"] == "tool"
        assert results[-1][0]["tool_call_id"] == "call_123"


async def test_exec_tool_calls_with_error():
    """Test exec_tool_calls function with tool execution error."""
    tool_calls: list[ChatCompletionMessageToolCallParam] = [
        {
            "id": "call_123",
            "type": "function",
            "function": {"name": "failing_tool", "arguments": '{"arg": "value"}'},
        }
    ]

    with patch("swarmx.mcp_client.CLIENT_REGISTRY.call_tool") as mock_call_tool:
        mock_call_tool.side_effect = Exception("Tool execution failed")

        results = []
        async for chunk in exec_tool_calls(tool_calls):
            results.append(chunk)

        assert len(results) == 2
        assert isinstance(results[-1], list)
        assert len(results[-1]) == 1
        assert results[-1][0]["role"] == "tool"
        assert results[-1][0]["tool_call_id"] == "call_123"
        assert "Tool execution failed" in results[-1][0]["content"]


async def test_exec_tool_calls_invalid_json():
    """Test exec_tool_calls function with invalid JSON arguments."""
    tool_calls: list[ChatCompletionMessageToolCallParam] = [
        {
            "id": "call_123",
            "type": "function",
            "function": {"name": "test_tool", "arguments": "invalid json"},
        }
    ]

    results = []
    async for chunk in exec_tool_calls(tool_calls):
        results.append(chunk)

    assert len(results) == 2
    assert isinstance(results[-1], list)
    assert len(results[-1]) == 1
    assert results[-1][0]["role"] == "tool"
    assert results[-1][0]["tool_call_id"] == "call_123"
    assert "Expecting value" in results[-1][0]["content"]


async def test_result_to_message_with_structured_output():
    """Test result_to_message with structured output."""
    result = CallToolResult.model_validate(
        {
            "content": [{"type": "text", "text": "Regular content"}],
            "structuredOutput": {"key": "value", "number": 42},
        }
    )

    message = result_to_message(result, "test_tool_call_id")

    assert message["role"] == "tool"
    assert message["tool_call_id"] == "test_tool_call_id"
    assert not isinstance(message["content"], str)
    parts = list(message["content"])
    assert len(parts) == 1
    assert parts[0]["text"] == "Regular content"
