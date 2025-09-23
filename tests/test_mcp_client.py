import json
import sys
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock

import pytest
from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters
from mcp.types import (
    CallToolResult,
    EmbeddedResource,
    ImageContent,
    Tool,
)
from openai.types.chat import ChatCompletionMessageToolCallParam

from swarmx.mcp_client import (
    ClientRegistry,
    _image_to_md,
    _resource_to_md,
    result_to_chunk,
    result_to_content,
    result_to_message,
)

pytestmark = pytest.mark.anyio


async def test_client_registry_add_server_already_exists():
    """Test ClientRegistry add_server when server already exists."""
    registry = ClientRegistry()

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
    registry = ClientRegistry()

    server_params = StdioServerParameters(
        command=sys.executable, args=["-m", "mcp_server_time"]
    )
    await registry.add_server("time", server_params)

    assert "time" in registry.mcp_clients
    assert any(tool.name == "get_current_time" for tool in registry._tools["time"])
    assert any(
        tool["function"]["name"] == "time/get_current_time" for tool in registry.tools
    )


async def test_client_registry_call_tool_success():
    """Test ClientRegistry call_tool success."""
    registry = ClientRegistry()

    # Mock MCP client
    server_params = StdioServerParameters(
        command=sys.executable, args=["-m", "mcp_server_time"]
    )
    await registry.add_server("time", server_params)
    result = await registry.call_tool("time/get_current_time", {"timezone": "UTC"})
    assert (text_part := result.content[0]).type == "text"
    obj = json.loads(text_part.text)
    assert (
        obj["timezone"] == "UTC"
        and (dt := datetime.fromisoformat(obj["datetime"])).tzinfo == UTC
    )
    assert datetime.now(tz=UTC) - dt < timedelta(seconds=1)


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
    registry._tools["test_server"] = [
        Tool(name="test_tool", inputSchema={"type": "object", "properties": {}})
    ]

    callback = AsyncMock()

    result = await registry.call_tool(
        "test_server/test_tool",
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
    registry._tools["server1"] = [tool1]
    registry._tools["server2"] = [tool2]

    tools = registry.tools

    assert tools[0]["function"]["name"] == "server1/tool1"
    assert tools[1]["function"]["name"] == "server2/tool2"


async def test_client_registry_close():
    """Test ClientRegistry close method."""
    registry = ClientRegistry()

    # Mock exit stack
    mock_exit_stack = AsyncMock()
    registry.exit_stack = mock_exit_stack

    await registry.close()

    mock_exit_stack.aclose.assert_called_once()


async def test_resource_to_md_filename_from_host():
    """Test _resource_to_md filename determination from host."""
    # Create a resource with host but no path
    resource = EmbeddedResource.model_validate(
        {
            "type": "resource",
            "resource": {
                "uri": "example://example.txt",
                "text": "Test content",
            },
        }
    )

    result = _resource_to_md(resource)
    assert "example.txt" in result
    assert "Test content" in result


async def test_resource_to_md_filename_from_mimetype():
    """Test _resource_to_md filename determination from mimeType."""
    resource = EmbeddedResource.model_validate(
        {
            "type": "resource",
            "resource": {
                "uri": "data://",
                "text": "Test content",
                "mimeType": "text/plain",
            },
        }
    )

    result = _resource_to_md(resource)
    assert "Test content" in result


async def test_resource_to_md_filename_error():
    """Test _resource_to_md filename determination error."""

    # Create a resource with no path, host, or mimeType
    resource = EmbeddedResource.model_validate(
        {"type": "resource", "resource": {"uri": "data://", "text": "Test content"}}
    )

    with pytest.raises(ValueError, match="Cannot determine filename for resource"):
        _resource_to_md(resource)


async def test_resource_to_md_lexer_not_found():
    """Test _resource_to_md with lexer not found"""
    resource = EmbeddedResource.model_validate(
        {
            "type": "resource",
            "resource": {
                "uri": "file:///test.unknown",
                "text": "Test content",
                "mimeType": "application/unknown",
            },
        }
    )

    result = _resource_to_md(resource)
    # Should default to "text" when lexer is not found
    assert "```text" in result
    assert "Test content" in result


async def test_resource_to_md_mystmd_flavor():
    """Test _resource_to_md with mystmd flavor"""
    resource = EmbeddedResource.model_validate(
        {
            "type": "resource",
            "resource": {
                "uri": "file:///test.py",
                "text": "print('hello')",
                "mimeType": "text/python",
            },
        }
    )

    result = _resource_to_md(resource, flavor="mystmd")
    assert "```{code}" in result
    assert ":filename:" in result
    assert "print('hello')" in result


async def test_resource_to_md_blob_content():
    """Test _resource_to_md with blob content."""
    resource = EmbeddedResource.model_validate(
        {
            "type": "resource",
            "resource": {
                "uri": "file:///test.png",
                "blob": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg==",
                "mimeType": "image/png",
            },
        }
    )

    result = _resource_to_md(resource)
    assert "<embed" in result
    assert 'type="image/png"' in result
    assert "data:image/png;base64," in result
    assert 'title="file:///test.png"' in result


async def test_result_to_content_audio():
    """Test result_to_content with audio content."""

    result = CallToolResult.model_validate(
        {
            "content": [
                {"type": "audio", "data": "base64audiodata", "mimeType": "audio/wav"}
            ]
        }
    )

    content = result_to_content(result)
    assert len(content) == 1
    assert content[0]["type"] == "text"
    assert '<audio src="data:audio/wav;base64,base64audiodata"' in content[0]["text"]


async def test_result_to_content_resource_link():
    """Test result_to_content with resource_link content."""

    result = CallToolResult.model_validate(
        {
            "content": [
                {
                    "type": "resource_link",
                    "name": "Test Resource",
                    "uri": "file:///test.txt",
                }
            ]
        }
    )

    content = result_to_content(result)
    assert len(content) == 1
    assert content[0]["type"] == "text"
    assert "[Test Resource](blob:file:///test.txt)" in content[0]["text"]


async def test_client_registry_tools_include_description():
    registry = ClientRegistry()
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
    registry = ClientRegistry()
    with pytest.raises(ValueError):
        registry._parse_name("invalid")
    registry.mcp_clients["server"] = AsyncMock()
    registry._tools["server"] = []
    with pytest.raises(KeyError):
        registry._parse_name("server/missing")


async def test_client_registry_call_tool_missing_server():
    registry = ClientRegistry()
    with pytest.raises(KeyError):
        await registry.call_tool("ghost/tool", {})


async def test_image_to_md_and_result_to_content_image():
    image = ImageContent.model_validate(
        {
            "type": "image",
            "data": "base64",
            "mimeType": "image/png",
            "meta": {"alt": "alt"},
        }
    )
    md = _image_to_md(image)
    assert md.startswith("![") and "data:image/png" in md

    result = CallToolResult.model_validate({"content": [image.model_dump()]})
    parts = result_to_content(result)
    assert md in parts[0]["text"]


async def test_result_to_message_and_chunk_error_path():
    tool_call: ChatCompletionMessageToolCallParam = {
        "id": "tool",
        "type": "function",
        "function": {"name": "server/tool", "arguments": "{}"},
    }
    message = result_to_message(tool_call, Exception("boom"))
    assert message["content"] == "boom"

    chunk = result_to_chunk(tool_call, Exception("boom"))
    assert chunk.choices[0].delta.content == "boom"
