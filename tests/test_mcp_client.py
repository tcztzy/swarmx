import pytest
from mcp.types import CallToolResult

from swarmx.mcp_client import result_to_message

pytestmark = pytest.mark.anyio


async def test_result_to_message_text():
    """Test converting a text result to a message."""

    result = CallToolResult.model_validate(
        {"content": [{"type": "text", "text": "Hello, world!"}]}
    )
    message = result_to_message(result, "test_tool_call_id")
    assert message["role"] == "tool"
    assert message["tool_call_id"] == "test_tool_call_id"
    assert message["content"][0]["text"] == "Hello, world!"


async def test_result_to_message_image():
    """Test converting an image result to a message."""

    result = CallToolResult.model_validate(
        {"content": [{"type": "image", "data": "base64data", "mimeType": "image/png"}]}
    )
    message = result_to_message(result, "test_tool_call_id")
    assert message["role"] == "tool"
    assert message["tool_call_id"] == "test_tool_call_id"
    assert message["content"][0]["text"] == "![](data:image/png;base64,base64data)"


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
    assert "File content" in message["content"][0]["text"]


async def test_result_to_message_audio():
    """Test converting an audio result to a message."""

    result = CallToolResult.model_validate(
        {"content": [{"type": "audio", "data": "base64data", "mimeType": "audio/wav"}]}
    )
    message = result_to_message(result, "test_tool_call_id")
    assert message["role"] == "tool"
    assert message["tool_call_id"] == "test_tool_call_id"
    assert (
        message["content"][0]["text"]
        == '<audio src="data:audio/wav;base64,base64data" />'
    )


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
    assert len(message["content"]) == 3
    assert message["content"][0]["text"] == "Hello, world!"
    assert message["content"][1]["text"] == "![](data:image/png;base64,base64data)"
    assert "File content" in message["content"][2]["text"]
