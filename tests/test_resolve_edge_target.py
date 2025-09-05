import json
from unittest.mock import patch

import pytest
from mcp.types import CallToolResult, ImageContent, TextContent

from swarmx.agent import Agent

pytestmark = pytest.mark.anyio


async def test_resolve_edge_target_node_exists():
    """Test _resolve_edge_target when target is an existing node."""
    node = Agent(name="existing_node")
    agent = Agent(
        name="test_agent",
        nodes={node},
    )

    result = await agent._resolve_edge_target("existing_node", {})
    assert result == {node}


async def test_resolve_edge_target_node_not_exists():
    """Test _resolve_edge_target when target is not an existing node."""
    agent = Agent(name="test_agent")

    # Mock CLIENT_REGISTRY.call_tool to raise KeyError (tool not found)
    # and mock evaluate to return empty list
    with (
        patch("swarmx.agent.CLIENT_REGISTRY.call_tool") as mock_call_tool,
        patch("swarmx.agent.evaluate") as mock_evaluate,
    ):
        mock_call_tool.side_effect = KeyError("Tool not found")
        mock_evaluate.side_effect = RuntimeError("CEL evaluation failed")

        with pytest.raises(ValueError):
            result = await agent._resolve_edge_target("nonexistent_node", {})
            assert result == []


async def test_resolve_edge_target_tool_returns_string():
    """Test _resolve_edge_target when tool returns a string."""
    target_node = Agent(name="target_node")
    agent = Agent(name="test_agent", nodes={target_node})

    # Mock tool result with string content
    structured_output = {"result": "target_node"}
    mock_result = CallToolResult(
        structuredContent=structured_output,
        content=[TextContent(text=json.dumps(structured_output), type="text")],
    )

    with patch("swarmx.agent.CLIENT_REGISTRY.call_tool") as mock_call_tool:
        mock_call_tool.return_value = mock_result

        result = await agent._resolve_edge_target("test_tool", {})
        assert result == {target_node}


async def test_resolve_edge_target_tool_returns_list():
    """Test _resolve_edge_target when tool returns a list of strings."""
    node1, node2, node3 = Agent(name="node1"), Agent(name="node2"), Agent(name="node3")
    agent = Agent(name="test_agent", nodes={node1, node2, node3})

    # Mock tool result with list content
    structed_output = {"result": ["node1", "node2", "node3"]}
    mock_result = CallToolResult(
        structuredContent=structed_output,
        content=[TextContent(text=json.dumps(structed_output), type="text")],
    )

    with patch("swarmx.agent.CLIENT_REGISTRY.call_tool") as mock_call_tool:
        mock_call_tool.return_value = mock_result

        result = await agent._resolve_edge_target("test_tool", {})
        assert result == {node1, node2, node3}


async def test_resolve_edge_target_tool_returns_invalid_list():
    """Test _resolve_edge_target when tool returns invalid list (non-string items)."""
    agent = Agent(name="test_agent")

    # Mock tool result with invalid list content
    structured_output = {"result": ["node1", 123, "node3"]}
    mock_result = CallToolResult(
        structuredContent=structured_output,
        content=[TextContent(text=json.dumps(structured_output), type="text")],
    )

    with patch("swarmx.agent.CLIENT_REGISTRY.call_tool") as mock_call_tool:
        mock_call_tool.return_value = mock_result

        with pytest.raises(
            TypeError,
            match="Conditional edge should return string or list of string only",
        ):
            await agent._resolve_edge_target("test_tool", {})


async def test_resolve_edge_target_tool_returns_invalid_type():
    """Test _resolve_edge_target when tool returns invalid type."""
    agent = Agent(name="test_agent")

    # Mock tool result with invalid type
    structured_output = {"result": 123}
    mock_result = CallToolResult(
        structuredContent=structured_output,
        content=[TextContent(text=json.dumps(structured_output), type="text")],
    )

    with patch("swarmx.agent.CLIENT_REGISTRY.call_tool") as mock_call_tool:
        mock_call_tool.return_value = mock_result

        with pytest.raises(
            TypeError,
            match="Conditional edge should return string or list of string only",
        ):
            await agent._resolve_edge_target("test_tool", {})


async def test_resolve_edge_target_tool_returns_text_content():
    """Test _resolve_edge_target when tool returns text content (no structuredContent)."""
    target_node = Agent(name="target_node")
    agent = Agent(name="test_agent", nodes={target_node})

    # Mock tool result with text content
    mock_result = CallToolResult(
        structuredContent=None, content=[TextContent(text="target_node", type="text")]
    )

    with patch("swarmx.agent.CLIENT_REGISTRY.call_tool") as mock_call_tool:
        mock_call_tool.return_value = mock_result

        result = await agent._resolve_edge_target("test_tool", {})
        assert result == {target_node}


async def test_resolve_edge_target_tool_returns_multiple_text_contents():
    """Test _resolve_edge_target when tool returns multiple text contents."""
    agent = Agent(name="test_agent")

    # Mock tool result with multiple text contents
    mock_result = CallToolResult(
        structuredContent=None,
        content=[
            TextContent(text="first", type="text"),
            TextContent(text="second", type="text"),
        ],
    )

    with patch("swarmx.agent.CLIENT_REGISTRY.call_tool") as mock_call_tool:
        mock_call_tool.return_value = mock_result

        with pytest.raises(
            ValueError,
            match="Conditional edge should return one text content block only",
        ):
            await agent._resolve_edge_target("test_tool", {})


async def test_resolve_edge_target_tool_returns_non_text_content():
    """Test _resolve_edge_target when tool returns non-text content."""
    agent = Agent(name="test_agent")

    # Mock tool result with non-text content
    mock_result = CallToolResult(
        structuredContent=None,
        content=[ImageContent(type="image", data="...", mimeType="image/png")],
    )

    with patch("swarmx.agent.CLIENT_REGISTRY.call_tool") as mock_call_tool:
        mock_call_tool.return_value = mock_result

        with pytest.raises(
            ValueError,
            match="Conditional edge should return one text content block only",
        ):
            await agent._resolve_edge_target("test_tool", {})


async def test_resolve_edge_target_with_context():
    """Test _resolve_edge_target passes context to tool call."""
    target_node = Agent(name="target_node")
    agent = Agent(name="test_agent", nodes={target_node})

    context = {"user": "test_user", "data": "test_data"}

    # Mock tool result
    mock_result = CallToolResult(
        structuredContent={"result": "target_node"},
        content=[TextContent(text=json.dumps({"result": "target_node"}), type="text")],
    )

    with patch("swarmx.agent.CLIENT_REGISTRY.call_tool") as mock_call_tool:
        mock_call_tool.return_value = mock_result

        result = await agent._resolve_edge_target("test_tool", context)
        assert result == {target_node}

        # Verify context was passed to tool call
        mock_call_tool.assert_called_with("test_tool", context)


async def test_resolve_edge_target_none_context():
    """Test _resolve_edge_target with None context."""
    target_node = Agent(name="target_node")
    agent = Agent(name="test_agent", nodes={target_node})

    # Mock tool result
    mock_result = CallToolResult(
        structuredContent={"result": "target_node"},
        content=[TextContent(text=json.dumps({"result": "target_node"}), type="text")],
    )

    with patch("swarmx.agent.CLIENT_REGISTRY.call_tool") as mock_call_tool:
        mock_call_tool.return_value = mock_result

        result = await agent._resolve_edge_target("test_tool", None)
        assert result == {target_node}

        # Verify empty dict was passed when context is None
        mock_call_tool.assert_called_with("test_tool", {})
