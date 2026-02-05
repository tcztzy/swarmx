import pytest

from swarmx import settings
from swarmx.agent import Agent
from swarmx.mcp_server import create_mcp_server

pytestmark = pytest.mark.anyio


async def test_create_mcp_server_exposes_agent_tool(model):
    """Ensure the MCP server registers the agent as a callable tool."""

    settings.agents_md = []
    agent = Agent(
        name="test-agent",
        description="Test agent description",
        instructions="You are a helpful AI assistant.",
    )
    server = create_mcp_server(agent)
    assert server.instructions == agent.description

    tools = await server.list_tools()
    assert any(tool.name == agent.name for tool in tools)

    result = await server._tool_manager.call_tool(  # type: ignore[attr-defined]
        agent.name,
        {"messages": [{"role": "user", "content": "Hello"}], "model": model},
    )
    assert "messages" in result and result["messages"][0]["role"] == "assistant"
