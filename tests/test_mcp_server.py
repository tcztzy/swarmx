import pytest

from swarmx import settings
from swarmx.agent import Agent
from swarmx.mcp_server import create_mcp_server

pytestmark = pytest.mark.anyio


async def test_create_mcp_server_exposes_agent_tool(model):
    """Ensure the MCP server registers the agent as a callable tool."""

    settings.agents_md = []
    async with Agent(
        name="test-agent",
        description="Test agent description",
        instructions="You are a helpful AI assistant.",
        model=model,
    ) as agent:
        server = create_mcp_server(agent)
        assert server.instructions == agent.instructions

        tools = await server.list_tools()
        assert any(tool.name == agent.name for tool in tools)

        messages = [{"role": "user", "content": "Hello"}]
        result = await server._tool_manager.call_tool(  # type: ignore[attr-defined]
            agent.name,
            {"messages": messages, "auto": False},
            context=server.get_context(),
            convert_result=False,
        )
        assert "messages" in result and result["messages"][0]["role"] == "assistant"
