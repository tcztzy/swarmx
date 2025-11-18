import sys

import pytest
from mcp.client.stdio import StdioServerParameters

from swarmx import Agent, settings

pytestmark = pytest.mark.anyio


@pytest.fixture
async def hello_agent(model):
    # disable local AGENTS.md for context length.
    settings.agents_md = []
    async with Agent(
        name="hello-agent", instructions="You are a helpful AI assistant.", model=model
    ) as agent:
        yield agent


async def test_agent_run(hello_agent: Agent):
    response = await hello_agent({"messages": [{"role": "user", "content": "Hello"}]})
    assert len(response) >= 1 and response[0]["role"] == "assistant"


@pytest.fixture
async def hello_agent_with_time(model):
    # disable local AGENTS.md for context length.
    settings.agents_md = []
    async with Agent(
        name="hello-agent",
        instructions="You are a helpful AI assistant.",
        model=model,
        mcpServers={
            "time": StdioServerParameters(
                command=sys.executable, args=["-m", "mcp_server_time"]
            )
        },
    ) as agent:
        yield agent


async def test_agent_run_with_mcp_tool_call(hello_agent_with_time: Agent):
    response = await hello_agent_with_time(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "What's the time now? Response in ISO format exactly without any other characters.",
                }
            ]
        },
        auto_execute_tools=True,
    )
    message = response[-1]
    assert message["role"] == "assistant"
