import sys

import pytest
from mcp.client.stdio import StdioServerParameters

from swarmx import Agent, settings

pytestmark = pytest.mark.anyio


@pytest.fixture
def hello_agent(model):
    # disable local AGENTS.md for context length.
    settings.agents_md = []
    return Agent(
        name="hello-agent", instructions="You are a helpful AI assistant.", model=model
    )


async def test_agent_run(hello_agent: Agent):
    response = await hello_agent.run(messages=[{"role": "user", "content": "Hello"}])
    assert len(response) >= 1 and response[0]["role"] == "assistant"


async def test_agent_run_stream(hello_agent: Agent):
    response = await hello_agent.run(
        messages=[{"role": "user", "content": "Hello"}], stream=True
    )
    first_id = None
    async for chunk in response:
        if first_id is None:
            first_id = chunk.id
        else:
            assert chunk.id == first_id


@pytest.fixture
def hello_agent_with_time(model):
    # disable local AGENTS.md for context length.
    settings.agents_md = []
    return Agent(
        name="hello-agent",
        instructions="You are a helpful AI assistant.",
        model=model,
        mcpServers={
            "time": StdioServerParameters(
                command=sys.executable, args=["-m", "mcp_server_time"]
            )
        },
    )


async def test_agent_run_with_mcp_tool_call(hello_agent_with_time: Agent):
    response = await hello_agent_with_time.run(
        messages=[
            {
                "role": "user",
                "content": "What's the time now? Response in ISO format exactly without any other characters.",
            }
        ],
        auto_execute_tools=True,
    )
    message = response[-1]
    assert message["role"] == "assistant"

    response = await hello_agent_with_time.run(
        messages=[
            {
                "role": "user",
                "content": "What's the time now? Response in ISO format exactly without any other characters.",
            }
        ],
        stream=True,
        auto_execute_tools=True,
    )
    async for c in response:
        if len(c.choices) > 0 and c.choices[0].delta.role is not None:
            assert c.choices[0].delta.role in ("assistant", "tool")
