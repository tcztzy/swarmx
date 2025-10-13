import re
import sys

import pytest
from mcp.client.stdio import StdioServerParameters
from pydantic import BaseModel

from swarmx import Agent, settings
from swarmx.agent import STRUCTURED_OUTPUT_TOOL_PREFIX, Parameters

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
    response = await hello_agent.run(messages=[{"role": "user", "content": "Hello"}])
    assert len(response) >= 1 and response[0]["role"] == "assistant"


class StructuredPayload(BaseModel):
    message: str
    count: int


def test_parameters_response_format_coercion():
    params = Parameters(response_format=StructuredPayload)
    response_format = params.model_dump()["response_format"]
    assert response_format is not None
    assert response_format["type"] == "json_schema"
    json_schema = response_format["json_schema"]
    schema = json_schema["schema"]
    assert schema["type"] == "object"
    assert schema["properties"]["message"]["type"] == "string"
    assert schema["properties"]["count"]["type"] == "integer"
    name = json_schema["name"]
    assert "RootModel" not in name
    assert re.fullmatch(r"[A-Za-z0-9_]+", name)


def test_parameters_response_format_sanitizes_root_models():
    params = Parameters(response_format=dict[str, list[int]])
    response_format = params.model_dump()["response_format"]
    assert response_format is not None
    json_schema = response_format["json_schema"]
    name = json_schema["name"]
    assert "RootModel" not in name
    assert re.fullmatch(r"[A-Za-z0-9_]+", name)


async def test_agent_prepares_tool_fallback_for_json_schema(
    monkeypatch: pytest.MonkeyPatch,
):
    model_name = "fallback-model"
    settings.format_fallback_tool_models = {model_name}
    agent = Agent(
        name="Structured Agent!",
        model=model_name,
        parameters=Parameters(response_format=StructuredPayload),
    )
    params = await agent._prepare_chat_completion_params(
        messages=[{"role": "user", "content": "Hello"}]
    )
    tool_name = params["tool_choice"]["function"]["name"]
    assert "response_format" not in params
    assert params["tool_choice"]["function"]["name"] == tool_name
    assert tool_name.startswith(STRUCTURED_OUTPUT_TOOL_PREFIX)
    assert re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", tool_name)
    tools = params["tools"]
    assert any(
        tool["function"]["name"] == tool_name
        and tool["function"]["parameters"]["type"] == "object"
        for tool in tools
    )


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
