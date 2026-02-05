import sys
from unittest.mock import AsyncMock

import pytest
from httpx import Timeout
from mcp.client.stdio import StdioServerParameters
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionChunk

import swarmx.agent as agent_module
from swarmx import Agent, settings

pytestmark = pytest.mark.anyio


def test_validate_client_accepts_dict_and_async_openai():
    client_dict = {
        "api_key": "test",
        "timeout": {"connect": 1, "read": 2, "write": 2, "pool": 1},
    }
    agent = Agent(name="client", client=client_dict)  # type: ignore
    assert isinstance(agent.client, AsyncOpenAI)
    assert isinstance(agent.client.timeout, Timeout)

    agent2 = Agent(
        name="client2",
        client=agent_module.AsyncOpenAI(
            api_key="key",
            base_url="https://example.com",
            max_retries=5,
            default_headers={"X-Test": "true"},
            default_query={"mode": "full"},
        ),
    )
    dumped = agent2.model_dump(mode="json")
    serialized = dumped["client"]
    assert serialized["base_url"] == "https://example.com"
    assert serialized["max_retries"] == 5
    assert serialized["default_headers"] == {"X-Test": "true"}
    assert serialized["default_query"] == {"mode": "full"}
    assert "api_key" not in serialized


@pytest.fixture
async def hello_agent(model):
    # disable local AGENTS.md for context length.
    settings.agents_md = []
    agent = Agent(name="hello-agent", instructions="You are a helpful AI assistant.")
    return agent


async def test_agent_run(hello_agent: Agent):
    response = await hello_agent({"messages": [{"role": "user", "content": "Hello"}]})
    messages = response["messages"]
    assert len(messages) >= 1 and messages[0]["role"] == "assistant"


@pytest.fixture
async def hello_agent_with_time(model):
    # disable local AGENTS.md for context length.
    settings.agents_md = []
    agent = Agent(
        name="hello-agent",
        instructions="You are a helpful AI assistant.",
        mcpServers={
            "time": StdioServerParameters(
                command=sys.executable, args=["-m", "mcp_server_time"]
            )
        },
    )
    return agent


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
    )
    message = response["messages"][-1]
    assert message["role"] == "assistant"


class DummyStream:
    """Simple async iterable over preset chunks."""

    def __init__(self, chunks):
        self._iter = iter(chunks)

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            return next(self._iter)
        except StopIteration:
            raise StopAsyncIteration


def _chunk(data: dict) -> ChatCompletionChunk:
    return ChatCompletionChunk.model_validate(data)


async def test_agent_stream_calls_progress_callable(model):
    settings.agents_md = []
    agent = Agent(
        name="hello-agent",
        instructions="You are a helpful AI assistant.",
        client={"api_key": "test"},
    )

    chunks = [
        _chunk(
            {
                "id": "chatcmpl-1",
                "choices": [
                    {
                        "index": 0,
                        "delta": {"role": "assistant", "content": "Hello "},
                        "finish_reason": None,
                    }
                ],
                "created": 1,
                "model": "gpt-test",
                "object": "chat.completion.chunk",
            }
        ),
        _chunk(
            {
                "id": "chatcmpl-1",
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": "world"},
                        "finish_reason": "stop",
                    }
                ],
                "created": 1,
                "model": "gpt-test",
                "object": "chat.completion.chunk",
            }
        ),
    ]
    assert agent.client is not None
    agent.client.chat.completions.create = AsyncMock(return_value=DummyStream(chunks))

    seen: list[str] = []

    async def progress(
        progress: float, total: float | None, message: str | None
    ) -> None:
        seen.append(message or "")

    await agent(
        {"messages": [{"role": "user", "content": "Hello"}], "stream": True},
        progress_callable=progress,
    )
    assert seen == [chunk.model_dump_json(exclude_unset=True) for chunk in chunks]
