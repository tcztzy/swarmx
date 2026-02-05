"""Tests for Messages graph behavior."""

from unittest.mock import AsyncMock

import pytest
from mcp.types import CallToolResult
from openai.types.chat import ChatCompletion

from swarmx import Agent, settings
from swarmx.messages import Messages


def _completion(content: str, *, request_id: str | None = None) -> ChatCompletion:
    completion = ChatCompletion.model_validate(
        {
            "id": "chatcmpl-test",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": "stop",
                }
            ],
            "created": 0,
            "model": "gpt-test",
            "object": "chat.completion",
        }
    )
    if request_id is not None:
        completion._request_id = request_id
    return completion


def test_messages_initial_chain_ids_and_branches():
    messages = Messages(
        [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
        ]
    )

    assert messages.origin_ids == [0, 1]
    assert messages.main_ids == [0, 1]
    assert messages.branches["origin"].start == 0
    assert messages.branches["origin"].stop == 1
    assert messages.branches["main"].start == 0
    assert messages.branches["main"].stop == 1
    assert list(messages) == [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi"},
    ]
    assert messages.graph.has_edge(0, 1)


def test_messages_append_llm_and_tool_updates_graph():
    messages = Messages([{"role": "user", "content": "hello"}])
    completion = _completion("hi", request_id="req-123")
    llm_message = {"role": "assistant", "content": "hi"}

    llm_id = messages.append_llm_message(llm_message, completion)
    assert llm_id == "req-123"
    assert messages.graph.nodes[llm_id]["completion"] is completion

    tool_result = CallToolResult.model_validate(
        {
            "content": [{"type": "text", "text": "{}"}],
            "structuredContent": {"destination": "node"},
        }
    )
    tool_message = {"role": "tool", "tool_call_id": "call-1", "content": "{}"}

    tool_id = messages.append_tool_message("call-1", tool_message, tool_result)
    assert tool_id == "call-1"
    assert messages.graph.nodes[tool_id]["result"] is tool_result
    assert messages.branches["origin"].stop == tool_id
    assert messages.branches["main"].stop == tool_id
    assert len(messages) == 3


@pytest.mark.anyio
async def test_agent_updates_messages_graph(model):
    settings.agents_md = []
    agent = Agent(
        name="messages-agent",
        instructions="You are a helpful AI assistant.",
        client={"api_key": "test"},
    )
    assert agent.client is not None
    agent.client.chat.completions.create = AsyncMock(return_value=_completion("hello"))

    messages = Messages([{"role": "user", "content": "hello"}])

    await agent({"messages": messages, "model": model})

    completion_nodes = [
        node_id
        for node_id, data in messages.graph.nodes(data=True)
        if "completion" in data
    ]
    assert len(completion_nodes) == 1
    assert len(messages) == 2
