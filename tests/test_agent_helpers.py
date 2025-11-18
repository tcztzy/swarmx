"""Unit tests for synchronous helpers in swarmx.agent."""

import json

import pytest
from httpx import Timeout
from openai import AsyncOpenAI

import swarmx.agent as agent_module
from swarmx.agent import Agent, Edge


def test_agents_duplicate_detection():
    child = Agent(name="duplicate", model="m")
    with pytest.raises(ValueError, match="Duplicated agent name"):
        Agent(name="duplicate", model="m", nodes={child})


def test_validate_client_accepts_dict_and_async_openai():
    client_dict = {
        "api_key": "test",
        "timeout": {"connect": 1, "read": 2, "write": 2, "pool": 1},
    }
    agent = Agent(name="client", model="m", client=client_dict)  # type: ignore
    assert isinstance(agent.client, AsyncOpenAI)
    assert isinstance(agent.client.timeout, Timeout)

    agent2 = Agent(
        name="client2",
        model="m",
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


def test_extra_tools_modes():
    agent = Agent(name="tools", model="m")
    manual = agent._builtin_tools("locked")
    assert manual == []

    agent.nodes.add(Agent(name="child", model="m"))
    semi = agent._builtin_tools("handoff")
    auto = agent._builtin_tools("expand")
    assert any(tool["function"]["name"] == "create_edge" for tool in semi)
    assert any(tool["function"]["name"] == "create_agent" for tool in auto)


def test_agents_property_returns_all_levels():
    leaf = Agent(name="leaf", model="m")
    child = Agent(name="child", model="m", nodes={leaf})
    root = Agent(name="root", model="m", nodes={child})
    agents = root.agents
    assert set(agents) == {"root", "child", "leaf"}


def test_model_dump_roundtrip_includes_edges_and_nodes():
    child = Agent(name="child", model="m")
    parent = Agent(
        name="parent",
        model="m",
        nodes={child},
        edges={Edge(source="parent", target="child")},
    )
    data = json.loads(parent.model_dump_json())
    assert any(edge["target"] == "child" for edge in data["edges"])
