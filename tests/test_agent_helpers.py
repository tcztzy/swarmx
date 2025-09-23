"""Unit tests for synchronous helpers in swarmx.agent."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from httpx import Timeout
from openai import AsyncOpenAI

import swarmx.agent as agent_module
from swarmx.agent import Agent, Edge, _apply_message_slice, _parse_front_matter


def test_parse_front_matter_falls_back_to_manual_parsing():
    raw = "invalid: [unclosed"  # YAML parser will raise
    parsed = _parse_front_matter(raw)
    assert parsed == {"invalid": "[unclosed"}

    manual = "name: demo\ndescription: test"
    parsed_manual = _parse_front_matter(manual)
    assert parsed_manual == {"name": "demo", "description": "test"}


def test_apply_message_slice_invalid_pattern():
    with pytest.raises(ValueError, match="Invalid message slice"):
        _apply_message_slice([{"role": "user", "content": "hi"}], "bad:slice")


def test_model_validate_md_accepts_bytes_and_detects_invalid():
    md = b"---\nname: demo\nmodel: test-model\n---\nBody content"
    agent = Agent.model_validate_md(md)
    assert agent.instructions == "Body content"

    with pytest.raises(ValueError, match="Invalid agent markdown"):
        Agent.model_validate_md("no front matter here")


def test_as_agent_md_and_dump(tmp_path: Path):
    agent = Agent(name="demo", description="Demo", model="m", instructions="Hi")
    md_text = agent.as_agent_md()
    assert "Demo" in md_text and "Hi" in md_text

    nested = Agent(name="child", model="m")
    parent = Agent(name="parent", model="m", nodes={nested})
    parent.dump_agent_md(tmp_path)
    assert (tmp_path / "parent.md").exists()
    assert (tmp_path / "child.md").exists()

    with pytest.raises(TypeError):
        parent.dump_agent_md(tmp_path / "not_a_dir")


def test_agents_duplicate_detection():
    child = Agent(name="duplicate", model="m")
    with pytest.raises(ValueError, match="Duplicated agent name"):
        Agent(name="duplicate", model="m", nodes={child})


def test_validate_client_accepts_dict_and_async_openai(monkeypatch):
    client_dict = {
        "api_key": "test",
        "timeout": {"connect": 1, "read": 2, "write": 2, "pool": 1},
    }
    agent = Agent(name="client", model="m", client=client_dict)
    assert isinstance(agent.client, AsyncOpenAI)
    assert isinstance(agent.client.timeout, Timeout)

    raw_client = AsyncOpenAI(
        api_key="key", base_url="https://example.com", max_retries=5
    )
    raw_client._custom_headers["X-Test"] = "true"
    raw_client._custom_query["mode"] = "full"
    agent2 = Agent(
        name="client2",
        model="m",
        client=agent_module.AsyncOpenAI(
            api_key="key", base_url="https://example.com", max_retries=5
        ),
    )
    agent2.client._custom_headers["X-Test"] = "true"
    agent2.client._custom_query["mode"] = "full"
    dumped = agent2.model_dump(mode="json")
    serialized = dumped["client"]
    assert serialized["base_url"] == "https://example.com"
    assert serialized["max_retries"] == 5
    assert serialized["default_headers"] == {"X-Test": "true"}
    assert serialized["default_query"] == {"mode": "full"}


def test_validate_parameters_and_serializer():
    agent = Agent(
        name="params",
        model="m",
        parameters={"temperature": 0.7, "messages": ["ignored"], "model": "ignored"},
    )
    params = agent.parameters.model_dump()
    assert params["temperature"] == 0.7
    dumped = agent.model_dump(mode="json")
    assert "messages" not in dumped["parameters"]
    assert "model" not in dumped["parameters"]


def test_extra_tools_modes(monkeypatch):
    agent = Agent(name="tools", model="m")
    manual = agent.extra_tools("manual")
    assert manual == []

    agent.nodes.add(Agent(name="child", model="m"))
    semi = agent.extra_tools("semi")
    auto = agent.extra_tools("automatic")
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
