"""Tests for swarm routing behavior."""

from typing import Any
from unittest.mock import AsyncMock

import pytest
from mcp.types import CallToolResult

from swarmx.agent import Agent
from swarmx.edge import Edge
from swarmx.swarm import Swarm

pytestmark = pytest.mark.anyio


class RecordingAgent(Agent):
    async def __call__(
        self,
        arguments: Any,
        *,
        context: dict[str, Any] | None = None,
        timeout: float | None = None,  # noqa: ARG002
        progress_callable: Any = None,  # noqa: ARG002
    ) -> dict[str, Any]:
        if context is not None:
            context.setdefault("visited", []).append(self.name)
        return {"messages": [{"role": "assistant", "content": self.name}]}


def _nodes(*names: str) -> dict[str, RecordingAgent]:
    return {name: RecordingAgent(name=name, instructions="test") for name in names}


def test_swarm_rejects_unconditional_cycle():
    nodes = _nodes("a", "b")
    with pytest.raises(ValueError, match="DAG"):
        Swarm(
            name="swarm",
            parameters={},
            nodes=nodes,
            edges=[
                Edge(source="a", target="b"),
                Edge(source="b", target="a"),
            ],
            root="a",
        )


def test_swarm_allows_conditional_cycle():
    nodes = _nodes("a", "b")
    swarm = Swarm(
        name="swarm",
        parameters={},
        nodes=nodes,
        edges=[
            Edge(source="a", target="b"),
            Edge(source="b", target="a", condition="flag"),
        ],
        root="a",
    )
    assert swarm.root == "a"


async def test_swarm_routes_via_cel_target():
    nodes = _nodes("root", "b", "c")
    swarm = Swarm(
        name="swarm",
        parameters={},
        nodes=nodes,
        edges=[Edge(source="root", target="flag ? 'b' : 'c'")],
        root="root",
    )
    context: dict[str, Any] = {"flag": True}

    await swarm(
        {"messages": [{"role": "user", "content": "hi"}]},
        context=context,
    )

    assert set(context.get("visited", [])) == {"root", "b"}


async def test_swarm_routes_via_mcp_target():
    nodes = _nodes("root", "b")
    swarm = Swarm(
        name="swarm",
        parameters={},
        nodes=nodes,
        edges=[Edge(source="root", target="mcp__router__pick")],
        root="root",
    )

    async def fake_call_tool(name: str, arguments: dict[str, Any], *args, **kwargs):
        assert name == "mcp__router__pick"
        assert "context" in arguments
        return CallToolResult.model_validate(
            {
                "content": [{"type": "text", "text": '{"destination": "b"}'}],
                "structuredContent": {"destination": "b"},
            }
        )

    swarm._mcp_manager.call_tool = AsyncMock(side_effect=fake_call_tool)

    context: dict[str, Any] = {}
    await swarm(
        {"messages": [{"role": "user", "content": "hi"}]},
        context=context,
    )

    assert set(context.get("visited", [])) == {"root", "b"}
