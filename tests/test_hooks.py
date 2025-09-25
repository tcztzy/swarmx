"""Tests for SwarmX hook execution with a real MCP server."""

import pytest

from swarmx import settings
from swarmx.agent import Agent
from swarmx.hook import Hook

pytestmark = pytest.mark.anyio


async def test_execute_hooks_updates_context_via_mcp(hooks_params, model):
    settings.agents_md = []
    agent = Agent(
        name="hook-agent",
        description="Agent with hooks",
        instructions="You are a helpful AI assistant.",
        model=model,
        hooks=[Hook(on_start="hooks/record")],
        mcpServers={"hooks": hooks_params},
    )

    context = {"hook_counter": 0}

    await agent.run(messages=[{"role": "user", "content": "Hello"}], context=context)

    assert context["hook_counter"] == 1
    assert context["last_hook"]["messages_count"] == 1  # type: ignore
    assert context["last_hook"]["agent_name"] == "hook-agent"  # type: ignore
