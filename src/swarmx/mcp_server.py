"""MCP server exposing SwarmX agents as tools."""

from __future__ import annotations

from typing import Any

from mcp.server.fastmcp import FastMCP
from openai.types.chat import ChatCompletionMessageParam

from .agent import Agent
from .swarm import Swarm
from .types import CompletionCreateParams


def create_mcp_server(agent: Swarm) -> FastMCP:
    """Return an MCP server that exposes the agent as a callable tool."""
    server = FastMCP(name=agent.name, instructions=agent.description)

    def wrapper(agent: Swarm | Agent):
        async def call_agent(
            messages: list[ChatCompletionMessageParam],
            context: dict[str, Any] | None = None,
            max_tokens: int | None = None,
        ) -> dict:
            state: CompletionCreateParams = {"messages": messages}
            if max_tokens is not None:
                state["max_tokens"] = max_tokens
            result = await agent(
                state,
                context=context,
            )

            return {"messages": result, "context": context}

        return call_agent

    server.tool(name=agent.name, description=agent.description)(wrapper(agent))
    for name, data in agent.nodes(data=True):
        if data["type"] in ("agent", "swarm"):
            subagent: Swarm | Agent = data[data["type"]]
            server.tool(name=name, description=subagent.description)(wrapper(subagent))

    return server
