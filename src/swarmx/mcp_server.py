"""MCP server exposing SwarmX agents as tools."""

from __future__ import annotations

from typing import Any

from mcp.server.fastmcp import FastMCP
from openai.types.chat import ChatCompletionMessageParam

from .agent import Agent
from .types import CompletionCreateParams


def create_mcp_server(agent: Agent) -> FastMCP:
    """Return an MCP server that exposes the agent as a callable tool."""
    server = FastMCP(name=agent.name, instructions=agent.instructions)

    def wrapper(agent: Agent):
        async def call_agent(
            messages: list[ChatCompletionMessageParam],
            context: dict[str, Any] | None = None,
            auto_execute_tools: bool = True,
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

    for subagent in agent.agents.values():
        if isinstance(subagent, Agent):
            server.tool(name=agent.name, description=agent.description)(
                wrapper(subagent)
            )

    return server
