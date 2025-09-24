"""MCP server exposing SwarmX agents as tools."""

from __future__ import annotations

from typing import Any

from mcp.server.fastmcp import FastMCP
from openai.types.chat import ChatCompletionMessageParam

from .agent import Agent, Mode


def create_mcp_server(agent: Agent) -> FastMCP:
    """Return an MCP server that exposes the agent as a callable tool."""
    server = FastMCP(name=agent.name, instructions=agent.instructions)

    def wrapper(agent: Agent):
        async def call_agent(
            messages: list[ChatCompletionMessageParam],
            context: dict[str, Any] | None = None,
            auto: bool = True,
            mode: Mode = "automatic",
            max_tokens: int | None = None,
        ) -> dict:
            result = await agent.run(
                messages=messages,
                context=context,
                stream=False,
                mode=mode,
                max_tokens=max_tokens,
                auto=auto,
            )

            return {"messages": result, "context": context}

        return call_agent

    for subagent in agent.agents.values():
        server.tool(name=agent.name, description=agent.description)(wrapper(subagent))

    return server
