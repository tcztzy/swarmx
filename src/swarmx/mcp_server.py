"""MCP server exposing SwarmX agents as tools."""

from collections.abc import Awaitable, Callable
from typing import Any

from mcp.server.fastmcp import FastMCP
from openai.types.chat import ChatCompletionMessageParam

from .agent import Agent
from .swarm import Swarm


def _sanitize_tool_fn_name(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in name)


def _wrap_agent_tool(agent: Swarm | Agent) -> Callable[..., Awaitable[dict[str, Any]]]:
    async def run_agent(
        messages: list[ChatCompletionMessageParam],
        model: str | None = None,
        stream: bool = False,
        max_tokens: int | None = None,
    ) -> dict[str, Any]:
        arguments: dict[str, Any] = {"messages": messages}
        if model is not None:
            arguments["model"] = model
        if stream:
            arguments["stream"] = stream
        if max_tokens is not None:
            arguments["max_tokens"] = max_tokens
        result = await agent(arguments)
        if isinstance(result, list):
            return {"messages": result}
        return result

    run_agent.__name__ = _sanitize_tool_fn_name(agent.name) or "agent_tool"
    return run_agent


def create_mcp_server(agent: Swarm | Agent) -> FastMCP:
    """Return an MCP server that exposes the agent as a callable tool.

    If you want using tool node, just add its belonging mcp server.
    """
    server = FastMCP(name=agent.name, instructions=agent.description)

    server.tool(name=agent.name, description=agent.description)(_wrap_agent_tool(agent))
    for name, subagent in agent.nodes.items() if isinstance(agent, Swarm) else ():
        if isinstance(subagent, Swarm | Agent):
            server.tool(name=name, description=subagent.description)(
                _wrap_agent_tool(subagent)
            )

    return server
