"""Swarm module."""

import asyncio
import json
from collections.abc import Awaitable, Callable
from functools import partial
from textwrap import dedent
from typing import Any, Self, TypedDict

import cel
import networkx as nx
from mcp.types import CallToolResult
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionFunctionToolParam,
    ChatCompletionMessageParam,
    CompletionCreateParams,
)
from pydantic import (
    Field,
    PrivateAttr,
    TypeAdapter,
    model_validator,
)

from .agent import Agent
from .conversion import completion_to_message, result_to_message
from .edge import Edge
from .jsonpatch import PatchOperation
from .mcp_manager import MCPManager
from .messages import Messages
from .node import Node
from .tool import Tool
from .types import MCPServer
from .utils import GenerateJsonSchemaNoTitles

MAX_STEPS = 100


class NodeAttributes(TypedDict, total=False):
    """Node attributes."""

    visited: bool


class EdgeAttributes(TypedDict, total=False):
    """Edge attributes."""

    condition: str


class Graph(nx.DiGraph):
    """Graph."""

    _node: dict[str, NodeAttributes]
    _adj: dict[str, dict[str, EdgeAttributes]]


class Swarm(Node):
    """Orchestrator for agents, tools, and sub-swarms.

    A swarm is a declarative, directed workflow that wires together agents, tools, and
    nested swarms. It handles message flow, conditional routing, and optional MCP-
    backed tools so complex tasks can be decomposed and recombined.

    Use a swarm when a single agent is not enough: multi-stage workflows, branching/
    handoff between specialists, or scenarios where a queen agent should reshape the
    graph on the fly (adding nodes, editing edges, or compressing history).

    How to use a swarm?
    - Define `nodes` with the Agent/Tool/Swarm instances you want to compose.
    - Define `edges` that connect node names or CEL expressions for conditional routing.
    - Optionally set `queen` to an Agent that can patch the swarm structure at runtime.
    - Invoke the swarm like an agent: `await swarm({"messages": [...]}, context={...})`.
    """

    mcpServers: dict[str, MCPServer] = Field(default_factory=dict)
    """Model Context Protocol servers."""
    queen: Node | None = None
    """Optional queen agent for meta-level swarm orchestration."""
    nodes: dict[str, Node]
    """Nodes."""
    edges: list[Edge]
    """Edges."""
    root: str
    """Root node of this swarm."""

    _mcp_manager: MCPManager = PrivateAttr(default_factory=MCPManager)
    _graph: Graph = PrivateAttr(default_factory=Graph)
    _unconditional_predecessors: dict[str, set[str]] = PrivateAttr(default_factory=dict)

    @model_validator(mode="after")
    def is_directed_acyclic_graph(self) -> Self:
        """Check this swarm is DAG."""
        self._rebuild_graphs()
        return self

    def _rebuild_graphs(self) -> None:
        graph = Graph()
        graph.add_nodes_from(self.nodes.keys())
        unconditional_graph = nx.DiGraph()
        unconditional_graph.add_nodes_from(self.nodes.keys())
        predecessors: dict[str, set[str]] = {name: set() for name in self.nodes}

        for edge in self.edges:
            if edge.source not in self.nodes:
                raise ValueError(f"Unknown edge source {edge.source} in swarm")
            if edge.target in self.nodes:
                graph.add_edge(edge.source, edge.target, condition=edge.condition)
            if edge.condition is None and edge.target in self.nodes:
                unconditional_graph.add_edge(edge.source, edge.target)
                predecessors[edge.target].add(edge.source)

        if not nx.is_directed_acyclic_graph(unconditional_graph):
            raise ValueError("Swarm should be a DAG")

        self._graph = graph
        self._unconditional_predecessors = predecessors

    def _build_queen_tools(
        self, available_tools: list[dict]
    ) -> list[ChatCompletionFunctionToolParam]:
        """Build tools for queen agent with embedded swarm state."""
        return [
            {
                "type": "function",
                "function": {
                    "name": "patch_swarm",
                    "description": f"""Modify the swarm structure using SwarmX-flavored JSON Patch (RFC 6902) operations.

Current swarm state:
{self.model_dump(exclude={"mcpServers", "queen"}, exclude_none=True)}

Available MCP tools:
{json.dumps(available_tools, indent=2)}

Use JSON Patch to:
- Add new agents: {{"op": "add", "path": "/nodes/new_agent", "value": {{"name": "new_agent", "model": "..."}}}}
- Remove nodes: {{"op": "remove", "path": "/nodes/node_name"}}
- Add edges: {{"op": "add", "path": "/edges/-", "value": {{"source": "a", "target": "b"}}}}
- Remove edges: {{"op": "remove", "path": "/edges/0"}}""",
                    "parameters": TypeAdapter(list[PatchOperation]).json_schema(
                        schema_generator=GenerateJsonSchemaNoTitles
                    ),
                },
            }
        ]

    def _get_callable(self, name: str) -> Callable[..., Awaitable[CallToolResult]]:
        return partial(self._mcp_manager.call_tool, name)

    def add_node(self, node: Node):
        """Add node."""
        self.nodes[node.name] = node
        self._rebuild_graphs()

    def add_edge(self, u: str, v: str, *, condition: str | None = None):
        """Add edge."""
        edge = Edge(
            source=u,
            target=v,
            condition=condition,
        )
        self.edges.append(edge)
        try:
            self._rebuild_graphs()
        except Exception:
            self.edges.pop()
            raise

    def _edge_condition_passes(self, edge: Edge, context: dict[str, Any]) -> bool:
        condition = edge.condition or "true"
        return bool(cel.evaluate(condition, context))

    async def _resolve_edge_targets(
        self,
        edge: Edge,
        *,
        context: dict[str, Any],
        messages: list[ChatCompletionMessageParam],
    ) -> list[str]:
        target = edge.target
        if target in self.nodes:
            return [target]
        if target.startswith("mcp__"):
            result = await self._mcp_manager.call_tool(
                target,
                {
                    "messages": messages,
                    "context": context,
                    "edge": edge.model_dump(),
                },
            )
            if not isinstance(result, CallToolResult):
                raise TypeError("MCP routing tool returned invalid result")
            return self._parse_tool_destinations(result)
        value = cel.evaluate(target, context)
        return self._normalize_destinations(value)

    def _parse_tool_destinations(self, result: CallToolResult) -> list[str]:
        if result.structuredContent is not None:
            return self._normalize_destinations(result.structuredContent)
        for part in result.content:
            if getattr(part, "type", None) == "text":
                text = getattr(part, "text", "") or ""
                text = text.strip()
                if not text:
                    continue
                try:
                    return self._normalize_destinations(json.loads(text))
                except json.JSONDecodeError:
                    return [text]
        raise ValueError("MCP routing tool returned no destinations")

    def _normalize_destinations(self, value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            return [value]
        if isinstance(value, (list, tuple, set)):
            return [str(item) for item in value]
        if isinstance(value, dict):
            if "destination" in value:
                return [str(value["destination"])]
            if "destinations" in value:
                destinations = value["destinations"]
                if isinstance(destinations, (list, tuple, set)):
                    return [str(item) for item in destinations]
                if destinations is None:
                    return []
                return [str(destinations)]
        raise ValueError("Invalid destinations payload")

    async def __call__(
        self,
        arguments: CompletionCreateParams,
        *,
        context: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> list[ChatCompletionMessageParam]:
        """Entrypoint."""
        for name, server_params in self.mcpServers.items():
            await self._mcp_manager.add_server(name, server_params)
        messages_input = arguments["messages"]
        messages_list = list(messages_input)
        new_messages: list[ChatCompletionMessageParam] = []
        if isinstance(messages_input, Messages):
            arguments["messages"] = messages_input
        else:
            arguments["messages"] = messages_list
        if context is None:
            context = {}
        for node_name in self._graph.nodes:
            self._graph.nodes[node_name]["visited"] = False
        async with asyncio.TaskGroup() as tg:
            pending: set[asyncio.Task[list[ChatCompletionMessageParam]]] = set()
            task_to_node: dict[asyncio.Task[list[ChatCompletionMessageParam]], str] = {}
            scheduled: set[str] = set()
            visited: set[str] = set()

            async def run_node(node_name: str) -> list[ChatCompletionMessageParam]:
                node = self.nodes[node_name]
                match node:
                    case Agent():
                        result = await node(arguments, context=context)
                        if isinstance(result, dict) and "messages" in result:
                            return list(result["messages"])
                        if isinstance(result, ChatCompletion):
                            return [completion_to_message(result)]
                        raise TypeError("Agent returned unexpected result type.")
                    case Tool():
                        result = await node()
                        return [result_to_message("", result)]
                    case Swarm():
                        return await node(arguments, context=context)
                    case _:
                        raise TypeError(f"Unknown node type `{type(node)}`")

            def schedule(node_name: str) -> None:
                if node_name in visited or node_name in scheduled:
                    return
                task = tg.create_task(run_node(node_name))
                pending.add(task)
                task_to_node[task] = node_name
                scheduled.add(node_name)

            schedule(self.root)

            while pending:
                done, pending = await asyncio.wait(
                    pending, return_when=asyncio.FIRST_COMPLETED
                )
                for task in done:
                    node_name = task_to_node.pop(task)
                    node_messages = task.result()
                    if node_messages:
                        messages_list.extend(node_messages)
                        new_messages.extend(node_messages)
                    self._graph.nodes[node_name]["visited"] = True
                    visited.add(node_name)
                    for edge in self.edges:
                        if edge.source != node_name:
                            continue
                        if not self._edge_condition_passes(edge, context):
                            continue
                        targets = await self._resolve_edge_targets(
                            edge, context=context, messages=messages_list
                        )
                        for target in targets:
                            if target not in self.nodes:
                                raise KeyError(f"Unknown target {target} in swarm")
                            if target in visited or target in scheduled:
                                continue
                            required = self._unconditional_predecessors.get(
                                target, set()
                            )
                            if not required.issubset(visited):
                                continue
                            schedule(target)

        return new_messages


queen = Swarm.model_validate(
    dict(
        name="queen",
        description=dedent("""
            Agent for meta-level swarm orchestration.

            The queen is a pure orchestrator that doesn't participate in business logic:
            - Does NOT add messages to the conversation (like a queen bee doesn't gather nectar)
            - Analyzes conversation context to decide structural changes
            - Creates new agents/tools/sub-swarms based on conversation needs
            - Can rewrite entire conversation (e.g., compress long history into background + query)
            - Modifies edges to control workflow routing
            - Manages memory retrieval and storage

            To rewrite conversation, queen sets context["messages"] to the new message list.
            Example: Compress 100 messages into [background_summary, latest_user_query].

            The queen operates at the meta-level, organizing business rather than doing business.
            """),
        root="",
        parameters={},
        nodes={},
        edges=[],
        mcpServers={
            "basic-memory": {"command": "uvx", "args": ["basic-memory", "mcp"]}
        },
    )
)
"""Example queen for simplicity, user can define their own queen."""
