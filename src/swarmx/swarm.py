"""Swarm module."""

import asyncio
import re
from collections.abc import Iterable
from copy import deepcopy
from typing import (
    Annotated,
    Any,
    Literal,
    Required,
    TypeAlias,
    TypedDict,
    cast,
    overload,
)

import networkx as nx
from cel import evaluate
from openai.types.chat import (
    ChatCompletionFunctionToolParam,
    ChatCompletionMessageParam,
    ChatCompletionMessageToolCallUnionParam,
)
from pydantic import (
    PlainSerializer,
    PlainValidator,
    TypeAdapter,
    WithJsonSchema,
)

from . import settings
from .agent import Agent
from .edge import Edge
from .mcp_manager import MCPManager
from .node import Node
from .tool import Tool
from .types import CompletionCreateParams, MCPServer, MessagesState


class _Edge(TypedDict):
    source: str | tuple[str, ...]
    target: str


class _Agent(TypedDict):
    type: Literal["agent"]
    agent: Any


class _Tool(TypedDict):
    type: Literal["tool"]
    tool: Any


_Node: TypeAlias = "_Swarm | _Agent | _Tool"


class _Swarm_(TypedDict, total=False):
    name: Required[str]
    nodes: Required[list[_Node]]
    edges: Required[list[_Edge]]
    description: str


class _Swarm(TypedDict):
    type: Literal["swarm"]
    swarm: _Swarm_


def node_validator(value: Any) -> "Swarm | Agent | Tool":
    """Validate and convert value to a Node (Swarm, Agent, or Tool)."""
    match value:
        case Swarm() | Agent() | Tool():
            return value
        case dict():
            data = cast(_Node, TypeAdapter(_Node).validate_python(value))
            match data["type"]:
                case "swarm":
                    return swarm_validator(data["swarm"])
                case "agent":
                    return Agent.model_validate(data["agent"])
                case "tool":
                    return Tool.model_validate(data["tool"])
        case _:
            raise ValueError(f"Cannot convert {type(value)} to Node")


def swarm_validator(value: Any) -> "Swarm":
    """Validate Swarm."""
    match value:
        case Swarm():
            return value
        case dict():
            graph = TypeAdapter(_Swarm).validate_python(value)
            data = graph["swarm"]
            swarm = Swarm(name=data["name"])
            if description := data.get("description"):
                swarm.description = description
            for node in data["nodes"]:
                swarm.add_node(node_validator(node))
            for edge in data["edges"]:
                swarm.add_edge(
                    edge["source"],
                    edge["target"],
                    **{k: v for k, v in edge.items() if k not in ("source", "target")},
                )
            return swarm
        case _:
            raise ValueError(f"Cannot convert {type(value)} to NetworkX graph")


def swarm_serializer(swarm: "Swarm") -> _Swarm:
    """Serialize Swarm to dict format.

    Extracts node and edge data from the NetworkX graph, handling:
    - Nodes: Serializes based on type (agent/tool/swarm)
    - Edges: Includes source, target, and all edge attributes
    """
    nodes: list[_Node] = []
    for _, node_data in swarm.nodes(data=True):
        # Extract the typed node based on node type
        node_type = node_data.get("type")
        if node_type not in ("agent", "tool", "swarm"):
            raise TypeError(f"Node type '{node_type}' is not supported")
        node_obj = node_data.get(node_type, None)
        match node_obj:
            case Agent() | Tool():
                nodes.append(
                    {"type": node_type, node_type: node_obj.model_dump(mode="json")}  # type: ignore
                )
            case Swarm():
                nodes.append(swarm_serializer(node_obj))
            case None:
                raise KeyError(f"Node type '{node_type}' declared but object not found")
            case _:
                raise TypeError(
                    f"Node type '{node_type}' declared but got '{type(node_obj)}' object {node_obj}"
                )

    return {
        "type": "swarm",
        "swarm": {
            "name": swarm.name,
            "description": swarm.description,
            "nodes": nodes,
            "edges": [  # type: ignore
                {"source": source, "target": target, **attr}
                for source, target, attr in swarm.edges(data=True)
            ],
        },
    }


SwarmBase = Annotated[
    nx.DiGraph,
    PlainValidator(swarm_validator),
    PlainSerializer(swarm_serializer),
    WithJsonSchema(
        {
            "$ref": "#/defs/swarm",
            "$defs": {
                "swarm": {
                    "type": "object",
                    "properties": {
                        "type": {"const": "swarm"},
                        "swarm": {
                            "type": "object",
                            "properties": {
                                "name": {
                                    "type": "string",
                                    "maxLength": 64,
                                    "minLength": 1,
                                },
                                "nodes": {
                                    "type": "array",
                                    "items": {"$ref": "#/$defs/node"},
                                },
                                "edges": {"type": "array"},
                            },
                            "required": ["name", "nodes", "edges"],
                        },
                    },
                    "required": ["type", "swarm"],
                },
                "agent": {
                    "type": "object",
                    "properties": {
                        "type": {"const": "agent"},
                        "agent": {"type": "object"},
                    },
                    "required": ["type", "agent"],
                },
                "tool": {
                    "type": "object",
                    "properties": {
                        "type": {"const": "tool"},
                        "tool": {"type": "object"},
                    },
                    "required": ["type", "tool"],
                },
                "node": {
                    "oneOf": [
                        {"$ref": "#/$defs/swarm"},
                        {"$ref": "#/$defs/agent"},
                        {"$ref": "#/$defs/tool"},
                    ]
                },
                "edge": {
                    "type": "object",
                    "properties": {
                        "source": {
                            "oneOf": [
                                {"type": "string"},
                                {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "uniqueItems": True,
                                    "minItems": 2,
                                },
                            ]
                        },
                        "target": {"type": "string"},
                    },
                    "required": ["source", "target"],
                },
            },
        }
    ),
]


class Swarm(SwarmBase):
    """Swarm workflow graph - DAG with conditional routing.

    A Swarm is both a workflow graph and an executable Node, allowing
    swarms to be nested within other swarms.
    """

    mcpServers: dict[str, MCPServer]
    """MCP configuration for the agent. Should be compatible with claude code."""
    _mcp_manager: MCPManager
    _node: dict
    _adj: dict[Any, dict[Any, dict]]
    adjlist_inner_dict_factory: type[dict]  # type: ignore
    node_attr_dict_factory: type[dict]  # type: ignore
    edge_attr_dict_factory: type[dict]  # type: ignore

    def __init__(  # noqa: D107
        self,
        incoming_graph_data=None,
        *,
        name: str,
        mcpServers: dict[str, MCPServer] | None = None,
        **attr,
    ):
        super().__init__(incoming_graph_data, name=name, **attr)
        self._mcp_manager = MCPManager()
        self.mcpServers = mcpServers or {}

    def __hash__(self):
        """Swarm's name is unique."""
        return hash(self.name)

    def __eq__(self, other):
        """Two Swarms are equal if they have the same name."""
        if isinstance(other, Swarm):
            return (
                self.name == other.name
                and self.nodes == other.nodes
                and self.edges == other.edges
                and self.graph == other.graph
            )
        return False

    @property
    def name(self) -> str:
        """Name."""
        return self.graph.get("name", "")

    @name.setter
    def name(self, s: Any):
        if not isinstance(s, str):
            raise TypeError("Name must be str")
        if not re.match(r"^[A-Za-z][A-Za-z0-9_-]{0,63}$", s):
            raise ValueError(
                "Mal-format name, name should has length between 1 and 64 and only alphabet & digits & dash & underline."
            )
        self.graph["name"] = s

    @property
    def description(self):
        """Description."""
        return self.graph.get("description", "")

    @description.setter
    def description(self, value: Any):
        if not isinstance(value, str):
            raise TypeError("Description must be str")
        self.graph["description"] = value

    def add_node(self, node_for_adding: Node | Any, **attr) -> None:
        """Add a node to the Swarm graph.

        Parameters
        ----------
        node_for_adding : Node | Any
            Either a Node object (with a 'name' attribute) or a node identifier.
            If a Node object is provided, its name will be used as the node ID
            and the Node itself will be stored as the 'node' attribute.
        **attr
            Keyword arguments to add as node attributes. Should not be provided
            if node_for_adding is a Node object.

        Examples
        --------
        >>> swarm = Swarm()
        >>> # Add with Node object
        >>> agent = Agent(name="agent_a", instructions="...")
        >>> swarm.add_node(agent)
        >>> # Add with ID and attributes
        >>> swarm.add_node("agent_b", type="agent", model="gpt-4")

        """
        if isinstance(node_for_adding, Node):
            # When Node is passed, use its name as ID and store the Node
            if attr:
                raise ValueError(
                    "Cannot provide attributes when adding a Node object. "
                    "Node attributes are inferred from the Node itself."
                )
            match node_for_adding:
                case Agent():
                    super().add_node(
                        node_for_adding.name, type="agent", agent=node_for_adding
                    )
                case Tool():
                    super().add_node(
                        node_for_adding.name, type="tool", tool=node_for_adding
                    )
                case Swarm():
                    super().add_node(
                        node_for_adding.name, type="swarm", swarm=node_for_adding
                    )
        else:
            # Regular node ID with optional attributes
            super().add_node(node_for_adding, **attr)

    @overload
    def add_edge(
        self,
        u_of_edge: str | tuple[str, ...],
        v_of_edge: str,
        **attr,
    ): ...
    @overload
    def add_edge(
        self,
        u_of_edge: Edge,
        **attr,
    ): ...
    def add_edge(
        self,
        u_of_edge: str | tuple[str, ...] | Edge,
        v_of_edge: str | None = None,
        **attr,
    ):
        """Add edge."""
        if isinstance(u_of_edge, Edge):
            self.add_edge(u_of_edge.source, u_of_edge.target, **attr)
        u, v = u_of_edge, v_of_edge
        if isinstance(u_of_edge, tuple):
            if u not in self._adj:
                self._adj[u] = self.adjlist_inner_dict_factory()
            for uu in u:
                if uu not in self._node:
                    if uu is None:
                        raise ValueError("None cannot be a node")
                    self.add_node(uu)
            if v not in self._node:
                if v is None:
                    raise ValueError("None cannot be a node")
                self._adj[v] = self.adjlist_inner_dict_factory()
                self._node[v] = self.node_attr_dict_factory()
            datadict = self._adj[u].get(v, self.edge_attr_dict_factory())
            datadict.update(attr)
            self._adj[u][v] = datadict
            self._adj[v][u] = datadict
            nx._clear_cache(self)
        else:
            super().add_edge(u, v, **attr)

    async def __call__(
        self,
        arguments: CompletionCreateParams,
        *,
        context: dict[str, Any] | None = None,
        **kwargs,
    ) -> MessagesState:
        """Execute the Swarm workflow.

        The Swarm acts as a Node, executing the workflow graph starting from
        the entry node and following edges based on routing conditions.

        Parameters
        ----------
        arguments : CompletionCreateParams
            Dict containing messages and completion settings
        context : dict[str, Any] | None, optional
            The context variables to pass through the workflow, by default None
        **kwargs
            Additional keyword arguments (e.g., quota_manager)

        Returns
        -------
        MessagesState
            The final messages state after workflow execution

        Raises
        ------
        NotImplementedError
            Swarm execution is not yet fully implemented

        """
        # TODO: Implement workflow execution logic
        # This should:
        # 1. Find the entry node (node with no predecessors or marked as entry)
        # 2. Execute nodes following the edges
        # 3. Evaluate routing conditions (CEL expressions, MCP tools)
        # 4. Handle cycles with conditional edges
        # 5. Return the final MessagesState
        for name, server_params in (
            (settings.mcp_servers if settings is not None else {}) | self.mcpServers
        ).items():
            await self._mcp_manager.add_server(name, server_params)
        messages = deepcopy(arguments["messages"])
        init_len = len(messages)
        if (tool_calls := messages[-1].get("tool_calls")) is not None:
            self._handle_tool_calls(tool_calls, messages)
        pending_sources: set[str] = {self.name}
        while pending_sources:
            targets: dict[str, Node] = {}
            for edge in self.edges:
                if not self._edge_triggers(edge, pending_sources):
                    continue
                resolved = await self._resolve_edge_target(edge.target, context)
                for target in resolved:
                    targets[target.name] = target
            if not targets:
                break
            task_entries: list[tuple[Node, asyncio.Task[MessagesState]]] = []
            async with asyncio.TaskGroup() as tg:
                for target in targets.values():
                    if isinstance(target, Agent):
                        task = tg.create_task(target({"messages": messages}))
                    elif isinstance(target, Tool):
                        task = tg.create_task(
                            self._run_tool_node(target, {"messages": messages}, context)
                        )
                    else:
                        raise TypeError("Unknown type of Node.")
                    task_entries.append((target, task))
            pending_sources = set()
            for target, task in task_entries:
                result = task.result()
                messages.extend(result["messages"])
                pending_sources.add(target.name)
        await self._execute_hooks("on_end", messages, context)
        return {"messages": messages[init_len:]}

    def _builtin_tools(self) -> list[ChatCompletionFunctionToolParam]:
        """Graph management tools available to every agent."""
        return [
            *[
                agent.as_call_tool()
                for agent in [self, *self.nodes]
                if isinstance(agent, Agent)
            ]
        ] + [
            Agent.as_init_tool(),
            Edge.as_tool(),
        ]

    def _visible_tools(self):
        return [tool for tool in self._mcp_manager.tools] + self._builtin_tools()

    def _get_node_by_name(self, name) -> Node:
        if name not in self.nodes:
            raise KeyError(f"Node {name} not exist in nodes")
        node_data = self.nodes[name]
        node_type = node_data["type"]
        return node_data[node_type]

    async def _resolve_edge_target(
        self, target: str, context: dict[str, Any] | None = None
    ) -> set[Node]:
        """Resolve edge target, which can be a node name, function name, or CEL expression."""
        # First check if target exists as a node
        try:
            return {self._get_node_by_name(target)}
        except KeyError:
            pass

        # Then check if target is a function available through this registry
        try:
            result = await self._mcp_manager.call_tool(target, context or {})

            if result.structuredContent is not None:
                r = result.structuredContent.get("result")
                if isinstance(r, list) and all(isinstance(item, str) for item in r):
                    return {self._get_node_by_name(s) for s in r}
                elif isinstance(r, str):
                    return {self._get_node_by_name(r)}
                else:
                    raise TypeError(
                        "Conditional edge should return string or list of string only"
                    )
            else:
                if len(result.content) != 1 or result.content[0].type != "text":
                    raise ValueError(
                        "Conditional edge should return one text content block only"
                    )
                return {self._get_node_by_name(result.content[0].text)}
        except KeyError:
            pass

        # Finally try to evaluate as CEL expression
        try:
            result = evaluate(target, context)
            if isinstance(result, str):
                return {self._get_node_by_name(result)}
            elif isinstance(result, list) and all(
                isinstance(item, str) for item in result
            ):
                return {self._get_node_by_name(s) for s in result}
            else:
                raise ValueError(
                    f"CEL expression must return str or list[str], got {type(result)}"
                )
        except Exception as e:
            raise ValueError(f"Invalid edge target '{target}': {e}")

    def _handle_tool_calls(
        self,
        tool_calls: Iterable[ChatCompletionMessageToolCallUnionParam],
        messages: list[ChatCompletionMessageParam],
    ) -> None:
        """Add tool nodes for pending tool calls."""
        for tool_call in tool_calls:
            if tool_call["type"] == "custom":
                continue
            name = tool_call["function"]["name"]
            match name:
                case "Agent":
                    self.add_node(
                        Agent.model_validate_json(tool_call["function"]["arguments"])
                    )
                case "Edge":
                    self.add_edge(
                        Edge.model_validate_json(tool_call["function"]["arguments"])
                    )
                case known if known in self:
                    self.add_edge(self.name, known)
                case _:
                    pass
