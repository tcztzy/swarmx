"""SwarmX Agent module."""

import asyncio
import json
import logging
import uuid
import warnings
from collections import defaultdict
from copy import deepcopy
from typing import Annotated, Any, Iterable, cast, get_args

from cel import evaluate
from httpx import Timeout
from jinja2 import Template
from openai import DEFAULT_MAX_RETRIES, AsyncOpenAI
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionChunk,
    ChatCompletionFunctionToolParam,
    ChatCompletionMessageParam,
    ChatCompletionMessageToolCallUnionParam,
)
from openai.types.chat.completion_create_params import (
    CompletionCreateParamsBase,
)
from pydantic import (
    Field,
    PrivateAttr,
    TypeAdapter,
    field_serializer,
    field_validator,
)

from . import settings
from .edge import Edge
from .hook import Hook, HookType
from .mcp_manager import MCPManager, result_to_message
from .nodes import Node, Tool
from .quota import QuotaManager
from .types import CompletionCreateParams, MCPServer, MessagesState
from .utils import completion_to_message

warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")
logging.basicConfig(filename=".swarmx.log", level=logging.INFO)
logger = logging.getLogger(__name__)


class Agent(Node[CompletionCreateParams, MessagesState]):
    """Agent in the agent graph.

    Using this when you need to break down complex tasks into specialized sub-tasks and create reusable, composable AI components. All existing agents are list in tools before this one.

    # Examples:
    - Context: User has requested you to write Python code to solve his/her problem.
        user: "I want to create a Python script for very professional and academic task."
        assistant: "Currently, there are no Python experts in the agent graph (aka swarm), so we need create a new agent who are good at Python programming."
    - Context: User needs to analyze complex data and generate visualizations
        user: "I have a large dataset with sales figures and customer demographics that needs analysis and visualization"
        assistant: "The current swarm lacks data analysis expertise. I'll create a DataAnalyst agent specialized in pandas, matplotlib, and statistical analysis to handle this task."
    - Context: User requests content creation with specific tone and style
        user: "I need marketing copy written for a new SaaS product launch, targeting enterprise customers with a professional tone"
        assistant: "There's no marketing content specialist in the swarm. I'll create a ContentWriter agent focused on B2B marketing copy, brand voice consistency, and conversion optimization."
    - Context: User needs API integration and web scraping capabilities
        user: "I want to build a system that scrapes product data from e-commerce sites and integrates with our inventory management API"
        assistant: "The swarm needs web scraping and API integration expertise. I'll create a WebIntegration agent specialized in BeautifulSoup, requests, and REST API development."
    """

    name: Annotated[
        str, Field(pattern=r"([A-Za-z][A-Za-z0-9]*(?:_[A-Za-z0-9]+)*)", frozen=True)
    ]
    """User-friendly name for the display.

    The name is unique among all nodes, including nested nodes.
    """

    description: str | None = None
    """Agent's description for tool generation and documentation.

    Here is a template for description.
    ```
    Using this agent when <condition or necessity for this role>.

    Examples:
    - Context: <introduce the background of the real world case>
        user: "<user's query>"
        assistant: "<assistant's response>"
    - <more examples like the first one>
    ```
    """

    inputSchema: dict[str, Any] = Field(
        default_factory=lambda: TypeAdapter(CompletionCreateParams).json_schema()
    )
    """Empty inputSchema represent OpenAI chat completions API create parameters."""

    model: str
    """The default model to use for the agent."""

    instructions: str | None = None
    """Agent's instructions, could be a Jinja2 template."""

    mcp_servers: dict[str, MCPServer] = Field(default_factory=dict, alias="mcpServers")
    """MCP configuration for the agent. Should be compatible with claude code."""

    client: AsyncOpenAI | None = None
    """The client to use for the node"""

    nodes: "set[Node]" = Field(default_factory=set)
    """The nodes in the Agent's graph"""

    edges: set[Edge] = Field(default_factory=set)
    """The edges in the Agent's graph"""

    hooks: list[Hook] = Field(default_factory=list)
    """Hooks to execute at various points in the agent lifecycle"""

    _visited: "dict[str, bool]" = PrivateAttr(
        default_factory=lambda: defaultdict(lambda: False)
    )
    _mcp_manager: MCPManager = PrivateAttr(default_factory=MCPManager)

    @property
    def agents(self) -> dict[str, "Agent"]:
        """Return all agents in the graph keyed by their unique name."""
        return self._collect_agents()

    def _collect_agents(self) -> dict[str, "Agent"]:
        collected: dict[str, "Agent"] = {}

        def visit(node: "Agent") -> None:
            if node.name in collected:
                raise ValueError(f"Duplicated agent name: {node.name}")
            collected[node.name] = node
            for child in node.nodes:
                if isinstance(child, Agent):
                    visit(child)

        visit(self)
        return collected

    @classmethod
    def as_init_tool(cls) -> ChatCompletionFunctionToolParam:
        """As init tool."""
        return {
            "type": "function",
            "function": {
                "name": "create_agent",
                "description": cls.__doc__ or "",
                "parameters": cls.model_json_schema(),
            },
        }

    def as_call_tool(self) -> ChatCompletionFunctionToolParam:
        """As call tool."""
        tool: ChatCompletionFunctionToolParam = {
            "type": "function",
            "function": {
                "name": self.name,
                "parameters": self.inputSchema,
            },
        }
        if self.description is not None:
            tool["function"]["description"] = self.description
        return tool

    @field_validator("client", mode="plain")
    def validate_client(cls, v: Any) -> AsyncOpenAI | None:
        """Validate the client.

        If it's a dict, we create a new AsyncOpenAI client from it.
        Otherwise, we assume it's already a valid AsyncOpenAI client.

        """
        if v is None:
            return None
        if isinstance(v, AsyncOpenAI):
            return v
        if isinstance(timeout_dict := v.get("timeout"), dict):
            v["timeout"] = Timeout(**timeout_dict)
        return AsyncOpenAI(**v)

    @field_serializer("client", mode="plain")
    def serialize_client(self, v: AsyncOpenAI | None) -> dict[str, Any] | None:
        """Serialize the client.

        We only serialize the non-default parameters. api_key would not be serialized
        you can manually set it when deserializing.

        """
        if v is None:
            return None
        client: dict[str, Any] = {}
        if str(v.base_url) != "https://api.openai.com/v1":
            client["base_url"] = str(v.base_url)
        for key in (
            "organization",
            "project",
            "websocket_base_url",
        ):
            if (attr := getattr(v, key, None)) is not None:
                client[key] = attr
        if isinstance(v.timeout, float | None):
            client["timeout"] = v.timeout
        elif isinstance(v.timeout, Timeout):
            client["timeout"] = v.timeout.as_dict()
        if v.max_retries != DEFAULT_MAX_RETRIES:
            client["max_retries"] = v.max_retries
        if bool(v._custom_headers):
            client["default_headers"] = v._custom_headers
        if bool(v._custom_query):
            client["default_query"] = v._custom_query
        return client

    def model_post_init(self, context):
        """Post."""
        mcp_manager = (self.model_extra or {}).get("mcp_manager")
        if isinstance(mcp_manager, MCPManager):
            self._mcp_manager = mcp_manager
        # Validate the agent graph immediately so duplicates fail fast
        self._collect_agents()

    async def __call__(
        self,
        params: CompletionCreateParams | None = None,
        *,
        context: dict[str, Any] | None = None,
        quota_manager: QuotaManager | None = None,
    ) -> MessagesState:
        """Run the agent.

        Args:
            params: Dict containing messages and completion settings
            context: The context variables to pass to the agent
            quota_manager: The quota manager for tokens and other resources
            auto_execute_tools: Automatically execute tools or not

        """
        for name, server_params in (
            (settings.mcp_servers if settings is not None else {}) | self.mcp_servers
        ).items():
            await self._mcp_manager.add_server(name, server_params)
        if params is None:
            params = {"messages": []}
        stream = params.get("stream", False)
        if stream:
            raise NotImplementedError("Stream mode is not yet supported now.")
        params = deepcopy(params)
        messages = params["messages"]
        init_len = len(messages)
        if context is None:
            context = {}
        if quota_manager is None:
            quota_manager = QuotaManager(params.get("max_tokens"))  # type: ignore[arg-type]
        self._visited.clear()
        self._cleanup_runtime_tools()
        await self._execute_hooks("on_start", messages, context)
        try:
            completion = await self._create_chat_completion(
                params=params,
                context=context,
                quota_manager=quota_manager,
            )
            total_tokens = completion.usage.total_tokens if completion.usage else 0
            await quota_manager.consume(self.name, total_tokens)
            logger.info(
                json.dumps(
                    completion.model_dump(mode="json", exclude_unset=True)
                    | {"request_id": completion._request_id}
                )
            )

            # Extract message for hook processing and tool calls
            message = completion_to_message(completion)
            message["name"] = f"{self.name} ({completion.id})"
            messages.append(message)

            if (tool_calls := messages[-1].get("tool_calls")) is not None:
                self._register_tool_calls(tool_calls, messages)
            self._visited[self.name] = True
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
                                self._run_tool_node(
                                    target, {"messages": messages}, context
                                )
                            )
                        else:
                            raise TypeError("Unknown type of Node.")
                        task_entries.append((target, task))
                pending_sources = set()
                for target, task in task_entries:
                    result = task.result()
                    messages.extend(result["messages"])
                    self._visited[target.name] = True
                    pending_sources.add(target.name)
            await self._execute_hooks("on_end", messages, context)
            return {"messages": messages[init_len:]}
        finally:
            self._cleanup_runtime_tools()

    def _edge_triggers(self, edge: Edge, pending_sources: set[str]) -> bool:
        """Check whether an edge is ready to fire based on freshly completed sources."""
        if isinstance(edge.source, tuple):
            return all(self._visited.get(name, False) for name in edge.source) and any(
                name in pending_sources for name in edge.source
            )
        return edge.source in pending_sources

    async def _run_tool_node(
        self,
        tool: Tool,
        state: MessagesState,
        context: dict[str, Any],
    ) -> MessagesState:
        """Execute a tool node with hook notifications."""
        messages = state["messages"]
        await self._execute_hooks(
            "on_tool_start",
            messages,
            context,
            tool_name=tool.name,
        )
        try:
            result = await tool(context)
        except Exception:
            await self._execute_hooks(
                "on_tool_end",
                messages,
                context,
                tool_name=tool.name,
            )
            raise
        await self._execute_hooks(
            "on_tool_end",
            messages,
            context,
            tool_name=tool.name,
        )
        return {"messages": [result_to_message(tool.name, result)]}

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
        hook_names = [
            getattr(hook, hook_type)
            for hook in self.hooks
            for hook_type in get_args(HookType)
            if getattr(hook, hook_type, None) is not None
        ]
        return [
            tool
            for tool in self._mcp_manager.tools
            if tool["function"]["name"] not in hook_names
        ] + self._builtin_tools()

    async def _execute_hooks(
        self,
        hook_type: HookType,
        messages: list[ChatCompletionMessageParam],
        context: dict[str, Any],
        *,
        tool_name: str | None = None,
        available_tools: list[ChatCompletionFunctionToolParam] | None = None,
        to_agent: "Agent | None" = None,
        chunk: ChatCompletionChunk | None = None,
        completion: ChatCompletion | None = None,
    ):
        """Execute hooks of a specific type.

        Args:
            hook_type: The type of hook to execute (e.g., 'on_start', 'on_end')
            messages: The current messages to pass to hook tools
            context: The context variables to pass to the hook tools
            tool_name: The name of the tool being called (for on_tool_start and on_tool_end)
            available_tools: The available tools can be called
            to_agent: The agent being handed off to (for on_handoff)
            chunk: The ChatCompletionChunk object (for on_chunk)
            completion: The ChatCompletion object (for on_llm_end)

        """
        for hook in [h for h in self.hooks if getattr(h, hook_type, None) is not None]:
            hook_name: str = getattr(hook, hook_type)
            hook_tool = self._mcp_manager.get_tool(hook_name)
            properties = hook_tool.inputSchema["properties"]
            arguments: dict[str, Any] = {}
            available = {"messages": messages, "context": context}
            if chunk is not None:
                available["chunk"] = chunk
            if completion is not None:
                available["completion"] = completion
            if tool_name is not None:
                available["tool"] = self._mcp_manager.get_tool(tool_name)
            if to_agent is not None:
                available["to_agent"] = to_agent.model_dump(
                    mode="json", exclude_unset=True
                )
            else:
                available["agent"] = self.model_dump(mode="json", exclude_unset=True)
            if available_tools is not None:
                available["available_tools"] = available_tools
            if chunk is not None:
                available["chunk"] = chunk
            if completion is not None:
                available["completion"] = completion
            for key, value in available.items():
                if key in properties:
                    arguments |= {key: value}
            try:
                result = await self._mcp_manager.call_tool(hook_name, arguments)
                if result.structuredContent is None:
                    raise ValueError("Hook tool must return structured content")
                context |= result.structuredContent
            except Exception as e:
                raise e

    async def _get_system_prompt(
        self,
        context: dict[str, Any] | None = None,
    ) -> str | None:
        """Get the system prompt for the agent.

        Args:
            context: The context variables to pass to the agent

        """
        parts = []
        if self.instructions is not None:
            parts.append(
                await Template(self.instructions, enable_async=True).render_async(
                    context or {}
                )
            )
        if len(agent_md_content := settings.get_agents_md_content()) > 0:
            parts.append(
                "Following are extra contexts, what were considered as long-term memory.\n"
                + agent_md_content
            )
        if len(parts) > 0:
            return "\n\n".join(parts)
        return None

    async def _prepare_chat_completion_params(
        self,
        parameters: CompletionCreateParams,
        context: dict[str, Any] | None = None,
    ) -> CompletionCreateParamsBase:
        """Prepare parameters for chat completion."""
        messages = [
            cast(
                ChatCompletionMessageParam,
                {
                    k: v
                    for k, v in m.items()
                    if k not in ("parsed", "reasoning_content")
                },
            )
            for m in parameters["messages"]
            if not (m.get("role") == "user" and m.get("name") == "approval")
        ]
        system_prompt = await self._get_system_prompt(context)
        if system_prompt is not None:
            messages = [{"role": "system", "content": system_prompt}, *messages]
        tools: list[ChatCompletionFunctionToolParam] = [*parameters.get("tools", [])]
        existing_tool_names = {
            tool["function"]["name"] for tool in tools if tool["type"] == "function"
        }
        for tool in self._visible_tools():
            if (
                tool["type"] == "function"
                and tool["function"]["name"] in existing_tool_names
            ):
                continue
            tools.append(tool)
            if tool["type"] == "function":
                existing_tool_names.add(tool["function"]["name"])
        if tools:
            parameters["tools"] = tools
        else:
            parameters.pop("tools", None)
        return parameters | {
            "messages": messages,
            "model": self.model,
        }  # type: ignore

    async def _create_chat_completion(
        self,
        *,
        params: CompletionCreateParams,
        context: dict[str, Any],
        quota_manager: QuotaManager,
    ) -> ChatCompletion:
        """Get a chat completion for the agent with UUID tracing.

        Args:
            params: Parameters to create a chat completion
            context: The context variables to pass to the agent
            quota_manager: Quota manager for tokens and other resource.

        """
        # Even OpenAI support x-request-id header, but most providers don't support
        # So we should manually set it for each.
        request_id = str(uuid.uuid4())
        parameters = await self._prepare_chat_completion_params(
            params,
            context,
        )
        logger.info(json.dumps(parameters | {"request_id": request_id}))
        client = self.client or AsyncOpenAI(
            api_key=settings.OPENAI_API_KEY,
            base_url=settings.OPENAI_BASE_URL,
        )
        messages = params["messages"]
        # Execute on_llm_start hook before making the request
        await self._execute_hooks(
            "on_llm_start",
            messages,
            context,
            available_tools=self._visible_tools(),
        )

        result = await client.chat.completions.create(**parameters)
        result._request_id = request_id
        total_tokens = result.usage.total_tokens if result.usage else 0
        await quota_manager.consume(self.name, total_tokens)
        # Trigger on_llm_end hook for non‑stream response
        await self._execute_hooks("on_llm_end", messages, context, completion=result)
        return result

    def _register_tool_calls(
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
                case "create_agent":
                    self.nodes.add(
                        Agent.model_validate_json(tool_call["function"]["arguments"])
                    )
                case "create_edge":
                    self.edges.add(
                        Edge.model_validate_json(tool_call["function"]["arguments"])
                    )
                case known if known in [
                    self.name,
                    *[node.name for node in self.nodes if isinstance(node, Agent)],
                ]:
                    self.edges.add(Edge(source=self.name, target=known))
                case _:
                    if self._tool_call_completed(messages, tool_call["id"]):
                        continue
                    tool_node = self._mcp_manager.make_tool_node(name, tool_call["id"])
                    self.nodes.add(tool_node)
                    forward = Edge(source=self.name, target=tool_node.name)
                    backward = Edge(source=tool_node.name, target=self.name)
                    self.edges.add(forward)
                    self.edges.add(backward)

    def _tool_call_completed(
        self,
        messages: list[ChatCompletionMessageParam],
        tool_call_id: str,
    ) -> bool:
        return any(
            message.get("tool_call_id") == tool_call_id
            for message in messages
            if message["role"] == "tool"
        )

    def _cleanup_runtime_tools(self) -> None:
        """Remove transient tool nodes and their edges."""
        runtime_nodes = [
            node
            for node in self.nodes
            if isinstance(node, Tool) and getattr(node, "tool_call_id", None)
        ]
        if not runtime_nodes:
            return
        runtime_names = {node.name for node in runtime_nodes}
        self.nodes = {node for node in self.nodes if node.name not in runtime_names}
        remaining_edges: set[Edge] = set()
        for edge in self.edges:
            sources = (
                set(edge.source) if isinstance(edge.source, tuple) else {edge.source}
            )
            if sources & runtime_names:
                continue
            if edge.target in runtime_names:
                continue
            remaining_edges.add(edge)
        self.edges = remaining_edges
        for name in runtime_names:
            self._visited.pop(name, None)

    def _get_node_by_name(self, name: str) -> Node:
        """Get agent by name.

        Only self & level 1 sub agents would be returned, avoid directly handoff to
        sub-sub-agent.
        """
        if name == self.name:
            return self
        for node in self.nodes:
            if node.name == name:
                return node
        raise KeyError(f"Node {name} not exist in nodes")

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
