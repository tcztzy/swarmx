"""SwarmX Agent module."""

import logging
import warnings
from collections import defaultdict
from typing import (
    Annotated,
    Any,
    AsyncGenerator,
    Generic,
    Literal,
    TypeVar,
    cast,
    overload,
)

from httpx import Timeout
from jinja2 import Template
from mcp.types import Tool
from openai import DEFAULT_MAX_RETRIES, AsyncOpenAI, AsyncStream
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionAssistantMessageParam,
    ChatCompletionChunk,
    ChatCompletionMessageParam,
    ChatCompletionMessageToolCallParam,
)
from openai.types.chat.completion_create_params import CompletionCreateParamsBase
from openai.types.chat_model import ChatModel
from pydantic import (
    BaseModel,
    Field,
    PrivateAttr,
    TypeAdapter,
    field_serializer,
    field_validator,
)
from pydantic.json_schema import GenerateJsonSchema

from .mcp_client import CLIENT_REGISTRY, exec_tool_calls
from .types import MCPServer
from .utils import join

warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")
logging.basicConfig(filename=".swarmx.log", level=logging.INFO)
logger = logging.getLogger(__name__)

DEFAULT_CLIENT: AsyncOpenAI | None = None
T = TypeVar("T", bound=dict | BaseModel)
U = TypeVar("U", bound=dict | BaseModel)


def _merge_chunk(
    messages: dict[str, ChatCompletionMessageParam],
    chunk: ChatCompletionChunk,
) -> None:
    message = messages[chunk.id]
    delta = chunk.choices[0].delta
    content = message.get("content")
    if delta.content is not None:
        if isinstance(content, str) or content is None:
            message["content"] = (content or "") + delta.content
        else:
            message["content"] = [*content, {"type": "text", "text": delta.content}]  # type: ignore

    if delta.refusal is not None:
        assert message["role"] == "assistant"
        message["refusal"] = (message.get("refusal") or "") + delta.refusal
    if delta.tool_calls is not None:
        # We use defaultdict as intermediate structure here instead of list
        if message["role"] != "assistant":
            raise ValueError("Tool calls can only be added to assistant messages")
        tool_calls = cast(
            defaultdict[int, ChatCompletionMessageToolCallParam],
            message.get(
                "tool_calls",
                defaultdict(
                    lambda: {
                        "id": "",
                        "type": "function",
                        "function": {"arguments": "", "name": ""},
                    },
                ),
            ),
        )
        for call in delta.tool_calls:
            function = call.function
            tool_call = tool_calls[call.index]
            if call.id:
                tool_call["id"] = call.id
            if function:
                tool_call["function"]["arguments"] += function.arguments or ""
                tool_call["function"]["name"] += function.name or ""
            tool_calls[call.index] = tool_call
        message["tool_calls"] = tool_calls  # type: ignore


class SwarmXGenerateJsonSchema(GenerateJsonSchema):
    """Remove the title field from the JSON schema."""

    def field_title_should_be_set(self, schema) -> bool:
        """No title for all fields."""
        return False


class Edge(BaseModel):
    """Edge in the agent graph."""

    source: str | list[str]
    """Name of the source node"""
    target: str
    """Name of the target node"""


class Agent(BaseModel, Generic[T], use_attribute_docstrings=True):
    """Agent node in the swarm.

    An agent is a node in the swarm that can send and receive messages.
    It can have tools and instructions.
    It can also have a client to use for the chat completion API.

    """

    name: Annotated[str, Field(strict=True, max_length=256, frozen=True)] = "Agent"
    """User-friendly name for the display"""

    model: ChatModel | str = "deepseek-reasoner"
    """The default model to use for the agent."""

    instructions: str | None = None
    """Agent's instructions, could be a Jinja2 template"""

    mcp_servers: dict[str, MCPServer] = Field(default_factory=dict, alias="mcpServers")
    """MCP configuration for the agent. Should be compatible with claude code."""

    completion_create_params: CompletionCreateParamsBase = Field(
        default_factory=lambda: {"model": "DUMMY", "messages": iter([])}
    )
    """Additional parameters to pass to the chat completion API."""

    client: AsyncOpenAI | None = None
    """The client to use for the node"""

    distill_agent: "Agent | None" = None
    """Distill context to subagents"""

    summary_agent: "Agent | None" = None
    """Summary subagents' outputs"""

    entry_point: str | None = None
    """The entry point for the subagents"""

    finish_point: str | None = None
    """The finish point for the subagents"""

    nodes: "dict[str, Agent]" = Field(default_factory=dict)
    """The nodes in the Agent's graph"""

    edges: list[Edge | Tool] = Field(default_factory=list)
    """The edges in the Agent's graph"""

    _visited: dict[str, bool] = PrivateAttr(
        default_factory=lambda: defaultdict(lambda: False)
    )

    @classmethod
    def as_tool(cls) -> Tool:
        """Convert the agent to a tool."""
        schema = cls.model_json_schema(schema_generator=SwarmXGenerateJsonSchema)
        return Tool(
            name="swarmx.Agent",
            description="Create new Agent",
            inputSchema=schema,
            outputSchema=schema,
        )

    @field_validator("client", mode="plain")
    def validate_client(cls, v: Any) -> AsyncOpenAI | None:
        """Validate the client.

        If it's a dict, we create a new AsyncOpenAI client from it.
        If it's None, we use the global DEFAULT_CLIENT.
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
            if getattr(v, key, None) is not None:
                client[key] = getattr(v, key)
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

    @overload
    async def _create_chat_completion(
        self,
        *,
        messages: list[ChatCompletionMessageParam],
        context: T | None = None,
        stream: Literal[True],
    ) -> AsyncStream[ChatCompletionChunk]: ...

    @overload
    async def _create_chat_completion(
        self,
        *,
        messages: list[ChatCompletionMessageParam],
        context: T | None = None,
        stream: Literal[False] = False,
    ) -> ChatCompletion: ...

    async def _create_chat_completion(
        self,
        *,
        messages: list[ChatCompletionMessageParam],
        context: T | None = None,
        stream: bool = False,
    ) -> ChatCompletion | AsyncStream[ChatCompletionChunk]:
        """Get a chat completion for the agent.

        Args:
            messages: The messages to start the conversation with
            context: The context variables to pass to the agent
            stream: Whether to stream the response

        """
        system_prompt = await self.get_system_prompt(context)
        if system_prompt is not None:
            messages = [{"role": "system", "content": system_prompt}, *messages]
        params: CompletionCreateParamsBase = {
            "messages": messages,
            "model": self.model,
        }
        logger.debug("Getting chat completion for...:", messages)

        # if tools are not specified, use all available tools from the tool registry
        tools = CLIENT_REGISTRY.tools

        params = self.completion_create_params | params
        if params.get("tools") is None and len(tools) > 0:
            params["tools"] = tools

        return await (
            self.client or DEFAULT_CLIENT or AsyncOpenAI()
        ).chat.completions.create(stream=stream, **params)

    async def _run_node_stream(
        self,
        *,
        node: str | None = None,
        messages: list[ChatCompletionMessageParam],
        context: T | None = None,
    ) -> AsyncGenerator[ChatCompletionChunk, None]:
        """Run the node and its successors."""
        if node is None:
            node = self.entry_point
        if node is None:
            return
        agent = self.nodes[node]
        async for chunk in await agent.run(messages=messages, stream=True):
            yield chunk
        self._visited[node] = True
        if node == self.finish_point:
            return
        generators: list[AsyncGenerator[ChatCompletionChunk, None]] = []
        for edge in self.edges:
            match edge:
                case Edge():
                    if edge.source == node:
                        generators.append(
                            self._run_node_stream(
                                node=edge.target,
                                messages=messages,
                                context=context,
                            )
                        )
                    elif (
                        isinstance(edge.source, list)
                        and node in edge.source
                        and all(self._visited[n] for n in edge.source)
                    ):
                        generators.append(
                            self._run_node_stream(
                                node=edge.target,
                                messages=messages,
                                context=context,
                            )
                        )
                case Tool() as tool:
                    assert tool.outputSchema == self.model_json_schema() or (
                        tool.outputSchema
                        == {
                            "type": "object",
                            "property": {"target": {"type": "string"}},
                            "required": ["target"],
                        }
                    )
                    result = await CLIENT_REGISTRY.call_tool(
                        tool.name,
                        context.model_dump()
                        if isinstance(context, BaseModel)
                        else context or {},
                    )
                    assert (content := result.structuredContent) is not None
                    if (target := content.get("target")) in self.nodes:
                        agent = self.nodes[target]
                    else:
                        agent = Agent.model_validate(result.structuredContent)
                    generators.append(await agent.run(messages=messages, stream=True))

        async for chunk in join(*generators):
            yield chunk

    async def _run_stream(
        self,
        *,
        messages: list[ChatCompletionMessageParam],
        context: T | None = None,
    ) -> AsyncGenerator[ChatCompletionChunk, None]:
        """Run the agent and stream the response.

        Args:
            messages: The messages to start the conversation with
            context: The context variables to pass to the agent

        """
        new_messages: list[ChatCompletionMessageParam] = []
        _messages: dict[str, ChatCompletionMessageParam] = defaultdict(
            lambda: {"role": "assistant"}
        )
        async for chunk in await self._create_chat_completion(
            messages=messages,
            context=context,
            stream=True,
        ):
            logger.info(chunk.model_dump_json(exclude_unset=True))
            _merge_chunk(_messages, chunk)
            yield chunk
            if chunk.choices[0].finish_reason is not None:
                new_messages.append(_messages[chunk.id])
        else:
            if len(new_messages) != len(_messages):
                raise ValueError("Number of messages does not match number of chunks")
        latest_message = new_messages[-1]
        if (
            latest_message["role"] == "assistant"
            and (tool_calls := latest_message.get("tool_calls")) is not None
        ):
            new_messages.extend(await exec_tool_calls(tool_calls))
        if self.distill_agent is not None:
            (
                distilled_messages,
                distilled_context,
            ) = await self.distill_agent._update_context(
                messages=messages + new_messages, context=context
            )
        else:
            distilled_messages = messages + new_messages
            distilled_context = context
        async for chunk in self._run_node_stream(
            messages=distilled_messages,
            context=distilled_context,
        ):
            yield chunk

    async def _run_node(
        self,
        *,
        node: str | None = None,
        messages: list[ChatCompletionMessageParam],
        context: T | None = None,
    ) -> list[ChatCompletionMessageParam]:
        """Run the node and its successors."""
        if node is None:
            node = self.entry_point
        if node is None:
            return []
        agent = self.nodes[node]
        result = await agent.run(messages=messages, stream=False, context=context)
        self._visited[node] = True
        if node == self.finish_point:
            return result
        results: list[ChatCompletionMessageParam] = []
        for edge in self.edges:
            match edge:
                case Edge():
                    if edge.source == node:
                        results.extend(
                            await self._run_node(
                                node=edge.target,
                                messages=messages,
                                context=context,
                            )
                        )
                    elif (
                        isinstance(edge.source, list)
                        and node in edge.source
                        and all(self._visited[n] for n in edge.source)
                    ):
                        results.extend(
                            await self._run_node(
                                node=edge.target,
                                messages=messages,
                                context=context,
                            )
                        )
                case Tool() as tool:
                    assert tool.outputSchema == self.model_json_schema() or (
                        tool.outputSchema
                        == {
                            "type": "object",
                            "property": {"target": {"type": "string"}},
                            "required": ["target"],
                        }
                    )
                    tool_result = await CLIENT_REGISTRY.call_tool(
                        tool.name,
                        context.model_dump()
                        if isinstance(context, BaseModel)
                        else context or {},
                    )
                    assert (content := tool_result.structuredContent) is not None
                    if (target := content.get("target")) in self.nodes:
                        agent = self.nodes[target]
                    else:
                        agent = Agent.model_validate(tool_result.structuredContent)
                    results.extend(await agent.run(messages=messages, stream=False))
        return result + results

    async def _run(
        self,
        *,
        messages: list[ChatCompletionMessageParam],
        context: T | None = None,
    ) -> list[ChatCompletionMessageParam]:
        """Run the agent without streaming the response.

        Args:
            messages: The messages to start the conversation with
            context: The context variables to pass to the agent

        """
        new_messages: list[ChatCompletionMessageParam] = []
        completion = await self._create_chat_completion(
            messages=messages,
            context=context,
            stream=False,
        )
        logger.info(completion.model_dump_json(exclude_unset=True))
        message = completion.choices[0].message
        m = cast(
            ChatCompletionAssistantMessageParam,
            message.model_dump(mode="json", exclude_none=True),
        )
        m["name"] = f"{self.name} ({completion.id})"
        new_messages.append(m)

        if m["role"] == "assistant" and (tool_calls := m.get("tool_calls")) is not None:
            new_messages.extend(await exec_tool_calls(tool_calls))

        if self.distill_agent is not None:
            (
                distilled_messages,
                distilled_context,
            ) = await self.distill_agent._update_context(
                messages=messages + new_messages, context=context
            )
        else:
            distilled_messages = messages + new_messages
            distilled_context = context
        node_results = await self._run_node(
            messages=distilled_messages,
            context=distilled_context,
        )

        return new_messages + node_results

    async def _update_context(
        self,
        messages: list[ChatCompletionMessageParam],
        context: T | None = None,
    ) -> tuple[list[ChatCompletionMessageParam], T | None]:
        """Update context.

        Args:
            messages: The messages to update
            context: The context variables to update

        Returns:
            A tuple of (distilled_messages, distilled_context)

        """
        if self is None:
            return messages, context

        _messages = await self.run(
            messages=messages,
            stream=False,
            context=context,
        )
        if len(messages) == 1 and (message := _messages[0])["role"] == "assistant":
            content = message.get("content") or "[[],{}]"
            if not isinstance(content, str):
                content = "".join(p["text"] for p in content if p["type"] == "text")
            try:
                return TypeAdapter(
                    tuple[list[ChatCompletionMessageParam], T]
                ).validate_python(content)
            except Exception:
                return messages, context
        else:
            return messages, context

    async def get_system_prompt(
        self,
        context: T | None = None,
    ) -> str | None:
        """Get the system prompt for the agent.

        Args:
            context: The context variables to pass to the agent

        """
        if self.instructions is None:
            return None
        return await Template(self.instructions, enable_async=True).render_async(
            context or {}
        )

    @overload
    async def run(
        self,
        *,
        messages: list[ChatCompletionMessageParam],
        stream: Literal[True],
        context: T | None = None,
    ) -> AsyncGenerator[ChatCompletionChunk, None]: ...

    @overload
    async def run(
        self,
        *,
        messages: list[ChatCompletionMessageParam],
        stream: Literal[False] = False,
        context: T | None = None,
    ) -> list[ChatCompletionMessageParam]: ...

    async def run(
        self,
        *,
        messages: list[ChatCompletionMessageParam],
        stream: bool = False,
        context: T | None = None,
    ) -> list[ChatCompletionMessageParam] | AsyncGenerator[ChatCompletionChunk, None]:
        """Run the agent.

        Args:
            messages: The messages to start the conversation with
            stream: Whether to stream the response
            context: The context variables to pass to the agent
            execute_tools: Whether to execute tool calls

        """
        for name, server_params in self.mcp_servers.items():
            await CLIENT_REGISTRY.add_server(name, server_params)
        if stream:
            return self._run_stream(
                messages=messages,
                context=context,
            )
        return await self._run(
            messages=messages,
            context=context,
        )
