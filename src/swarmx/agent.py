"""SwarmX Agent module."""

import asyncio
import logging
import re
import warnings
from collections import defaultdict
from copy import deepcopy
from typing import (
    Annotated,
    Any,
    AsyncGenerator,
    Literal,
    TypeVar,
    cast,
    overload,
)

from httpx import Timeout
from jinja2 import Template
from openai import DEFAULT_MAX_RETRIES, AsyncOpenAI, AsyncStream
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionAssistantMessageParam,
    ChatCompletionChunk,
    ChatCompletionMessageParam,
    ChatCompletionMessageToolCallParam,
    ParsedChatCompletion,
)
from openai.types.chat.completion_create_params import CompletionCreateParamsBase
from openai.types.chat_model import ChatModel
from pydantic import (
    BaseModel,
    Field,
    PrivateAttr,
    field_serializer,
    field_validator,
)
from pydantic.json_schema import GenerateJsonSchema

from .hook import Hook, HookType
from .mcp_client import CLIENT_REGISTRY, exec_tool_call
from .types import MCPServer
from .utils import join

warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")
logging.basicConfig(filename=".swarmx.log", level=logging.INFO)
logger = logging.getLogger(__name__)

DEFAULT_CLIENT: AsyncOpenAI | None = None
T = TypeVar("T")


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


def _apply_message_slice(
    messages: list[ChatCompletionMessageParam], message_slice: str
) -> list[ChatCompletionMessageParam]:
    """Apply message filters.

    Filters are applied in order. Filters can be either a slice string or a CEL expression. (CEL do not support slice natively)

    >>> _apply_message_slice(messages, "-100:") # take last 100 messages
    >>> _apply_message_slice(messages, "0:10") # take first 10 messages
    >>> _apply_message_slice(messages, ":0") # take no messages
    >>> _apply_message_slice(messages, ":") # take all messages, equivalent to no filter
    >>> _apply_message_slice(messages, "0:10:-1") # RARELY USED: take first 10 messages, reverse order
    """
    if re.match(r"-?\d*:-?\d*(:-?\d*)?", message_slice):
        return messages[
            slice(*[int(v) if v else None for v in message_slice.split(":")])
        ]
    raise ValueError(f"Invalid message slice: {message_slice}")


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


class Agent(BaseModel, use_attribute_docstrings=True):
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

    entry_point: str | None = None
    """The entry point for the subagents"""

    finish_point: str | None = None
    """The finish point for the subagents"""

    nodes: "dict[str, Agent]" = Field(default_factory=dict)
    """The nodes in the Agent's graph"""

    edges: list[Edge] = Field(default_factory=list)
    """The edges in the Agent's graph"""

    hooks: list[Hook] = Field(default_factory=list)
    """Hooks to execute at various points in the agent lifecycle"""

    _visited: dict[str, bool] = PrivateAttr(
        default_factory=lambda: defaultdict(lambda: False)
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

    def _get_client(self):
        return self.client or DEFAULT_CLIENT or AsyncOpenAI()

    async def _execute_hooks(
        self,
        hook_type: HookType,
        messages: list[ChatCompletionMessageParam],
        context: dict[str, Any],
        tool_name: str | None = None,
        to_agent: "Agent | None" = None,
    ):
        """Execute hooks of a specific type.

        Args:
            hook_type: The type of hook to execute (e.g., 'on_start', 'on_end')
            messages: The current messages to pass to hook tools
            context: The context variables to pass to the hook tools
            tool_name: The name of the tool being called (for on_tool_start and on_tool_end)
            to_agent: The agent being handed off to (for on_handoff)

        """
        for hook in [h for h in self.hooks if hasattr(h, hook_type)]:
            hook_name: str = getattr(hook, hook_type)
            hook_tool = CLIENT_REGISTRY.get_tool(hook_name)
            properties = hook_tool.inputSchema["properties"]
            arguments: dict[str, Any] = {}
            available = {"messages": messages, "context": context}
            if tool_name is not None:
                available["tool"] = CLIENT_REGISTRY.get_tool(tool_name)
            if to_agent is not None:
                available["from_agent"] = self.model_dump(
                    mode="json", exclude_unset=True
                )
                available["to_agent"] = to_agent.model_dump(
                    mode="json", exclude_unset=True
                )
            else:
                available["agent"] = self.model_dump(mode="json", exclude_unset=True)
            for key, value in available.items():
                if key in properties:
                    arguments |= {key: value}
            try:
                result = await CLIENT_REGISTRY.call_tool(hook_name, arguments)
                if result.structuredContent is None:
                    raise ValueError("Hook tool must return structured content")
                context |= result.structuredContent
            except Exception as e:
                logger.warning(f"Hook {hook_type} failed for {hook_name}: {e}")

    @overload
    async def _execute_tool_with_hooks(
        self,
        tool_call: ChatCompletionMessageToolCallParam,
        stream: Literal[False],
        messages: list[ChatCompletionMessageParam],
        context: dict[str, Any],
    ) -> ChatCompletion: ...

    @overload
    async def _execute_tool_with_hooks(
        self,
        tool_call: ChatCompletionMessageToolCallParam,
        stream: Literal[True],
        messages: list[ChatCompletionMessageParam],
        context: dict[str, Any],
    ) -> ChatCompletionChunk: ...

    async def _execute_tool_with_hooks(
        self,
        tool_call: ChatCompletionMessageToolCallParam,
        stream: bool,
        messages: list[ChatCompletionMessageParam],
        context: dict[str, Any],
    ) -> ChatCompletion | ChatCompletionChunk:
        """Execute a tool call with on_tool_start and on_tool_end hooks.

        Args:
            tool_call: The tool call to execute
            stream: Whether to stream the response
            messages: The current messages
            context: The context variables

        Returns:
            The result of the tool execution

        """
        tool_name = tool_call["function"]["name"]
        await self._execute_hooks("on_tool_start", messages, context, tool_name)

        try:
            result = await exec_tool_call(tool_call, stream)  # type: ignore
            await self._execute_hooks("on_tool_end", messages, context, tool_name)
            return result
        except Exception as e:
            logger.warning(f"Tool execution failed for {tool_name}: {e}")
            await self._execute_hooks("on_tool_end", messages, context, tool_name)
            raise

    @overload
    async def _create_chat_completion(
        self,
        *,
        messages: list[ChatCompletionMessageParam],
        context: dict[str, Any] | None = None,
        stream: Literal[True],
    ) -> AsyncStream[ChatCompletionChunk]: ...

    @overload
    async def _create_chat_completion(
        self,
        *,
        messages: list[ChatCompletionMessageParam],
        context: dict[str, Any] | None = None,
        stream: Literal[False] = False,
    ) -> ChatCompletion: ...

    @overload
    async def _create_chat_completion(
        self,
        *,
        messages: list[ChatCompletionMessageParam],
        context: dict[str, Any] | None = None,
        response_format: type[T] | None = None,
    ) -> ParsedChatCompletion[T]: ...

    async def _create_chat_completion(
        self,
        *,
        messages: list[ChatCompletionMessageParam],
        context: dict[str, Any] | None = None,
        response_format: type[T] | None = None,
        stream: bool = False,
    ) -> ChatCompletion | AsyncStream[ChatCompletionChunk] | ParsedChatCompletion[T]:
        """Get a chat completion for the agent.

        Args:
            messages: The messages to start the conversation with
            context: The context variables to pass to the agent
            response_format: The response type will be parsed
            stream: Whether to stream the response

        """
        message_slice: str | None = (context or {}).get("message_slice")
        if message_slice is not None:
            messages = _apply_message_slice(messages, message_slice)
        system_prompt = await self.get_system_prompt(context)
        if system_prompt is not None:
            messages = [{"role": "system", "content": system_prompt}, *messages]
        params: CompletionCreateParamsBase = {
            "messages": messages,
            "model": self.model,
        }
        logger.debug("Getting chat completion for...:", messages)

        params = self.completion_create_params | params
        if len(tools := (context or {}).get("tools", CLIENT_REGISTRY.tools)) > 0:
            params["tools"] = tools
        elif "tools" in params:
            del params["tools"]

        if response_format is None:
            return await self._get_client().chat.completions.create(
                stream=stream, **params
            )
        if stream:
            raise NotImplementedError("Streamed parsing is not supported.")
        return await self._get_client().chat.completions.parse(
            **(params | {"response_format": response_format}),  # type: ignore
        )

    async def _run_node_stream(
        self,
        *,
        node: str | None = None,
        messages: list[ChatCompletionMessageParam],
        context: dict[str, Any] | None = None,
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
            if edge.source == node or (
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

        async for chunk in join(*generators):
            yield chunk

    async def _run_stream(
        self,
        *,
        messages: list[ChatCompletionMessageParam],
        context: dict[str, Any] | None = None,
    ) -> AsyncGenerator[ChatCompletionChunk, None]:
        """Run the agent and stream the response.

        Args:
            messages: The messages to start the conversation with
            context: The context variables to pass to the agent

        """
        if context is None:
            context = {}
        init_len = len(messages)
        _messages: dict[str, ChatCompletionMessageParam] = defaultdict(
            lambda: {"role": "assistant"}
        )
        await self._execute_hooks("on_llm_start", messages, context)
        async for chunk in await self._create_chat_completion(
            messages=messages,
            context=context,
            stream=True,
        ):
            logger.info(chunk.model_dump_json(exclude_unset=True))
            _merge_chunk(_messages, chunk)
            yield chunk
            if chunk.choices[0].finish_reason is not None:
                messages.append(_messages[chunk.id])
        else:
            if len(messages) - init_len != len(_messages):
                raise ValueError("Number of messages does not match number of chunks")
        await self._execute_hooks("on_llm_end", messages, context)
        latest_message = messages[-1]
        if (
            latest_message["role"] == "assistant"
            and (tool_calls := latest_message.get("tool_calls")) is not None
        ):
            async with asyncio.TaskGroup() as tg:
                tasks = [
                    tg.create_task(
                        self._execute_tool_with_hooks(
                            tool_call, True, messages, context
                        )
                    )
                    for tool_call in tool_calls
                ]
                for future in asyncio.as_completed(tasks):
                    yield await future

        if len(self.nodes) > 0:
            async for chunk in self._run_node_stream(
                messages=messages,
                context=context,
            ):
                yield chunk

    async def _run_node(
        self,
        *,
        node: str | None = None,
        messages: list[ChatCompletionMessageParam],
        context: dict[str, Any] | None = None,
    ) -> AsyncGenerator[ChatCompletion, None]:
        """Run the node and its successors."""
        if node is None:
            node = self.entry_point
        if node is None:
            return
        agent = self.nodes[node]
        async for completion in await agent.run(
            messages=messages, stream=False, context=context
        ):
            yield completion
        self._visited[node] = True
        if node == self.finish_point:
            return
        for edge in self.edges:
            if edge.source == node or (
                isinstance(edge.source, list)
                and node in edge.source
                and all(self._visited[n] for n in edge.source)
            ):
                async for completion in self._run_node(
                    node=edge.target,
                    messages=messages,
                    context=context,
                ):
                    yield completion

    async def _run(
        self,
        *,
        messages: list[ChatCompletionMessageParam],
        context: dict[str, Any] | None = None,
    ) -> AsyncGenerator[ChatCompletion, None]:
        """Run the agent and yield ChatCompletion objects.

        Args:
            messages: The messages to start the conversation with
            context: The context variables to pass to the agent

        """
        if context is None:
            context = {}
        await self._execute_hooks("on_llm_start", messages, context)
        completion = await self._create_chat_completion(
            messages=messages,
            context=context,
            stream=False,
        )
        logger.info(completion.model_dump_json(exclude_unset=True))

        # Yield the completion object to preserve all metadata
        yield completion

        # Extract message for hook processing and tool calls
        message = cast(
            ChatCompletionAssistantMessageParam,
            completion.choices[0].message.model_dump(
                mode="json", exclude_unset=True, exclude_none=True
            ),
        )
        message["name"] = f"{self.name} ({completion.id})"
        messages.append(message)
        await self._execute_hooks("on_llm_end", messages, context)

        if (tool_calls := message.get("tool_calls")) is not None:
            # Apply hook decorator to exec_tool_calls
            async with asyncio.TaskGroup() as tg:
                tasks = [
                    tg.create_task(
                        self._execute_tool_with_hooks(
                            tool_call, False, messages, context
                        )
                    )
                    for tool_call in tool_calls
                ]
                for future in asyncio.as_completed(tasks):
                    yield await future
        if len(self.nodes) > 0:
            async for completion in self._run_node(
                messages=messages,
                context=context,
            ):
                yield completion

    async def get_system_prompt(
        self,
        context: dict[str, Any] | None = None,
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
        context: dict[str, Any] | None = None,
    ) -> AsyncGenerator[ChatCompletionChunk, None]: ...

    @overload
    async def run(
        self,
        *,
        messages: list[ChatCompletionMessageParam],
        stream: Literal[False] = False,
        context: dict[str, Any] | None = None,
    ) -> AsyncGenerator[ChatCompletion, None]: ...

    async def run(
        self,
        *,
        messages: list[ChatCompletionMessageParam],
        stream: bool = False,
        context: dict[str, Any] | None = None,
    ) -> (
        AsyncGenerator[ChatCompletion, None] | AsyncGenerator[ChatCompletionChunk, None]
    ):
        """Run the agent.

        Args:
            messages: The messages to start the conversation with
            stream: Whether to stream the response
            context: The context variables to pass to the agent
            execute_tools: Whether to execute tool calls

        """
        for name, server_params in self.mcp_servers.items():
            await CLIENT_REGISTRY.add_server(name, server_params)
        messages = deepcopy(messages)
        # context is intentionaly not deepcopied since it's mutable
        if context is None:
            context = {}
        await self._execute_hooks("on_start", messages, context)
        if stream:
            g = self._run_stream(
                messages=messages,
                context=context,
            )
        else:
            g = self._run(
                messages=messages,
                context=context,
            )
        await self._execute_hooks("on_end", messages, context)
        return g
