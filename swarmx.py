# mypy: disable-error-code="misc"
import asyncio
import copy
import inspect
import json
import logging
import warnings
from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Hashable
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from functools import wraps
from pathlib import Path
from typing import (
    Annotated,
    Any,
    AsyncIterator,
    Callable,
    Coroutine,
    Iterable,
    Literal,
    TypeAlias,
    cast,
    get_origin,
    get_type_hints,
    overload,
)

import mcp.types
import networkx as nx
import typer
from jinja2 import Template
from mcp import ClientSession, StdioServerParameters
from mcp.client.sse import sse_client
from mcp.client.stdio import stdio_client
from openai import AsyncOpenAI, AsyncStream
from openai.types import ChatModel
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionAssistantMessageParam,
    ChatCompletionChunk,
    ChatCompletionMessageParam,
    ChatCompletionMessageToolCall,
    ChatCompletionMessageToolCallParam,
    ChatCompletionToolChoiceOptionParam,
    ChatCompletionToolMessageParam,
    ChatCompletionToolParam,
)
from openai.types.chat.chat_completion_chunk import ChoiceDelta
from openai.types.chat.chat_completion_message_tool_call import Function
from openai.types.chat.completion_create_params import CompletionCreateParamsBase
from pydantic import (
    AfterValidator,
    AliasChoices,
    BaseModel,
    BeforeValidator,
    Discriminator,
    Field,
    ImportString,
    ModelWrapValidatorHandler,
    PrivateAttr,
    SecretStr,
    TypeAdapter,
    ValidationError,
    create_model,
    field_serializer,
    field_validator,
    model_serializer,
    model_validator,
)
from pydantic.json_schema import GenerateJsonSchema
from typing_extensions import Self, Unpack

warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")
logger = logging.getLogger(__name__)

__CTX_VARS_NAME__ = "context_variables"

ReturnType: TypeAlias = "str | Node | dict[str, Any] | Result"
AgentFunction: TypeAlias = Callable[..., ReturnType | Coroutine[Any, Any, ReturnType]]


def merge_chunk(
    message: ChatCompletionAssistantMessageParam, delta: ChoiceDelta
) -> None:
    content = message.get("content") or ""
    if isinstance(content, str):
        message["content"] = content + (delta.content or "")
    else:
        message["content"] = list(content) + [
            {"type": "text", "text": delta.content or ""}
        ]

    if delta.refusal is not None:
        message["refusal"] = (message.get("refusal") or "") + delta.refusal

    if delta.tool_calls is not None:
        tool_calls = {i: call for i, call in enumerate(message.get("tool_calls") or [])}
        for call in delta.tool_calls:
            function = call.function
            tool_call = tool_calls.get(
                call.index
            ) or ChatCompletionMessageToolCallParam(
                {
                    "id": call.id or "",
                    "function": {"arguments": "", "name": ""},
                    "type": "function",
                }
            )
            tool_call["id"] = call.id or tool_call["id"]
            tool_call["function"]["arguments"] += (
                function.arguments or "" if function else ""
            )
            tool_call["function"]["name"] = function.name or "" if function else ""
            tool_calls[call.index] = tool_call
        message["tool_calls"] = [
            tool_call for _, tool_call in sorted(tool_calls.items())
        ]


def does_function_need_context(func: AgentFunction) -> bool:
    signature = inspect.signature(func)
    return __CTX_VARS_NAME__ in signature.parameters


def function_to_json(func: AgentFunction) -> ChatCompletionToolParam:
    signature = inspect.signature(func)
    field_definitions = {}
    for param in signature.parameters.values():
        if param.name == __CTX_VARS_NAME__:
            continue
        field_definitions[param.name] = (
            param.annotation if param.annotation is not param.empty else str,
            param.default if param.default is not param.empty else ...,
        )
    arguments_model = create_model(func.__name__, **field_definitions)  # type: ignore[call-overload]
    function: ChatCompletionToolParam = {
        "type": "function",
        "function": {
            "name": func.__name__,
            "parameters": {
                k: v
                for k, v in arguments_model.model_json_schema(
                    schema_generator=SwarmXGenerateJsonSchema
                ).items()
                if k != "title"
            },
        },
    }
    if func.__doc__:
        function["function"]["description"] = func.__doc__
    return function


def validate_tool(tool: object) -> ChatCompletionToolParam:
    e = TypeError(
        "Agent function return type must be str, Agent, dict[str, Any], or Result"
    )
    match tool:
        case dict():
            tool = TypeAdapter(ChatCompletionToolParam).validate_python(tool)
            return tool
        case tool if callable(tool):
            annotation = get_type_hints(tool)
            if (return_anno := annotation.get("return")) is None:
                warnings.warn(
                    "Agent function return type is not annotated, assuming str. "
                    "This will be an error in a future version.",
                    FutureWarning,
                )
            if return_anno not in [str, Agent, dict[str, Any], Result, None]:
                raise e
            TOOL_REGISTRY.add_function(tool)
            return TOOL_REGISTRY.functions[getattr(tool, "__name__", str(tool))]
        case str():
            return validate_tool(TypeAdapter(ImportString).validate_python(tool))
        case _:
            raise e


def validate_tools(tools: list[object]) -> list[AgentFunction]:
    return [validate_tool(tool) for tool in tools]


def check_instructions(
    instructions: str | object,
) -> str | Callable[[dict[str, Any]], str]:
    if isinstance(instructions, str):
        return instructions
    err = ValueError(
        f"Instructions should be a string or a callable takes {__CTX_VARS_NAME__} and returns string"
    )
    if callable(instructions):
        sig = inspect.signature(instructions)
        if __CTX_VARS_NAME__ not in sig.parameters:
            raise err
        anno = sig.parameters[__CTX_VARS_NAME__].annotation
        if not (
            anno is inspect.Signature.empty or anno is dict or get_origin(anno) is dict
        ):
            raise err
        return cast(Callable[[dict[str, Any]], str], instructions)
    raise err


class SwarmXGenerateJsonSchema(GenerateJsonSchema):
    def field_title_should_be_set(self, schema) -> bool:
        return False


@dataclass
class ToolRegistry:
    functions: dict[str, ChatCompletionToolParam] = field(default_factory=dict)
    mcp_tools: dict[tuple[str, str], ChatCompletionToolParam] = field(
        default_factory=dict
    )
    exit_stack: AsyncExitStack = field(default_factory=AsyncExitStack)
    mcp_clients: dict[str, ClientSession] = field(default_factory=dict)
    _tools: dict[str, AgentFunction | str] = field(default_factory=dict)

    @property
    def tools(self) -> list[ChatCompletionToolParam]:
        return list(self.mcp_tools.values()) + list(self.functions.values())

    async def call_tool(
        self,
        name: str,
        arguments: dict[str, Any],
        context_variables: dict[str, Any] | None = None,
    ) -> "Result":
        callable_func = self._tools.get(name)
        if callable_func is None:
            raise ValueError(f"Tool {name} not found")
        if isinstance(callable_func, str):
            result = await self.mcp_clients[callable_func].call_tool(name, arguments)
            return handle_function_result(result)
        if does_function_need_context(callable_func):
            arguments[__CTX_VARS_NAME__] = context_variables or {}
        result = callable_func(**arguments)
        if inspect.isawaitable(result):
            result = await result
        return handle_function_result(result)

    async def add_mcp_server(
        self, name: str, server_params: StdioServerParameters | str
    ):
        if name in self.mcp_clients:
            return
        read_stream, write_stream = await self.exit_stack.enter_async_context(
            sse_client(server_params)
            if isinstance(server_params, str)
            else stdio_client(server_params)
        )
        client = cast(
            ClientSession,
            await self.exit_stack.enter_async_context(
                ClientSession(read_stream, write_stream)
            ),
        )
        await client.initialize()
        self.mcp_clients[name] = client
        for tool in (await client.list_tools()).tools:
            self._tools[tool.name] = name
            function_schema = tool.model_dump(exclude_none=True)
            function_schema["parameters"] = function_schema.pop("inputSchema")
            self.mcp_tools[name, tool.name] = ChatCompletionToolParam(
                type="function",
                function=function_schema,
            )

    def add_function(self, func: AgentFunction):
        func_json = function_to_json(func)
        name = func_json["function"]["name"]
        self.functions[name] = func_json
        self._tools[name] = func

    async def close(self):
        await self.exit_stack.aclose()


TOOL_REGISTRY = ToolRegistry()
DEFAULT_CLIENT = AsyncOpenAI()


class ContextVariables(BaseModel):
    type: Literal["context_variables"] = "context_variables"
    context_variables: dict[str, Any]


class Node(BaseModel, ABC):
    client: AsyncOpenAI | None = None
    """The client to use for the node"""

    @field_validator("client", mode="plain")
    def validate_client(cls, v: Any) -> AsyncOpenAI | None:
        if v is None:
            return None
        if isinstance(v, AsyncOpenAI):
            return v
        return AsyncOpenAI(**v)

    @field_serializer("client", mode="plain")
    def serialize_client(self, v: AsyncOpenAI | None) -> dict[str, Any] | None:
        if v is None:
            return None
        client = {}
        if str(v.base_url) != "https://api.openai.com/v1":
            client["base_url"] = str(v.base_url)
        for key in (
            "organization",
            "project",
            "websocket_base_url",
        ):
            if getattr(v, key, None) is not None:
                client[key] = getattr(v, key)
        return client

    @overload
    async def run(
        self,
        *,
        client: AsyncOpenAI | None = None,
        stream: Literal[True],
        context_variables: dict[str, Any] | None = None,
        max_turns: int | None = None,
        execute_tools: bool = True,
        **kwargs: Unpack[CompletionCreateParamsBase],
    ) -> AsyncIterator[
        ChatCompletionChunk | ChatCompletionMessageParam | "Node" | ContextVariables
    ]: ...

    @overload
    async def run(
        self,
        *,
        client: AsyncOpenAI | None = None,
        stream: Literal[False] = False,
        context_variables: dict[str, Any] | None = None,
        max_turns: int | None = None,
        execute_tools: bool = True,
        **kwargs: Unpack[CompletionCreateParamsBase],
    ) -> "Response": ...

    @abstractmethod
    async def run(
        self,
        *,
        client: AsyncOpenAI | None = None,
        stream: bool = False,
        context_variables: dict[str, Any] | None = None,
        max_turns: int | None = None,
        execute_tools: bool = True,
        **kwargs: Unpack[CompletionCreateParamsBase],
    ) -> (
        AsyncIterator[
            ChatCompletionChunk | ChatCompletionMessageParam | "Node" | ContextVariables
        ]
        | "Response"
    ): ...


class Response(BaseModel):
    messages: list[ChatCompletionMessageParam] = []
    agent: Node | None = None
    context_variables: dict[str, Any] = {}


class Result(mcp.types.CallToolResult):
    """
    Encapsulates the possible return values for an agent function.

    Attributes:
        agent (Agent): The agent instance, if applicable.
    """

    agent: Node | None = None

    @property
    def value(self) -> str:
        return "".join([c.text for c in self.content if c.type == "text"])


class Agent(Node):
    type: Literal["agent"] = "agent"

    name: str = "Agent"
    """The agent's name"""

    model: ChatModel | str = "gpt-4o"
    """The default model to use for the agent."""

    instructions: Annotated[ImportString | str, AfterValidator(check_instructions)] = (
        "You are a helpful agent."
    )
    """Agent's instructions, could be a Jinja2 template"""

    tools: Annotated[
        list[ChatCompletionToolParam],
        BeforeValidator(validate_tools),
    ] = Field(
        default_factory=list,
        validation_alias=AliasChoices("tools", "functions"),
    )
    """The tools available to the agent"""

    tool_choice: ChatCompletionToolChoiceOptionParam | None = None
    """The tool choice option for the agent"""

    parallel_tool_calls: bool = False
    """Whether to make tool calls in parallel"""

    base_url: str | None = None
    """The base URL for the OpenAI API"""

    api_key: SecretStr | None = None
    """The API key for the OpenAI API"""

    _tool_registry: dict[str, AgentFunction] = PrivateAttr(default_factory=dict)

    def _with_instructions(
        self,
        *,
        messages: Iterable[ChatCompletionMessageParam],
        context_variables: dict[str, Any] | None = None,
    ) -> list[ChatCompletionMessageParam]:
        content = (
            Template(self.instructions).render
            if isinstance(self.instructions, str)
            else cast(Callable[[dict[str, Any]], str], self.instructions)
        )(context_variables or {})
        return [{"role": "system", "content": content}, *messages]

    def preprocess(
        self,
        *,
        context_variables: dict[str, Any] | None = None,
        **kwargs: Unpack[CompletionCreateParamsBase],
    ) -> CompletionCreateParamsBase:
        """Preprocess the agent's messages and context variables."""
        model = kwargs.get("model") or self.model
        messages = self._with_instructions(
            messages=kwargs["messages"],
            context_variables=context_variables,
        )
        for message in messages:
            if message["role"] == "assistant" and "tool_calls" not in message:
                content = message.get("content")
                if isinstance(content, str) and "</think>" in content:
                    message["content"] = content.split("</think>")[1]
        return kwargs | {"messages": messages, "model": model}

    @overload
    async def get_chat_completion(
        self,
        client: AsyncOpenAI | None,
        context_variables: dict,
        stream: Literal[True],
        **kwargs: Unpack[CompletionCreateParamsBase],
    ) -> AsyncStream[ChatCompletionChunk]: ...

    @overload
    async def get_chat_completion(
        self,
        client: AsyncOpenAI | None,
        context_variables: dict,
        stream: Literal[False] = False,
        **kwargs: Unpack[CompletionCreateParamsBase],
    ) -> ChatCompletion: ...

    async def get_chat_completion(
        self,
        client: AsyncOpenAI | None,
        context_variables: dict,
        stream: bool = False,
        **kwargs: Unpack[CompletionCreateParamsBase],
    ) -> ChatCompletion | AsyncStream[ChatCompletionChunk]:
        create_params = self.preprocess(context_variables=context_variables, **kwargs)
        messages = create_params["messages"]
        logger.debug("Getting chat completion for...:", messages)

        # hide context_variables from model
        tools = self.model_dump(mode="json", exclude={"api_key"})["tools"]
        for tool in tools:
            params: dict[str, Any] = tool["function"].get(
                "parameters", {"properties": {}}
            )
            params["properties"].pop(__CTX_VARS_NAME__, None)
            if __CTX_VARS_NAME__ in params.get("required", []):
                params["required"].remove(__CTX_VARS_NAME__)

        if len(tools) > 0:
            create_params["tools"] = tools
            if self.tool_choice:
                create_params["tool_choice"] = self.tool_choice
            create_params["parallel_tool_calls"] = self.parallel_tool_calls
        else:
            create_params.pop("tools", None)
            create_params.pop("tool_choice", None)
            create_params.pop("parallel_tool_calls", None)

        return await (self.client or client or DEFAULT_CLIENT).chat.completions.create(
            stream=stream, **create_params
        )

    async def _run_and_stream(
        self,
        client: AsyncOpenAI | None,
        context_variables: dict[str, Any],
        execute_tools: bool,
        **kwargs: Unpack[CompletionCreateParamsBase],
    ):
        messages = list(kwargs["messages"])
        message: ChatCompletionMessageParam = {
            "content": "",
            "name": self.name,
            "role": "assistant",
            "tool_calls": defaultdict(
                lambda: {
                    "function": {"arguments": "", "name": ""},
                    "id": "",
                    "type": "",
                },
            ),
        }
        completion = await self.get_chat_completion(
            client=client, context_variables=context_variables, stream=True, **kwargs
        )
        reasoning = False
        async for chunk in completion:
            yield chunk
            delta = chunk.choices[0].delta
            # deepseek-reasoner would have extra "reasoning_content" field, we would
            # wrap it in <think></think> for further handling.
            if isinstance(content := getattr(delta, "reasoning_content", None), str):
                delta.content = ("<think>\n" if not reasoning else "") + content
                reasoning = True
            if reasoning and content is None:
                delta.content = "</think>\n" + (delta.content or "")
                reasoning = False
            merge_chunk(message, delta)
        message["tool_calls"] = list(message.get("tool_calls", {}).values())  # type: ignore
        if not message["tool_calls"]:
            message.pop("tool_calls", None)
        logger.debug("Received completion:", message)
        messages.append(message)
        yield message
        if not message.get("tool_calls") or not execute_tools:
            logger.debug("Ending turn.")
            return

        # convert tool_calls to objects
        tool_calls = []
        for tool_call in message["tool_calls"]:
            function = Function(
                arguments=tool_call["function"]["arguments"],
                name=tool_call["function"]["name"],
            )
            tool_call_object = ChatCompletionMessageToolCall(
                id=tool_call["id"], function=function, type=tool_call["type"]
            )
            tool_calls.append(tool_call_object)

        # handle function calls, updating context_variables, and switching agents
        partial_response = await handle_tool_calls(tool_calls)
        messages.extend(partial_response.messages)
        for message in partial_response.messages:
            yield message
        if partial_response.context_variables:
            context_variables |= partial_response.context_variables
            yield ContextVariables(context_variables=context_variables)
        if partial_response.agent:
            yield partial_response.agent

    @overload
    async def run(
        self,
        *,
        client: AsyncOpenAI | None = None,
        stream: Literal[True],
        context_variables: dict[str, Any] | None = None,
        max_turns: int | None = None,
        execute_tools: bool = True,
        **kwargs: Unpack[CompletionCreateParamsBase],
    ) -> AsyncIterator[
        ChatCompletionChunk | ChatCompletionMessageParam | Node | ContextVariables
    ]: ...

    @overload
    async def run(
        self,
        *,
        client: AsyncOpenAI | None = None,
        stream: Literal[False] = False,
        context_variables: dict[str, Any] | None = None,
        max_turns: int | None = None,
        execute_tools: bool = True,
        **kwargs: Unpack[CompletionCreateParamsBase],
    ) -> "Response": ...

    async def run(
        self,
        *,
        client: AsyncOpenAI | None = None,
        stream: bool = False,
        context_variables: dict[str, Any] | None = None,
        max_turns: int | None = None,
        execute_tools: bool = True,
        **kwargs: Unpack[CompletionCreateParamsBase],
    ) -> (
        "Response"
        | AsyncIterator[
            ChatCompletionChunk | ChatCompletionMessageParam | Node | ContextVariables
        ]
    ):
        context_variables = copy.deepcopy(context_variables or {})
        if stream:
            return self._run_and_stream(
                client=client,
                context_variables=context_variables,
                execute_tools=execute_tools,
                **kwargs,
            )
        messages = copy.deepcopy(list(kwargs["messages"]))
        init_len = len(messages)
        completion = await self.get_chat_completion(
            client=client,
            context_variables=context_variables,
            stream=False,
            **kwargs,
        )
        message = completion.choices[0].message
        logger.debug("Received completion:", message)
        m = cast(
            ChatCompletionAssistantMessageParam,
            message.model_dump(mode="json", exclude_none=True),
        )
        m["name"] = self.name
        messages.append(m)
        if message.tool_calls and execute_tools:
            partial_response = await handle_tool_calls(message.tool_calls)
            messages.extend(partial_response.messages)
            context_variables |= partial_response.context_variables
            agent = partial_response.agent or self
        else:
            agent = self
        return Response(
            messages=messages[init_len:],
            agent=agent,
            context_variables=context_variables,
        )


async def handle_tool_calls(
    tool_calls: list[ChatCompletionMessageToolCall],
) -> Response:
    partial_response = Response(messages=[], agent=None, context_variables={})

    for tool_call in tool_calls:
        name = tool_call.function.name
        # handle missing tool case, skip to next tool
        if name not in TOOL_REGISTRY._tools:
            logger.debug(f"Tool {name} not found in function map.")
            partial_response.messages.append(
                ChatCompletionToolMessageParam(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": f"Error: Tool {name} not found.",
                    }
                )
            )
            continue
        result = await TOOL_REGISTRY.call_tool(
            name, json.loads(tool_call.function.arguments)
        )
        partial_response.messages.append(
            ChatCompletionToolMessageParam(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result.value,
                }
            )
        )
        partial_response.context_variables.update(result.meta or {})
        if result.agent:
            partial_response.agent = result.agent

    return partial_response


def handle_function_result(result: Any) -> Result:
    match result:
        case Result() as result:
            return result

        case Agent() as agent:
            return Result(content=[], agent=agent)

        case dict():
            return Result(_meta=result, content=[])

        case mcp.types.CallToolResult():
            return Result.model_validate(result.model_dump())

        case _:
            return Result(
                content=[mcp.types.TextContent(type="text", text=str(result))]
            )


class Swarm(Node, nx.DiGraph):
    type: Literal["swarm"] = "swarm"
    mcp_servers: dict[str, StdioServerParameters | str] | None = Field(
        default=None,
        alias="mcpServers",
    )

    @model_validator(mode="wrap")
    @classmethod
    def node_link_validator(
        cls, data: Any, handler: ModelWrapValidatorHandler[Self]
    ) -> Self:
        if not isinstance(data, dict):
            raise ValidationError("Swarm must be a dictionary")
        swarm = handler(data)
        for node in data.get("nodes", []):
            swarm.add_node(
                node["id"],
                TypeAdapter(
                    Annotated[Agent | Swarm, Discriminator("type")]
                ).validate_python(node),
            )
        for edge in data.get("edges", []):
            edge = copy.deepcopy(edge)
            if not isinstance(edge, dict):
                raise ValidationError("Edge must be a dictionary")
            if (source := edge.pop("source", None)) is None or (
                target := edge.pop("target", None)
            ) is None:
                raise ValidationError("Edge must have source and target")
            swarm.add_edge(source, target, **edge)
        return swarm

    @model_serializer()
    def serialize_swarm(self):
        swarm = {
            "nodes": [
                {"id": node} | data.model_dump(mode="json")
                for node, data in self.nodes(data=True)
            ],
            "edges": [
                {
                    "source": u if isinstance(u, int) else str(u),
                    "target": v if isinstance(v, int) else str(v),
                }
                | data
                for u, v, data in self.edges(data=True)
            ],
        }
        if self.mcp_servers:
            swarm["mcpServers"] = {
                name: server_params.model_dump()
                if isinstance(server_params, StdioServerParameters)
                else server_params
                for name, server_params in self.mcp_servers.items()
            }
        return swarm

    @property
    def root(self) -> Any:
        roots = [node for node, degree in self.in_degree if degree == 0]
        if len(roots) != 1:
            raise ValueError("Swarm must have exactly one root node")
        return roots[0]

    def model_post_init(self, __context: Any) -> None:
        self._node = self.node_dict_factory()
        self._adj = self.adjlist_outer_dict_factory()
        self._pred = self.adjlist_outer_dict_factory()
        self.__networkx_cache__: dict = {}

    def add_node(self, node_for_adding: Hashable, node: Node):
        if node_for_adding not in self._succ:
            self._succ[node_for_adding] = self.adjlist_inner_dict_factory()
            self._pred[node_for_adding] = self.adjlist_inner_dict_factory()
        self._node[node_for_adding] = node
        nx._clear_cache(self)

    def add_edge(self, u: Hashable, v: Hashable, **attr: Any):
        if u not in self._node:
            raise ValueError(f"Node {u} not found")
        if v not in self._node:
            raise ValueError(f"Node {v} not found")
        super().add_edge(u, v, **attr)

    @property
    def _next_node(self) -> int:
        return max([i for i in self.nodes if isinstance(i, int)] + [-1]) + 1

    async def _run_and_stream(
        self,
        client: AsyncOpenAI | None = None,
        context_variables: dict[str, Any] | None = None,
        max_turns: int | None = None,
        execute_tools: bool = True,
        **kwargs: Unpack[CompletionCreateParamsBase],
    ):
        node = self.root
        agent = cast(Node, self.nodes[node])
        context_variables = copy.deepcopy(context_variables or {})
        messages = [m for m in copy.deepcopy(kwargs["messages"])]
        init_len = len(messages)

        while max_turns is None or len(messages) - init_len < max_turns:
            response = await agent.run(
                client=self.client or client or DEFAULT_CLIENT,
                stream=True,
                context_variables=context_variables,
                execute_tools=execute_tools,
                **kwargs,
            )
            async for message in response:
                yield message
                match message:
                    case Node():
                        next_node = self._next_node
                        self.add_node(next_node, message)
                        self.add_edge(node, next_node)
                        node, agent = next_node, message
                    case ContextVariables():
                        context_variables |= message.context_variables
                    case ChatCompletionChunk():
                        ...
                    case _:
                        messages = [*messages, message]
                        kwargs["messages"] = messages

    @overload
    async def run(
        self,
        *,
        client: AsyncOpenAI | None = None,
        stream: Literal[True],
        context_variables: dict[str, Any] | None = None,
        max_turns: int | None = None,
        execute_tools: bool = True,
        **kwargs: Unpack[CompletionCreateParamsBase],
    ) -> AsyncIterator[
        ChatCompletionChunk | ChatCompletionMessageParam | Node | ContextVariables
    ]: ...

    @overload
    async def run(
        self,
        *,
        client: AsyncOpenAI | None = None,
        stream: Literal[False] = False,
        context_variables: dict[str, Any] | None = None,
        max_turns: int | None = None,
        execute_tools: bool = True,
        **kwargs: Unpack[CompletionCreateParamsBase],
    ) -> Response: ...

    async def run(
        self,
        *,
        client: AsyncOpenAI | None = None,
        stream: bool = False,
        context_variables: dict[str, Any] | None = None,
        max_turns: int | None = None,
        execute_tools: bool = True,
        **kwargs: Unpack[CompletionCreateParamsBase],
    ) -> (
        AsyncIterator[
            ChatCompletionChunk | ChatCompletionMessageParam | Node | ContextVariables
        ]
        | Response
    ):
        for name, server_params in (self.mcp_servers or {}).items():
            await TOOL_REGISTRY.add_mcp_server(name, server_params)
        if stream:
            return self._run_and_stream(
                client=self.client or client or DEFAULT_CLIENT,
                context_variables=context_variables,
                max_turns=max_turns,
                execute_tools=execute_tools,
                **kwargs,
            )
        node = self.root
        agent = cast(Node, self.nodes[node])
        context_variables = copy.deepcopy(context_variables or {})
        messages = list(copy.deepcopy(kwargs["messages"]))
        init_len = len(messages)

        while max_turns is None or len(messages) - init_len < max_turns:
            # get completion with current history, agent
            response = await agent.run(
                client=self.client or client or DEFAULT_CLIENT,
                stream=stream,
                context_variables=context_variables,
                execute_tools=execute_tools,
                **kwargs,
            )
            # dump response to json avoiding pydantic's ValidatorIterator
            messages = [*messages, *json.loads(response.model_dump_json())["messages"]]
            # add agent and edge if exists
            if response.agent:
                next_node = self._next_node
                self.add_node(next_node, response.agent)
                self.add_edge(node, next_node)
                node, agent = next_node, response.agent
            if response.context_variables:
                context_variables |= response.context_variables
            last_message = response.messages[-1]
            if (
                last_message["role"] == "assistant"
                and not last_message.get("tool_calls")
                or not execute_tools
            ):
                break
            kwargs["messages"] = messages

        return Response(
            messages=messages[init_len:],
            agent=agent,
            context_variables=context_variables,
        )


async def main(
    *,
    model: Annotated[
        str, typer.Option("--model", "-m", help="The model to use for the agent")
    ] = "gpt-4o",
    file: Annotated[
        Path | None,
        typer.Option(
            "--file",
            "-f",
            exists=True,
            help="The path to the swarmx file (JSON file containing `mcpServers` and `agent`)",
        ),
    ] = None,
    output: Annotated[
        Path | None,
        typer.Option(
            "--output",
            "-o",
            writable=True,
            help="The path to the output file to save the conversation",
        ),
    ] = None,
    verbose: Annotated[
        bool,
        typer.Option(
            "--verbose/--quiet", "-v/-q", help="Print the data sent to the model"
        ),
    ] = False,
):
    """SwarmX Command Line Interface."""

    if file is None:
        data = {}
    else:
        data = json.loads(file.read_text())
    agent: Node = Agent.model_validate(data.pop("agent", {}))
    client = Swarm.model_validate(data)
    messages: list[ChatCompletionMessageParam] = []
    context_variables: dict[str, Any] = data.pop(__CTX_VARS_NAME__, {})
    while True:
        try:
            user_prompt = typer.prompt(">>>", prompt_suffix=" ")
            messages.append(
                {
                    "role": "user",
                    "content": user_prompt,
                }
            )
            async for chunk in await client.run(
                model=model,
                messages=messages,
                stream=True,
                context_variables=context_variables,
            ):
                # we would not support coloring ollama's deepseek-r1 because it couldn't
                # produce <think> xml tag stably
                match chunk:
                    case ChatCompletionChunk():
                        delta = chunk.choices[0].delta
                        if delta.content is not None:
                            typer.echo(delta.content, nl=False)
                        if isinstance(
                            c := getattr(delta, "reasoning_content", None), str
                        ):
                            typer.secho(c, nl=False, fg="green")
                        if delta.refusal is not None:
                            typer.secho(delta.refusal, nl=False, err=True, fg="purple")
                        if chunk.choices[0].finish_reason is not None:
                            typer.echo()
                    case Node():
                        agent = chunk
                        if verbose:
                            typer.secho(f"agent: {agent.model_dump_json()}", fg="cyan")
                    case ContextVariables():
                        context_variables = chunk.context_variables
                        if verbose:
                            typer.secho(
                                f"context: {json.dumps(context_variables)}", fg="gray"
                            )
                    case _ as message:
                        messages.append(message)
                        if verbose and not (
                            message["role"] == "assistant"
                            and message.get("name") == getattr(agent, "name", None)
                        ):
                            typer.secho(f"data: {json.dumps(message)}", fg="yellow")
        except KeyboardInterrupt:
            break
        except Exception as e:
            messages.append(
                {
                    "role": "assistant",
                    "refusal": f"{e}",
                }
            )
            typer.secho(f"{e}", err=True, fg="red")
            break
    if output is not None:
        output.write_text(json.dumps(messages, indent=2, ensure_ascii=False))
    await TOOL_REGISTRY.close()


def repl():
    """SwarmX REPL wrapper"""

    @wraps(main)
    def repl_main(*args, **kwargs):
        return asyncio.run(main(*args, **kwargs))

    typer.run(repl_main)


if __name__ == "__main__":
    repl()
