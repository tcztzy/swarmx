import asyncio
import copy
import inspect
import json
import logging
import secrets
import sys
import warnings
from collections import defaultdict
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    Annotated,
    Any,
    AsyncIterator,
    Callable,
    Coroutine,
    Hashable,
    Iterable,
    Literal,
    Self,
    TypeAlias,
    TypedDict,
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
from openai.types.chat.chat_completion import ChatCompletion
from openai.types.chat.chat_completion_assistant_message_param import (
    ChatCompletionAssistantMessageParam,
)
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk, ChoiceDelta
from openai.types.chat.chat_completion_content_part_image_param import (
    ChatCompletionContentPartImageParam,
)
from openai.types.chat.chat_completion_content_part_param import (
    ChatCompletionContentPartParam,
    File,
)
from openai.types.chat.chat_completion_content_part_text_param import (
    ChatCompletionContentPartTextParam,
)
from openai.types.chat.chat_completion_message_param import (
    ChatCompletionMessageParam as _ChatCompletionMessageParam,
)
from openai.types.chat.chat_completion_message_tool_call_param import (
    ChatCompletionMessageToolCallParam,
)
from openai.types.chat.chat_completion_tool_param import ChatCompletionToolParam
from openai.types.chat.completion_create_params import CompletionCreateParamsBase
from openai.types.chat_model import ChatModel
from pydantic import (
    AfterValidator,
    AliasChoices,
    BaseModel,
    BeforeValidator,
    Discriminator,
    Field,
    ImportString,
    ModelWrapValidatorHandler,
    TypeAdapter,
    ValidationError,
    create_model,
    field_serializer,
    field_validator,
    model_serializer,
    model_validator,
)
from pydantic.json_schema import GenerateJsonSchema

# SECTION 1: Backports or compatibility code
# Pydantic requires Python 3.12's typing.Required, TypedDict
PY_312 = sys.version_info >= (3, 12)

if PY_312:
    from typing import Required, TypedDict
else:
    from typing_extensions import Required, TypedDict

warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")
logger = logging.getLogger(__name__)


class SwarmXGenerateJsonSchema(GenerateJsonSchema):
    """Remove the title field from the JSON schema."""

    def field_title_should_be_set(self, schema) -> bool:
        return False


class ReasoningChatCompletionAssistantMessageParam(ChatCompletionAssistantMessageParam):
    """Add reasoning_content to the message."""

    reasoning_content: str | Iterable[ChatCompletionContentPartTextParam]


class AgentNodeData(TypedDict, total=False):
    type: Required[Literal["agent"]]  # type: ignore
    agent: Required["Agent"]  # type: ignore
    executed: bool


class SwarmNodeData(TypedDict, total=False):
    type: Required[Literal["swarm"]]  # type: ignore
    swarm: Required["Swarm"]  # type: ignore
    executed: bool


# SECTION 2: Constants and type aliases
__CTX_VARS_NAME__ = "context_variables"
DEFAULT_CLIENT = AsyncOpenAI()
RANDOM_STRING_CHARS = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"

# Union order matters, see https://github.com/pydantic/pydantic/discussions/11554
ChatCompletionMessageParam: TypeAlias = (
    ReasoningChatCompletionAssistantMessageParam | _ChatCompletionMessageParam
)
Node: TypeAlias = "Agent | Swarm"
NodeData = Annotated[AgentNodeData | SwarmNodeData, Discriminator("type")]
ReturnType: TypeAlias = "str | Node | dict[str, Any] | Result"
AgentFunction: TypeAlias = Callable[..., ReturnType | Coroutine[Any, Any, ReturnType]]


def get_random_string(length, allowed_chars=RANDOM_STRING_CHARS):
    """
    Return a securely generated random string.

    The bit length of the returned value can be calculated with the formula:
        log_2(len(allowed_chars)^length)

    For example, with default `allowed_chars` (26+26+10), this gives:
      * length: 12, bit length =~ 71 bits
      * length: 22, bit length =~ 131 bits
    """
    return "".join(secrets.choice(allowed_chars) for i in range(length))


# SECTION 3: Helper functions
def merge_chunk(message: ChatCompletionMessageParam, delta: ChoiceDelta) -> None:
    if delta.role:
        message["role"] = delta.role
    content = message.get("content") or ""
    if isinstance(content, str):
        message["content"] = content + (delta.content or "")
    else:
        message["content"] = [*content, {"type": "text", "text": delta.content or ""}]

    # Handle reasoning_content
    if hasattr(delta, "reasoning_content"):
        reasoning_content = message.get("reasoning_content") or ""
        if isinstance(reasoning_content, str):
            message["reasoning_content"] = reasoning_content + (
                delta.reasoning_content or ""
            )
        else:
            message["reasoning_content"] = list(reasoning_content) + [
                {"type": "text", "text": delta.reasoning_content or ""}
            ]

    if delta.refusal is not None:
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
            tool_call["id"] = call.id or tool_call["id"]
            tool_call["function"]["arguments"] += (
                function.arguments or "" if function else ""
            )
            tool_call["function"]["name"] += function.name or "" if function else ""
            tool_calls[call.index] = tool_call
        message["tool_calls"] = tool_calls


def merge_chunks(
    chunks: list[ChatCompletionChunk],
) -> list[ChatCompletionMessageParam]:
    messages: dict[str, ChatCompletionMessageParam] = defaultdict(
        lambda: {
            "content": "",
            "role": "",
        }
    )
    for chunk in chunks:
        message = messages[chunk.id]
        delta = chunk.choices[0].delta
        merge_chunk(message, delta)
        if not message.get("tool_calls"):
            message.pop("tool_calls", None)
        logger.debug("Received completion:", message)
    for message in messages.values():
        if message.get("tool_calls"):
            # dict to list
            tool_calls: dict[int, ChatCompletionMessageToolCallParam] = message.pop(
                "tool_calls"
            )
            # assert no gap in tool_calls keys
            for i, index in enumerate(sorted(tool_calls)):
                if i != index:
                    raise ValueError(f"Tool call index {index} is out of order")
            message["tool_calls"] = [tool_calls[i] for i in sorted(tool_calls)]
    return list(messages.values())


def does_function_need_context(func) -> bool:
    signature = inspect.signature(func)
    return __CTX_VARS_NAME__ in signature.parameters


def function_to_json(func: Any) -> ChatCompletionToolParam:
    if not callable(func):
        raise ValueError("Function is not callable")
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


def _image_content_to_url(
    content: mcp.types.ImageContent,
) -> ChatCompletionContentPartImageParam:
    return {
        "type": "image_url",
        "image_url": {
            "url": f"data:{content.mimeType};base64,{content.data}",
        },
    }


def _resource_to_file(
    resource: mcp.types.EmbeddedResource,
) -> ChatCompletionContentPartTextParam | File:
    match resource.resource:
        case mcp.types.TextResourceContents() as c:
            return {
                "type": "text",
                "text": c.text,
            }
        case mcp.types.BlobResourceContents() as c:
            return {
                "type": "file",
                "file": {
                    "file_data": f"data:{c.mimeType};base64,{c.blob}",
                },
            }
        case _:
            raise ValueError(f"Unknown resource type: {type(resource)}")


def _mcp_call_tool_result_to_content(
    result: mcp.types.CallToolResult,
) -> list[ChatCompletionContentPartParam]:
    content: list[ChatCompletionContentPartParam] = []
    for chunk in result.content:
        match chunk.type:
            case "text":
                content.append(chunk.model_dump(exclude={"annotations"}))
            case "image":
                content.append(_image_content_to_url(chunk))
            case "resource":
                content.append(_resource_to_file(chunk))
    return content


def check_instructions(
    instructions: str | object,
) -> str | Callable[..., str]:
    if isinstance(instructions, str):
        return instructions
    err = ValueError(
        f"Instructions should be a string or a callable that either has only one "
        f"Mapping[str, Any] parameter named {__CTX_VARS_NAME__} or has multiple "
        f"parameters excluding {__CTX_VARS_NAME__}"
    )
    if callable(instructions):
        sig = inspect.signature(instructions)
        params = sig.parameters

        # Case 1: Only one parameter named __CTX_VARS_NAME__
        if len(params) == 1 and __CTX_VARS_NAME__ in params:
            anno = params[__CTX_VARS_NAME__].annotation
            if not (
                anno is inspect.Signature.empty
                or anno is dict
                or get_origin(anno) is dict
            ):
                raise err
            return cast(Callable[[dict[str, Any]], str], instructions)

        # Case 2: Multiple parameters but none named __CTX_VARS_NAME__
        if __CTX_VARS_NAME__ not in params:
            return cast(Callable[..., str], instructions)

        raise err
    raise err


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


def validate_tools(tools: list[object]) -> list[Callable]:
    return [validate_tool(tool) for tool in tools]


# SECTION 4: Tool registry
@dataclass
class ToolRegistry:
    functions: dict[str, ChatCompletionToolParam] = field(default_factory=dict)
    mcp_tools: dict[tuple[str, str], ChatCompletionToolParam] = field(
        default_factory=dict
    )
    exit_stack: AsyncExitStack = field(default_factory=AsyncExitStack)
    mcp_clients: dict[str, ClientSession] = field(default_factory=dict)
    _tools: dict[str, "AgentFunction | str"] = field(default_factory=dict)

    @property
    def tools(self) -> dict[str, ChatCompletionToolParam]:
        return {
            **self.functions,
            **{tool_name: tool for (_, tool_name), tool in self.mcp_tools.items()},
        }

    async def call_tool(
        self,
        name: str,
        arguments: dict[str, Any],
        context_variables: dict[str, Any] | None = None,
    ):
        callable_func = self._tools.get(name)
        if callable_func is None:
            raise ValueError(f"Tool {name} not found")
        if isinstance(callable_func, str):
            return await self.mcp_clients[callable_func].call_tool(name, arguments)
        signature = inspect.signature(callable_func)
        if __CTX_VARS_NAME__ in signature.parameters:
            arguments[__CTX_VARS_NAME__] = context_variables or {}
        result = callable_func(**arguments)
        if inspect.isawaitable(result):
            result = await result
        return result

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

    def add_function(self, func: "AgentFunction"):
        func_json = function_to_json(func)
        name = func_json["function"]["name"]
        self.functions[name] = func_json
        self._tools[name] = func

    async def close(self):
        await self.exit_stack.aclose()


TOOL_REGISTRY = ToolRegistry()


async def call_tools(
    tool_calls: dict[Hashable, list[ChatCompletionMessageToolCallParam]],
    context_variables: dict[str, Any],
) -> tuple[
    dict[Hashable, list[ChatCompletionMessageParam]],
    dict[str, Any],
    dict[Hashable, list[Node]],
]:
    messages: dict[Hashable, list[ChatCompletionMessageParam]] = defaultdict(list)
    tasks: dict[
        Hashable, list[tuple[asyncio.Task, ChatCompletionMessageToolCallParam]]
    ] = defaultdict(list)
    async with asyncio.TaskGroup() as group:
        for node, _tool_calls in tool_calls.items():
            for tool_call in _tool_calls:
                name = tool_call["function"]["name"]
                arguments = json.loads(tool_call["function"]["arguments"])
                tasks[node].append(
                    (
                        group.create_task(
                            TOOL_REGISTRY.call_tool(name, arguments, context_variables)
                        ),
                        tool_call,
                    )
                )
    agents: dict[Hashable, list[Node]] = defaultdict(list)
    for node, _tasks in tasks.items():
        for task, tool_call in _tasks:
            i = tool_call["id"]
            name = tool_call["function"]["name"]
            match result := task.result():
                case str():
                    messages[node].append(
                        {"role": "tool", "content": result, "tool_call_id": i}
                    )
                case dict():
                    context_variables |= result
                case Agent() | Swarm():
                    agents[node].append(result)
                case list() if all(isinstance(i, Agent | Swarm) for i in result):
                    agents[node].extend(result)
                case Result() as response:
                    agents[node].extend(response.agents)
                    context_variables |= response.context_variables
                    new_messages = [m for m in response.messages]
                    for m in new_messages:
                        if m["role"] == "tool":
                            m["tool_call_id"] = i
                        else:
                            m["name"] = f"{name} ({i})"
                    messages[node].extend(new_messages)
                case mcp.types.CallToolResult():
                    content = _mcp_call_tool_result_to_content(result)
                    messages[node].append(
                        {"role": "user", "content": content, "name": f"{name} ({i})"}
                    )
                case _:
                    raise ValueError(f"Unknown result type: {type(result)}")
    return messages, context_variables, agents


# SECTION 5: Models
class Result(BaseModel):
    messages: list[ChatCompletionMessageParam] = Field(default_factory=list)
    agents: list[Node] = Field(default_factory=list)
    context_variables: dict[str, Any] = Field(default_factory=dict)


class Agent(BaseModel):
    name: str = Field("Agent", strict=True, max_length=256)
    """User-friendly name for the display"""

    model: ChatModel | str = "deepseek-reasoner"
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

    completion_create_params: CompletionCreateParamsBase = Field(
        default_factory=lambda: {"model": "DUMMY", "messages": iter([])}
    )

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

    @overload
    async def get_chat_completion(
        self,
        *,
        messages: list[ChatCompletionMessageParam],
        context_variables: dict,
        stream: Literal[True],
    ) -> AsyncStream[ChatCompletionChunk]: ...

    @overload
    async def get_chat_completion(
        self,
        *,
        messages: list[ChatCompletionMessageParam],
        context_variables: dict,
        stream: Literal[False] = False,
    ) -> ChatCompletion: ...

    async def get_chat_completion(
        self,
        *,
        messages: list[ChatCompletionMessageParam],
        context_variables: dict,
        stream: bool = False,
    ) -> ChatCompletion | AsyncStream[ChatCompletionChunk]:
        messages = self._with_instructions(
            messages=messages,
            context_variables=context_variables,
        )
        create_params = {"messages": messages, "model": self.model}
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

        create_params = self.completion_create_params | create_params | {"tools": tools}

        return await (self.client or DEFAULT_CLIENT).chat.completions.create(
            stream=stream, **create_params
        )

    async def _run_and_stream(
        self,
        *,
        messages: list[ChatCompletionMessageParam],
        context_variables: dict[str, Any] | None = None,
    ):
        async for chunk in await self.get_chat_completion(
            messages=messages,
            context_variables=context_variables or {},
            stream=True,
        ):
            yield chunk

    @overload
    async def run(
        self,
        *,
        messages: list[ChatCompletionMessageParam],
        stream: Literal[True],
        context_variables: dict[str, Any] | None = None,
        max_turns: int | None = None,
        execute_tools: bool = True,
    ) -> AsyncIterator[ChatCompletionChunk]: ...

    @overload
    async def run(
        self,
        *,
        messages: list[ChatCompletionMessageParam],
        stream: Literal[False] = False,
        context_variables: dict[str, Any] | None = None,
        max_turns: int | None = None,
        execute_tools: bool = True,
    ) -> list[ChatCompletionMessageParam]: ...

    async def run(
        self,
        *,
        messages: list[ChatCompletionMessageParam],
        stream: bool = False,
        context_variables: dict[str, Any] | None = None,
        max_turns: int | None = None,
        execute_tools: bool = True,
    ) -> list[ChatCompletionMessageParam] | AsyncIterator[ChatCompletionChunk]:
        if max_turns is not None and max_turns <= 0:
            raise RuntimeError("Reached max turns")
        context_variables = copy.deepcopy(context_variables or {})
        if stream:
            return self._run_and_stream(
                messages=messages,
                context_variables=context_variables,
            )
        completion = await self.get_chat_completion(
            messages=messages,
            context_variables=context_variables,
            stream=False,
        )
        message = completion.choices[0].message
        logger.debug("Received completion:", message)
        m = cast(
            ReasoningChatCompletionAssistantMessageParam,
            message.model_dump(mode="json", exclude_none=True),
        )
        m["name"] = f"{self.name} ({completion.id})"
        return [m]


class Swarm(BaseModel, nx.DiGraph):
    mcp_servers: dict[str, StdioServerParameters | str] | None = Field(
        None, alias="mcpServers"
    )
    graph: dict = Field(default_factory=dict)

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
                **TypeAdapter(NodeData).validate_python(node),
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
                {"id": node}
                | json.loads(TypeAdapter(NodeData).dump_json(data, exclude_none=True))
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
            "graph": self.graph,
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
        nx.DiGraph.__init__(self)

    def add_edge(self, u: Hashable, v: Hashable, **attr: Any):
        if u not in self._node:
            raise ValueError(f"Node {u} not found")
        if v not in self._node:
            raise ValueError(f"Node {v} not found")
        super().add_edge(u, v, **attr)

    @property
    def _next_node(self) -> int:
        return max([i for i in self.nodes if isinstance(i, int)] + [-1]) + 1

    def _would_early_stop(
        self,
        agents: dict[Hashable, Node],
        tool_calls: dict[Hashable, list[ChatCompletionMessageToolCallParam]],
        execute_tools: bool,
    ) -> bool:
        return (
            len([s for n in agents.keys() for s in self.successors(n)])
            == 0  # no successors
            and (
                len([c for tc in tool_calls.values() for c in tc]) == 0
                or not execute_tools
            )  # no tool calls or tool execution disabled
        )

    async def _execute_agents(
        self,
        agents: dict[Hashable, Node],
        messages: list[ChatCompletionMessageParam],
    ) -> dict[Hashable, list[ChatCompletionMessageParam]]:
        tasks: dict[Hashable, asyncio.Task] = {}
        async with asyncio.TaskGroup() as group:
            for node, agent in agents.items():
                tasks[node] = group.create_task(
                    agent.run(
                        stream=False,
                        context_variables=self.graph,
                        messages=messages,
                    )
                )
        node_messages: dict[Hashable, list[ChatCompletionMessageParam]] = {}
        for node, task in tasks.items():
            node_messages[node] = task.result()
            self.nodes[node]["executed"] = True
        return node_messages

    async def _execute_agents_stream(
        self,
        agents: dict[Hashable, Node],
        messages: list[ChatCompletionMessageParam],
    ) -> AsyncIterator[
        ChatCompletionChunk | dict[Hashable, list[ChatCompletionMessageParam]]
    ]:
        tasks: dict[Hashable, asyncio.Task] = {}
        async with asyncio.TaskGroup() as group:
            for node, agent in agents.items():
                tasks[node] = group.create_task(
                    agent.run(
                        stream=True,
                        context_variables=self.graph,
                        messages=messages,
                    )
                )
        node_messages: dict[Hashable, list[ChatCompletionMessageParam]] = {}
        for node, task in tasks.items():
            chunks = []
            async for chunk in task.result():
                yield chunk
                chunks.append(chunk)
            node_messages[node] = merge_chunks(chunks)  # type: ignore
            self.nodes[node]["executed"] = True
        yield node_messages

    async def _call_tools(
        self,
        tool_calls: dict[Hashable, list[ChatCompletionMessageToolCallParam]],
        node: Hashable,
    ) -> dict[Hashable, list[ChatCompletionMessageParam]]:
        messages: dict[Hashable, list[ChatCompletionMessageParam]] = defaultdict(list)
        tasks: dict[
            Hashable, list[tuple[asyncio.Task, ChatCompletionMessageToolCallParam]]
        ] = defaultdict(list)
        async with asyncio.TaskGroup() as group:
            for node, _tool_calls in tool_calls.items():
                for tool_call in _tool_calls:
                    name = tool_call["function"]["name"]
                    arguments = json.loads(tool_call["function"]["arguments"])
                    tasks[node].append(
                        (
                            group.create_task(
                                TOOL_REGISTRY.call_tool(name, arguments, self.graph)
                            ),
                            tool_call,
                        )
                    )
        agents: dict[Hashable, list[Node]] = defaultdict(list)
        for node, _tasks in tasks.items():
            for task, tool_call in _tasks:
                i = tool_call["id"]
                name = tool_call["function"]["name"]
                match result := task.result():
                    case str():
                        messages[node].append(
                            {"role": "tool", "content": result, "tool_call_id": i}
                        )
                    case dict():
                        self.graph |= result
                    case Agent() | Swarm():
                        agents[node].append(result)
                    case list() if all(isinstance(i, Agent | Swarm) for i in result):
                        agents[node].extend(result)
                    case Result() as response:
                        agents[node].extend(response.agents)
                        self.graph |= response.context_variables
                        new_messages = [m for m in response.messages]
                        for m in new_messages:
                            if m["role"] == "tool":
                                m["tool_call_id"] = i
                            else:
                                m["name"] = f"{name} ({i})"
                        messages[node].extend(new_messages)
                    case mcp.types.CallToolResult():
                        content = _mcp_call_tool_result_to_content(result)
                        messages[node].append(
                            {
                                "role": "user",
                                "content": content,
                                "name": f"{name} ({i})",
                            }
                        )
                    case _:
                        raise ValueError(f"Unknown result type: {type(result)}")
        for node, _successors in agents.items():
            for s in _successors:
                next_node = self._next_node
                self.add_node(
                    next_node,
                    **(
                        {"type": "agent", "agent": s}
                        if isinstance(s, Agent)
                        else {"type": "swarm", "swarm": s}
                    ),
                )
                self.add_edge(node, next_node)
        if len(agents) == 0:
            next_node = self._next_node
            self.add_node(next_node, **(self.nodes[node] | {"executed": False}))
            self.add_edge(node, next_node)
        return messages

    async def _run_and_stream(
        self,
        *,
        messages: list[ChatCompletionMessageParam],
        max_turns: int | None = None,
        execute_tools: bool = True,
    ):
        node_data = self.nodes[self.root]
        agents: dict[Hashable, Node] = {
            self.root: cast(Node, node_data[node_data["type"]])
        }
        init_len = len(messages)
        while max_turns is None or len(messages) - init_len < max_turns:
            _messages: dict[Hashable, list[ChatCompletionMessageParam]] = {}
            async for chunk in self._execute_agents_stream(
                messages=messages,
                agents=agents,
            ):
                if isinstance(chunk, dict):
                    _messages.update(chunk)
                else:
                    yield chunk
            messages = [*messages, *[m for nm in _messages.values() for m in nm]]
            tool_calls = {
                node: [
                    tc
                    for m in nm
                    if m["role"] == "assistant" and m.get("tool_calls")
                    for tc in m["tool_calls"]
                ]
                for node, nm in _messages.items()
            }
            if self._would_early_stop(agents, tool_calls, execute_tools):
                break
            _messages = await self._call_tools(tool_calls, next(iter(agents)))
            messages = [*messages, *[m for nm in _messages.values() for m in nm]]
            agents = {
                s: (
                    self.nodes[s]["agent"]
                    if self.nodes[s]["type"] == "agent"
                    else self.nodes[s]["swarm"]
                )
                for n in agents.keys()
                for s in self.successors(n)
                if all(self.nodes[p].get("executed") for p in self.predecessors(s))
            }

    @overload
    async def run(
        self,
        *,
        messages: list[ChatCompletionMessageParam],
        stream: Literal[True],
        context_variables: dict[str, Any] | None = None,
        max_turns: int | None = None,
        execute_tools: bool = True,
    ) -> AsyncIterator[ChatCompletionChunk]: ...

    @overload
    async def run(
        self,
        *,
        messages: list[ChatCompletionMessageParam],
        stream: Literal[False] = False,
        context_variables: dict[str, Any] | None = None,
        max_turns: int | None = None,
        execute_tools: bool = True,
    ) -> list[ChatCompletionMessageParam]: ...

    async def run(
        self,
        *,
        messages: list[ChatCompletionMessageParam],
        stream: bool = False,
        context_variables: dict[str, Any] | None = None,
        max_turns: int | None = None,
        execute_tools: bool = True,
    ) -> list[ChatCompletionMessageParam] | AsyncIterator[ChatCompletionChunk]:
        for name, server_params in (self.mcp_servers or {}).items():
            await TOOL_REGISTRY.add_mcp_server(name, server_params)
        self.graph |= context_variables or {}
        if stream:
            return self._run_and_stream(
                messages=messages,
                max_turns=max_turns,
            )
        node_data = self.nodes[self.root]
        agents: dict[Hashable, Node] = {
            self.root: cast(Node, node_data[node_data["type"]])
        }
        messages = list(copy.deepcopy(messages))
        init_len = len(messages)

        while max_turns is None or len(messages) - init_len < max_turns:
            # get completion with current history, agent
            _messages = await self._execute_agents(agents, messages=messages)
            # dump response to json avoiding pydantic's ValidatorIterator
            messages = [*messages, *[m for nm in _messages.values() for m in nm]]
            tool_calls = {
                node: [
                    tc
                    for m in nm
                    if m["role"] == "assistant" and m.get("tool_calls")
                    for tc in m["tool_calls"]
                ]
                for node, nm in _messages.items()
            }
            if self._would_early_stop(agents, tool_calls, execute_tools):
                break
            _messages = await self._call_tools(tool_calls, next(iter(agents)))
            messages = [*messages, *[m for nm in _messages.values() for m in nm]]
            agents = {
                s: (
                    self.nodes[s]["agent"]
                    if self.nodes[s]["type"] == "agent"
                    else self.nodes[s]["swarm"]
                )
                for n in agents.keys()
                for s in self.successors(n)
                if all(self.nodes[p].get("executed") for p in self.predecessors(s))
            }

        return messages[init_len:]


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
            help="The path to the swarmx file (networkx node_link_data with additional `mcpServers` key)",
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
    client = Swarm.model_validate(data)
    if not client.nodes:
        client.add_node(0, type="agent", agent=Agent(model=model))
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
                messages=messages,
                stream=True,
                context_variables=context_variables,
            ):
                delta = chunk.choices[0].delta
                if delta.content is not None:
                    typer.echo(delta.content, nl=False)
                if (
                    isinstance(c := getattr(delta, "reasoning_content", None), str)
                    and verbose
                ):
                    typer.secho(c, nl=False, fg="green")
                if delta.refusal is not None:
                    typer.secho(delta.refusal, nl=False, err=True, fg="purple")
                if chunk.choices[0].finish_reason is not None:
                    typer.echo()
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


if __name__ == "__main__":
    from functools import wraps

    def repl():
        """SwarmX REPL wrapper"""

        @wraps(main)
        def repl_main(*args, **kwargs):
            return asyncio.run(main(*args, **kwargs))

        typer.run(repl_main)

    repl()
