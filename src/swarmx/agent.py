"""SwarmX Agent module."""

import copy
import inspect
import logging
import sys
import warnings
from collections import defaultdict
from typing import (
    TYPE_CHECKING,
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

from jinja2 import Template
from openai import AsyncOpenAI, AsyncStream
from openai.types.chat.chat_completion import ChatCompletion
from openai.types.chat.chat_completion_assistant_message_param import (
    ChatCompletionAssistantMessageParam,
)
from openai.types.chat.chat_completion_chunk import (
    ChatCompletionChunk,
    ChoiceDelta,
)
from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam
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
    Field,
    ImportString,
    TypeAdapter,
    field_serializer,
    field_validator,
)

from .mcp_client import TOOL_REGISTRY

PY_312 = sys.version_info >= (3, 12)

if PY_312:
    pass
else:
    pass


if TYPE_CHECKING:
    from .swarm import Swarm

warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")
logger = logging.getLogger(__name__)


__CTX_VARS_NAME__ = "context_variables"
DEFAULT_CLIENT: AsyncOpenAI | None = None
RANDOM_STRING_CHARS = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"

Node: TypeAlias = "Agent | Swarm"
ReturnType: TypeAlias = "str | Node | dict[str, Any] | Result"
AgentFunction: TypeAlias = Callable[..., ReturnType | Coroutine[Any, Any, ReturnType]]


def merge_chunk(message: ChatCompletionMessageParam, delta: ChoiceDelta) -> None:
    """Merge a chunk into a message.

    This function mutates the message in-place.

    """
    if delta.role:
        message["role"] = delta.role  # type: ignore
    content = message.get("content")
    if delta.content is not None:
        if isinstance(content, str) or content is None:
            message["content"] = (content or "") + delta.content
        else:
            message["content"] = [*content, {"type": "text", "text": delta.content}]  # type: ignore

    # Handle reasoning_content
    if (rc := getattr(delta, "reasoning_content", None)) is not None:
        reasoning_content = message.get("reasoning_content") or ""
        if isinstance(reasoning_content, str):
            message["reasoning_content"] = reasoning_content + rc  # type: ignore
        else:
            message["reasoning_content"] = list(reasoning_content) + [  # type: ignore
                {"type": "text", "text": rc}
            ]

    if delta.refusal is not None:
        cast(ChatCompletionAssistantMessageParam, message)["refusal"] = (
            message.get("refusal") or ""
        ) + delta.refusal

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
        message["tool_calls"] = tool_calls  # type: ignore


def merge_chunks(
    chunks: list[ChatCompletionChunk],
) -> list[ChatCompletionMessageParam]:
    """Merge a list of chunks into a list of messages.

    This function is useful for streaming the messages to the client.

    """
    messages: dict[str, ChatCompletionMessageParam] = defaultdict(
        lambda: {"role": "assistant"}
    )
    for chunk in chunks:
        message = messages[chunk.id]
        delta = chunk.choices[0].delta
        merge_chunk(message, delta)
        if not message.get("tool_calls"):
            message.pop("tool_calls", None)
        logger.debug("Received completion:", message)
    for message in messages.values():
        if tool_calls := message.get("tool_calls"):
            tool_calls = cast(dict[int, ChatCompletionMessageToolCallParam], tool_calls)
            # assert no gap in tool_calls keys
            for i, index in enumerate(sorted(tool_calls)):
                if i != index:
                    raise ValueError(f"Tool call index {index} is out of order")
            message["tool_calls"] = [tool_calls[i] for i in sorted(tool_calls)]  # type: ignore
    return list(messages.values())


def does_function_need_context(func) -> bool:
    """Check if a function needs context variables.

    This is determined by whether the function has a parameter named __CTX_VARS_NAME__.

    """
    signature = inspect.signature(func)
    return __CTX_VARS_NAME__ in signature.parameters


def check_instructions(
    instructions: str | object,
) -> str | Callable[..., str]:
    """Check the instructions and convert it to a callable if necessary.

    The instructions can be a string or a callable.
    If it's a string, we return it as is.
    If it's a callable, we check if it's valid and return it.

    """
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
    """Validate the tool and convert it to the ChatCompletionToolParam format.

    This function also adds the tool to the TOOL_REGISTRY.

    """
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
            TOOL_REGISTRY.add_function(cast(AgentFunction, tool))
            return TOOL_REGISTRY.functions[getattr(tool, "__name__", str(tool))]
        case str():
            return validate_tool(TypeAdapter(ImportString).validate_python(tool))
        case _:
            raise e


def validate_tools(tools: list[object]) -> list[ChatCompletionToolParam]:
    """Validate the tools and convert them to the ChatCompletionToolParam format.

    This function also adds the tools to the TOOL_REGISTRY.

    """
    return [validate_tool(tool) for tool in tools]


class Result(BaseModel):
    """Result of running a node.

    This is used to return the result of running a node.
    The result can contain messages, agents, and context variables.

    """

    messages: list[ChatCompletionMessageParam] = Field(default_factory=list)
    agents: list[Node] = Field(default_factory=list)
    context_variables: dict[str, Any] = Field(default_factory=dict)


class Agent(BaseModel):
    """Agent node in the swarm.

    An agent is a node in the swarm that can send and receive messages.
    It can have tools and instructions.
    It can also have a client to use for the chat completion API.

    """

    name: Annotated[str, Field(strict=True, max_length=256)] = "Agent"
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
        """Validate the client.

        If it's a dict, we create a new AsyncOpenAI client from it.
        If it's None, we use the global DEFAULT_CLIENT.
        Otherwise, we assume it's already a valid AsyncOpenAI client.

        """
        if v is None:
            return None
        if isinstance(v, AsyncOpenAI):
            return v
        return AsyncOpenAI(**v)

    @field_serializer("client", mode="plain")
    def serialize_client(self, v: AsyncOpenAI | None) -> dict[str, Any] | None:
        """Serialize the client.

        We only serialize the non-default parameters.

        """
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
        """Get a chat completion for the agent.

        Args:
            messages: The messages to start the conversation with
            context_variables: The context variables to pass to the agent
            stream: Whether to stream the response

        """
        messages = self._with_instructions(
            messages=messages,
            context_variables=context_variables,
        )
        create_params: CompletionCreateParamsBase = {
            "messages": messages,
            "model": self.model,
        }
        logger.debug("Getting chat completion for...:", messages)

        # hide context_variables from model
        tools: list[ChatCompletionToolParam] = [
            *self.model_dump(mode="json", exclude={"api_key"})["tools"],
            *TOOL_REGISTRY.tools.values(),
        ]
        for tool in tools:
            params: dict[str, Any] = tool["function"].get(
                "parameters", {"properties": {}}
            )
            params["properties"].pop(__CTX_VARS_NAME__, None)
            if __CTX_VARS_NAME__ in params.get("required", []):
                params["required"].remove(__CTX_VARS_NAME__)

        create_params = self.completion_create_params | create_params
        if len(tools) > 0:
            create_params["tools"] = tools

        return await (
            self.client or DEFAULT_CLIENT or AsyncOpenAI()
        ).chat.completions.create(stream=stream, **create_params)

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
        """Run the agent.

        Args:
            messages: The messages to start the conversation with
            stream: Whether to stream the response
            context_variables: The context variables to pass to the agent
            max_turns: The maximum number of turns to run the agent for
            execute_tools: Whether to execute tool calls

        """
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
            ChatCompletionAssistantMessageParam,
            message.model_dump(mode="json", exclude_none=True),
        )
        m["name"] = f"{self.name} ({completion.id})"
        return [m]


def condition_parser(condition: str) -> Callable[[dict[str, Any]], bool]:
    """Parse a condition string into a callable that takes a context and returns a boolean."""
    # First, we should try to parse condition as ImportString
    return TypeAdapter(ImportString).validate_python(condition)
