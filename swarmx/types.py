import inspect
import json
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Literal, MutableMapping, overload

from jinja2 import Template
from loguru import logger
from openai import OpenAI
from openai.resources.chat.completions import NOT_GIVEN, NotGiven, Stream
from openai.types.chat.chat_completion import ChatCompletion as ChatCompletion
from openai.types.chat.chat_completion_assistant_message_param import (
    ChatCompletionAssistantMessageParam as ChatCompletionAssistantMessageParam,
)
from openai.types.chat.chat_completion_chunk import (
    ChatCompletionChunk as ChatCompletionChunk,
)
from openai.types.chat.chat_completion_chunk import (
    ChoiceDelta,
    ChoiceDeltaToolCall,
    ChoiceDeltaToolCallFunction,
)
from openai.types.chat.chat_completion_message import (
    ChatCompletionMessage as ChatCompletionMessage,
)
from openai.types.chat.chat_completion_message_param import (
    ChatCompletionMessageParam as ChatCompletionMessageParam,
)
from openai.types.chat.chat_completion_message_tool_call import (
    ChatCompletionMessageToolCall as ChatCompletionMessageToolCall,
)
from openai.types.chat.chat_completion_message_tool_call import (
    Function as Function,
)
from openai.types.chat.chat_completion_tool_choice_option_param import (
    ChatCompletionToolChoiceOptionParam as ChatCompletionToolChoiceOptionParam,
)
from openai.types.chat.chat_completion_tool_param import (
    ChatCompletionToolParam as ChatCompletionToolParam,
)
from openai.types.chat_model import ChatModel as ChatModel
from pydantic import BaseModel, create_model

from .config import settings
from .util import SwarmXGenerateJsonSchema

try:
    from langchain.tools import Tool as LangChainTool
    from langchain_core.utils.function_calling import convert_to_openai_tool
except ImportError:
    LangChainTool = None
    convert_to_openai_tool = None


@dataclass
class Tool:
    function: Callable[..., Any]
    arguments_model: BaseModel = field(init=False)

    def __post_init__(self):
        if LangChainTool is not None and isinstance(self.function, LangChainTool):
            self.name = self.function.name
            if not isinstance(self.function.args_schema, BaseModel):
                raise ValueError(
                    f"args_schema must be a Pydantic BaseModel for LangChainTool: {self.function.__name__}"
                )
            self.arguments_model = self.function.args_schema
        else:
            self.name = self.function.__name__
            try:
                signature = inspect.signature(self.function)
            except ValueError as e:
                raise ValueError(
                    f"Failed to get signature for function {self.function.__name__}: {str(e)}"
                )

            parameters = {}
            for param in signature.parameters.values():
                parameters[param.name] = (
                    param.annotation if param.annotation is not param.empty else str,
                    param.default if param.default is not param.empty else ...,
                )
            self.arguments_model = create_model(self.function.__name__, **parameters)  # type: ignore[call-overload]

    def __call__(self, *args, **kwargs) -> "Result":
        if LangChainTool is not None and isinstance(self.function, LangChainTool):
            result = self.function.run(kwargs)
        else:
            result = self.function(*args, **kwargs)
        match result:
            case Result() as result:
                return result

            case Agent() as agent:
                return Result(
                    value=json.dumps({"assistant": agent.name}),
                    agent=agent,
                )

            case BaseModel() as model:
                return Result(
                    value=model.model_dump_json(),
                )

            case _:
                try:
                    return Result(value=str(result))
                except Exception as e:
                    error_message = f"Failed to cast response to string: {result}. Make sure agent functions return a string or Result object. Error: {str(e)}"
                    logger.debug(error_message)
                    raise TypeError(error_message)

    def json(self) -> ChatCompletionToolParam:
        if LangChainTool is not None and isinstance(self.function, LangChainTool):
            return convert_to_openai_tool(self.function)
        return {
            "type": "function",
            "function": {
                "name": self.function.__name__,
                "description": self.function.__doc__ or "",
                "parameters": self.arguments_model.model_json_schema(
                    schema_generator=SwarmXGenerateJsonSchema
                ),
            },
        }


@dataclass
class Agent:
    name: str = "Agent"
    model: ChatModel | str = settings.default_model
    instructions: str | Callable[..., str] = "You are a helpful agent."
    functions: list[Callable[..., Any]] = field(default_factory=list)
    tool_choice: ChatCompletionToolChoiceOptionParam | NotGiven = NOT_GIVEN
    parallel_tool_calls: bool = True
    client: OpenAI | None = None

    @property
    def tools(self) -> list[Tool]:
        return [Tool(function) for function in self.functions]

    def _get_instructions(self, context_variables: dict[str, Any]) -> str:
        return (
            self.instructions
            if callable(self.instructions)
            else Template(self.instructions).render
        )(context_variables)

    @overload
    def run(
        self,
        client: OpenAI,
        messages: list[ChatCompletionMessageParam],
        model: ChatModel | str | None = None,
        stream: Literal[False] = False,
        context_variables: dict[str, Any] | None = None,
    ) -> ChatCompletion: ...

    @overload
    def run(
        self,
        client: OpenAI,
        messages: list[ChatCompletionMessageParam],
        model: ChatModel | str | None = None,
        stream: Literal[True] = True,
        context_variables: dict[str, Any] | None = None,
    ) -> Stream[ChatCompletionChunk]: ...

    def run(
        self,
        client: OpenAI,
        messages: list[ChatCompletionMessageParam],
        model: ChatModel | str | None = None,
        stream: bool = False,
        context_variables: dict[str, Any] | None = None,
    ) -> ChatCompletion | Stream[ChatCompletionChunk]:
        context_variables = defaultdict(str, context_variables or {})
        instructions = self._get_instructions(context_variables)
        messages = [
            {"role": "system", "content": instructions},
            *messages,
        ]
        logger.debug(f"Getting chat completion for...: {messages}")
        tools = [tool.json() for tool in self.tools] or NOT_GIVEN
        return (self.client or client).chat.completions.create(
            model=model or self.model,
            messages=messages,
            tools=tools,
            tool_choice=self.tool_choice,
            stream=stream,
            parallel_tool_calls=bool(tools) and self.parallel_tool_calls,
        )


@dataclass
class Response:
    messages: list[ChatCompletionMessageParam] = field(default_factory=list)
    agent: Agent | None = None
    context_variables: dict = field(default_factory=dict)


@dataclass
class Result:
    """
    Encapsulates the possible return values for an agent function.

    Attributes:
        value (str): The result value as a string.
        agent (Agent): The agent instance, if applicable.
        context_variables (dict): A dictionary of context variables.
    """

    value: str = ""
    agent: Agent | None = None
    context_variables: dict = field(default_factory=dict)


ToolCalls = MutableMapping[int, ChatCompletionMessageToolCall]


def _function(f: ChoiceDeltaToolCallFunction | None) -> Function:
    return Function(
        arguments=(f.arguments or "") if f else "",
        name=(f.name or "") if f else "",
    )


class PartialChatCompletionMessage(ChatCompletionMessage):
    _tool_calls: ToolCalls | None = None

    @staticmethod
    def from_delta(delta: ChoiceDelta) -> "PartialChatCompletionMessage":
        return PartialChatCompletionMessage(
            content=delta.content,
            refusal=delta.refusal,
            role="assistant",
            _tool_calls={
                tool.index: ChatCompletionMessageToolCall(
                    id=tool.id or "",
                    function=_function(tool.function),
                    type="function",
                )
                for tool in delta.tool_calls
            }
            if delta.tool_calls
            else None,
        )

    def __add__(self, other: ChoiceDelta) -> "PartialChatCompletionMessage":
        def _tool_calls_merge(
            a: ToolCalls | None, b: list[ChoiceDeltaToolCall] | None
        ) -> ToolCalls | None:
            if a is None:
                return (
                    {
                        tool.index: ChatCompletionMessageToolCall(
                            id=tool.id or "",
                            function=_function(tool.function),
                            type="function",
                        )
                        for tool in b
                    }
                    if b
                    else None
                )
            for tool in b or []:
                tool_call = a.get(tool.index)
                if tool_call is None:
                    a[tool.index] = ChatCompletionMessageToolCall(
                        id=tool.id or "",
                        function=_function(tool.function),
                        type="function",
                    )
                else:
                    tool_call.id += tool.id or ""
                    function = _function(tool.function)
                    tool_call.function.arguments += function.arguments
                    tool_call.function.name += function.name
            return a

        tool_calls = _tool_calls_merge(self._tool_calls, other.tool_calls)
        return PartialChatCompletionMessage(
            content=((self.content or "") + (other.content or "")) or None,
            refusal=((self.refusal or "") + (other.refusal or "")) or None,
            role="assistant",
            _tool_calls=tool_calls,
        )

    def oai_message(self) -> ChatCompletionMessage:
        if self._tool_calls is not None:
            tool_calls = []
            for i, index in enumerate(sorted(self._tool_calls)):
                if i != index:
                    logger.warning(f"Tool call index mismatch: {i} != {index}")
                    continue
                tool_calls.append(self._tool_calls[index])
        else:
            tool_calls = None
        return ChatCompletionMessage(
            content=self.content,
            refusal=self.refusal,
            role=self.role,
            tool_calls=tool_calls,
        )
