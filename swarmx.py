from __future__ import annotations

import copy
import inspect
import json
import logging
import os
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Literal, MutableMapping, cast, overload

from openai import NOT_GIVEN, NotGiven, OpenAI, Stream
from openai.types.chat.chat_completion import ChatCompletion
from openai.types.chat.chat_completion_assistant_message_param import (
    ChatCompletionAssistantMessageParam,
)
from openai.types.chat.chat_completion_chunk import (
    ChatCompletionChunk,
    ChoiceDelta,
    ChoiceDeltaToolCall,
    ChoiceDeltaToolCallFunction,
)
from openai.types.chat.chat_completion_message import ChatCompletionMessage
from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam
from openai.types.chat.chat_completion_message_tool_call import (
    ChatCompletionMessageToolCall,
    Function,
)
from openai.types.chat.chat_completion_tool_choice_option_param import (
    ChatCompletionToolChoiceOptionParam,
)
from openai.types.chat.chat_completion_tool_param import ChatCompletionToolParam
from openai.types.chat_model import ChatModel
from pydantic import BaseModel, create_model
from pydantic.json_schema import GenerateJsonSchema

logger = logging.getLogger(__name__)
__CTX_VARS_NAME__ = "context_variables"
DEFAULT_MODEL = os.getenv("SWARMX_DEFAULT_MODEL", "gpt-4o")


class SwarmXGenerateJsonSchema(GenerateJsonSchema):
    """hide context_variables from model"""

    def generate(self, schema, mode="validation"):
        json_schema = super().generate(schema, mode=mode)
        properties = {
            name: {k: v for k, v in property.items() if k != "title"}
            for name, property in json_schema.pop("properties", {}).items()
            if name != __CTX_VARS_NAME__
        }
        required = [
            r for r in json_schema.pop("required", []) if r != __CTX_VARS_NAME__
        ]
        return {
            **({k: v for k, v in json_schema.items() if k != "title"}),
            "properties": properties,
            **({"required": required} if required else {}),
        }


def handle_tool_calls(
    tool_calls: list[ChatCompletionMessageToolCall],
    tools: list[Tool],
    context_variables: dict[str, Any],
):
    tool_map = {f.name: f for f in tools}
    partial_response = Response(messages=[], agent=None, context_variables={})

    for tool_call in tool_calls:
        name = tool_call.function.name
        # handle missing tool case, skip to next tool
        if name not in tool_map:
            logger.debug(f"Tool {name} not found in function map.")
            partial_response.messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": f"Error: Tool {name} not found.",
                }
            )
            continue

        tool = tool_map[name]
        args = tool.arguments_model.model_validate_json(
            tool_call.function.arguments
        ).model_dump(mode="json")
        logger.debug(f"Processing tool call: {name} with arguments {args}")
        # pass context_variables to agent functions
        if __CTX_VARS_NAME__ in tool.function.__code__.co_varnames:
            args[__CTX_VARS_NAME__] = context_variables
        result = tool(**args)
        partial_response.messages.append(
            {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result.value,
            }
        )
        partial_response.context_variables.update(result.context_variables)
        if result.agent:
            partial_response.agent = result.agent

    return partial_response


AgentFunction = Callable[..., Any]
try:
    from langchain.tools import Tool as LangChainTool
    from langchain_core.utils.function_calling import convert_to_openai_tool

    AgentFunction |= LangChainTool
except ImportError:
    LangChainTool = None
    convert_to_openai_tool = None


@dataclass
class Tool:
    function: AgentFunction
    arguments_model: BaseModel = field(init=False)

    def __post_init__(self):
        if LangChainTool is not None and isinstance(self.function, LangChainTool):
            self.name = self.function.name
            if not isinstance(self.function.args_schema, BaseModel):
                raise ValueError(
                    f"args_schema must be a Pydantic BaseModel for LangChainTool: {self.name}"
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

    def __call__(self, *args, **kwargs) -> Result:
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


try:
    from jinja2 import Template
except ImportError:

    @dataclass
    class Template:  # type: ignore[no-redef]
        template: str

        def render(self, context_variables: dict[str, Any]):
            return self.template.format(**context_variables)


@dataclass
class Agent:
    name: str = "Agent"
    model: ChatModel | str = DEFAULT_MODEL
    instructions: str | Callable[..., str] = "You are a helpful agent."
    functions: list[AgentFunction | Tool] = field(default_factory=list)
    tool_choice: ChatCompletionToolChoiceOptionParam | NotGiven = NOT_GIVEN
    parallel_tool_calls: bool = True
    client: OpenAI | None = None

    @property
    def tools(self) -> list[Tool]:
        return [
            function if isinstance(function, Tool) else Tool(function)
            for function in self.functions
        ]

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


@dataclass
class Swarm:
    client: OpenAI = field(default_factory=OpenAI)

    def run_and_stream(
        self,
        agent: Agent,
        messages: list[ChatCompletionMessageParam],
        context_variables: dict[str, Any] = {},
        model_override: str | None = None,
        max_turns: int | float = float("inf"),
        execute_tools: bool = True,
    ):
        active_agent = agent
        context_variables = copy.deepcopy(context_variables)
        history = copy.deepcopy(messages)
        init_len = len(messages)

        while len(history) - init_len < max_turns:
            # get completion with current history, agent
            completion = active_agent.run(
                self.client, history, model_override, True, context_variables
            )

            yield {"delim": "start"}
            partial_message: PartialChatCompletionMessage | None = None
            for i, chunk in enumerate(completion):
                delta = chunk.choices[0].delta
                yield delta.model_dump(mode="json")
                if i == 0:
                    partial_message = PartialChatCompletionMessage.from_delta(delta)
                else:
                    partial_message += delta
            yield {"delim": "end"}
            if partial_message is None:
                logger.debug("No completion chunk received.")
                break
            message = partial_message.oai_message()
            logger.debug(f"Received completion: {message.model_dump_json()}")
            agent_message = cast(
                ChatCompletionAssistantMessageParam,
                message.model_dump(mode="json", exclude_none=True),
            )
            agent_message["name"] = active_agent.name
            history.append(agent_message)

            if not message.tool_calls or not execute_tools:
                logger.debug("Ending turn.")
                break

            # handle function calls, updating context_variables, and switching agents
            partial_response = handle_tool_calls(
                message.tool_calls, active_agent.tools, context_variables
            )
            history.extend(partial_response.messages)
            context_variables.update(partial_response.context_variables)
            if partial_response.agent:
                active_agent = partial_response.agent

        yield {
            "response": Response(
                messages=history[init_len:],
                agent=active_agent,
                context_variables=context_variables,
            )
        }

    def run(
        self,
        agent: Agent,
        messages: list[ChatCompletionMessageParam],
        context_variables: dict = {},
        model_override: str | None = None,
        stream: bool = False,
        max_turns: int | float = float("inf"),
        execute_tools: bool = True,
    ) -> Response:
        if stream:
            return self.run_and_stream(
                agent=agent,
                messages=messages,
                context_variables=context_variables,
                model_override=model_override,
                max_turns=max_turns,
                execute_tools=execute_tools,
            )
        active_agent = agent
        context_variables = copy.deepcopy(context_variables)
        history = copy.deepcopy(messages)
        init_len = len(messages)

        while len(history) - init_len < max_turns and active_agent:
            # get completion with current history, agent
            completion = active_agent.run(
                self.client, history, model_override, False, context_variables
            )
            message = completion.choices[0].message
            logger.debug(f"Received completion: {message}")
            if message.tool_calls is not None and len(message.tool_calls) == 0:
                message.tool_calls = None
            assistant_message = cast(
                ChatCompletionAssistantMessageParam,
                message.model_dump(mode="json", exclude_none=True),
            )
            assistant_message["name"] = active_agent.name
            history.append(assistant_message)

            if not message.tool_calls or not execute_tools:
                logger.debug("Ending turn.")
                break

            # handle function calls, updating context_variables, and switching agents
            partial_response = handle_tool_calls(
                message.tool_calls, active_agent.tools, context_variables
            )
            history.extend(partial_response.messages)
            context_variables.update(partial_response.context_variables)
            if partial_response.agent:
                active_agent = partial_response.agent

        return Response(
            messages=history[init_len:],
            agent=active_agent,
            context_variables=context_variables,
        )
