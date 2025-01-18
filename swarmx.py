# mypy: disable-error-code="misc"
import copy
import inspect
import json
import logging
import warnings
from collections import defaultdict
from contextlib import AsyncExitStack
from itertools import chain
from typing import (
    Any,
    AsyncIterable,
    Callable,
    Coroutine,
    Iterable,
    Literal,
    MutableSequence,
    TypeAlias,
    cast,
    get_type_hints,
    overload,
)

import mcp.types
from jinja2 import Template
from mcp import ClientSession, StdioServerParameters
from mcp.client.sse import sse_client
from mcp.client.stdio import stdio_client
from openai import AsyncOpenAI, AsyncStream, OpenAI, Stream
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
from openai.types.chat.chat_completion_message_tool_call import Function
from openai.types.chat.completion_create_params import CompletionCreateParamsBase
from pydantic import (
    BaseModel,
    Field,
    ImportString,
    PrivateAttr,
    computed_field,
    create_model,
)
from pydantic.json_schema import GenerateJsonSchema
from typing_extensions import Unpack

__CTX_VARS_NAME__ = "context_variables"

logger = logging.getLogger(__name__)

ReturnType: TypeAlias = "str | Agent | dict[str, Any] | Result"
AgentFunction: TypeAlias = Callable[..., ReturnType | Coroutine[Any, Any, ReturnType]]


def merge_fields(
    target: ChatCompletionAssistantMessageParam | ChatCompletionMessageToolCallParam,
    source: dict,
):
    for key, value in source.items():
        if isinstance(value, str):
            target[key] += value
        elif value is not None and isinstance(value, dict):
            merge_fields(target[key], value)


def merge_chunk(
    final_response: ChatCompletionAssistantMessageParam, delta: dict
) -> None:
    assert "tool_calls" in final_response
    delta.pop("role", None)
    merge_fields(final_response, delta)

    tool_calls = delta.get("tool_calls")
    if tool_calls and len(tool_calls) > 0:
        index = tool_calls[0].pop("index")
        merge_fields(
            cast(
                MutableSequence[ChatCompletionMessageToolCallParam],
                final_response["tool_calls"],
            )[index],
            tool_calls[0],
        )


def does_function_need_context(
    func: AgentFunction | Callable[..., mcp.types.CallToolResult],
) -> bool:
    try:
        return __CTX_VARS_NAME__ in func.__code__.co_varnames
    except AttributeError:
        return False


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
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        arguments_model = create_model(func.__name__, **field_definitions)  # type: ignore[call-overload]
    return {
        "type": "function",
        "function": {
            "name": func.__name__,
            "description": func.__doc__ or "",
            "parameters": {
                k: v
                for k, v in arguments_model.model_json_schema(
                    schema_generator=SwarmXGenerateJsonSchema
                ).items()
                if k != "title"
            },
        },
    }


def check_function(func: object) -> AgentFunction:
    if not callable(func):
        raise TypeError(f"Expected a callable object, got {type(func)}")
    annotation = get_type_hints(func)
    if annotation.get("return") not in [
        str,
        Agent,
        dict[str, Any],
        Result,
        Coroutine[Any, Any, str],
        Coroutine[Any, Any, Agent],
        Coroutine[Any, Any, dict[str, Any]],
        Coroutine[Any, Any, Result],
        None,
    ]:
        raise TypeError(
            "Agent function return type must be str, Agent, dict[str, Any], or Result"
        )
    return cast(AgentFunction, func)


class SwarmXGenerateJsonSchema(GenerateJsonSchema):
    def field_title_should_be_set(self, schema) -> bool:
        return False


class Agent(BaseModel):
    name: str = "Agent"
    """The agent's name"""

    model: ChatModel | str = "gpt-4o"
    """The default model to use for the agent."""

    instructions: str = "You are a helpful agent."
    """Agent's instructions, could be a Jinja2 template"""

    functions: list[ImportString] = Field(default_factory=list)
    """The tools available to the agent"""

    tool_choice: ChatCompletionToolChoiceOptionParam | None = None
    """The tool choice option for the agent"""

    parallel_tool_calls: bool = False
    """Whether to make tool calls in parallel"""

    _tool_registry: dict[str, AgentFunction | None] = PrivateAttr(default_factory=dict)

    def _with_instructions(
        self,
        *,
        messages: Iterable[ChatCompletionMessageParam],
        model: ChatModel | str,
        context_variables: dict[str, Any] | None = None,
    ) -> list[ChatCompletionMessageParam]:
        content = Template(self.instructions).render(context_variables or {})
        if model in ["o1", "o1-2024-12-17"]:
            instructions: ChatCompletionMessageParam = {
                "role": "developer",
                "content": content,
            }
        else:
            instructions = {
                "role": "system",
                "content": content,
            }
        return [instructions, *messages]

    def preprocess(
        self,
        *,
        context_variables: dict[str, Any] | None = None,
        **kwargs: Unpack[CompletionCreateParamsBase],
    ) -> CompletionCreateParamsBase:
        model = kwargs.get("model") or self.model
        messages = self._with_instructions(
            model=model,
            messages=kwargs["messages"],
            context_variables=context_variables,
        )
        return kwargs | {"messages": messages, "model": model}

    @computed_field  # type: ignore[prop-decorator]
    @property
    def tools(self) -> Iterable[ChatCompletionToolParam]:
        tools = []
        for function in self.functions:
            tools.append(function_to_json(check_function(function)))
            self._tool_registry[tools[-1]["function"]["name"]] = function
        return tools


class Response(BaseModel):
    messages: list[ChatCompletionMessageParam] = []
    agent: Agent | None = None
    context_variables: dict[str, Any] = {}


class Result(mcp.types.CallToolResult):
    """
    Encapsulates the possible return values for an agent function.

    Attributes:
        agent (Agent): The agent instance, if applicable.
    """

    agent: Agent | None = None

    @property
    def value(self) -> str:
        return "".join([c.text for c in self.content if c.type == "text"])


def handle_function_result(result) -> Result:
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
            try:
                return Result(
                    content=[mcp.types.TextContent(type="text", text=str(result))]
                )
            except Exception as e:
                error_message = f"Failed to cast response to string: {result}. Make sure agent functions return a string or Result object. Error: {str(e)}"
                logger.debug(error_message)
                raise TypeError(error_message)


class Swarm:
    def __init__(
        self,
        client: OpenAI | None = None,
    ):
        self.client = OpenAI() if client is None else client
        self.tool_registry: dict[tuple[str, str], ChatCompletionToolParam] = {}

    @overload
    def get_chat_completion(
        self,
        agent: Agent,
        context_variables: dict,
        stream: Literal[True],
        **kwargs: Unpack[CompletionCreateParamsBase],
    ) -> Stream[ChatCompletionChunk]: ...

    @overload
    def get_chat_completion(
        self,
        agent: Agent,
        context_variables: dict,
        stream: Literal[False] = False,
        **kwargs: Unpack[CompletionCreateParamsBase],
    ) -> ChatCompletion: ...

    def get_chat_completion(
        self,
        agent: Agent,
        context_variables: dict,
        stream: bool = False,
        **kwargs: Unpack[CompletionCreateParamsBase],
    ) -> ChatCompletion | Stream[ChatCompletionChunk]:
        create_params = agent.preprocess(context_variables=context_variables, **kwargs)
        messages = create_params["messages"]
        logger.debug("Getting chat completion for...:", messages)

        # hide context_variables from model
        for tool in cast(list[ChatCompletionToolParam], agent.tools):
            params: dict[str, Any] = tool["function"].get(
                "parameters", {"properties": {}}
            )
            params["properties"].pop(__CTX_VARS_NAME__, None)
            if __CTX_VARS_NAME__ in params.get("required", []):
                params["required"].remove(__CTX_VARS_NAME__)

        if agent.tools:
            if agent.tool_choice:
                create_params["tool_choice"] = agent.tool_choice
            create_params["parallel_tool_calls"] = agent.parallel_tool_calls
            create_params["tools"] = cast(list[ChatCompletionToolParam], agent.tools)

        return self.client.chat.completions.create(stream=stream, **create_params)

    def handle_tool_calls(
        self,
        tool_calls: list[ChatCompletionMessageToolCall],
        tools: dict[str, AgentFunction | None],
        context_variables: dict,
    ) -> Response:
        partial_response = Response(messages=[], agent=None, context_variables={})

        for tool_call in tool_calls:
            name = tool_call.function.name
            # handle missing tool case, skip to next tool
            if tools.get(name) is None:
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
            args = json.loads(tool_call.function.arguments)
            logger.debug(f"Processing tool call: {name} with arguments {args}")

            func = cast(AgentFunction, tools[name])
            # pass context_variables to agent functions
            if does_function_need_context(func):
                args[__CTX_VARS_NAME__] = context_variables
            raw_result = func(**args)

            result: Result = handle_function_result(raw_result)
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

    def run_and_stream(
        self,
        agent: Agent,
        context_variables: dict[str, Any] | None = None,
        max_turns: int | None = None,
        execute_tools: bool = True,
        **kwargs: Unpack[CompletionCreateParamsBase],
    ) -> Iterable[
        dict[Literal["delim"], Literal["start", "end"]]
        | dict[Literal["response"], Response]
        | dict[str, Any]
    ]:
        active_agent = agent
        context_variables = (
            copy.deepcopy(context_variables) if context_variables else {}
        )
        history = [m for m in copy.deepcopy(kwargs["messages"])]
        init_len = len(history)

        while max_turns is None or len(history) - init_len < max_turns:
            message: ChatCompletionMessageParam = {
                "content": "",
                "name": agent.name,
                "role": "assistant",
                "tool_calls": defaultdict(
                    lambda: {
                        "function": {"arguments": "", "name": ""},
                        "id": "",
                        "type": "",
                    }
                ),
            }

            # get completion with current history, agent
            completion = self.get_chat_completion(
                agent=active_agent,
                context_variables=context_variables,
                stream=True,
                **kwargs,
            )

            yield {"delim": "start"}
            for chunk in completion:
                delta = chunk.choices[0].delta.model_dump(mode="json")
                if delta["role"] == "assistant":
                    delta["name"] = active_agent.name
                yield delta
                delta.pop("role", None)
                delta.pop("name", None)
                merge_chunk(message, delta)
            yield {"delim": "end"}

            message["tool_calls"] = list(message.get("tool_calls", {}).values())  # type: ignore
            if not message["tool_calls"]:
                message.pop("tool_calls", None)
            logger.debug("Received completion:", message)
            history.append(message)

            if not message.get("tool_calls") or not execute_tools:
                logger.debug("Ending turn.")
                break

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
            partial_response = self.handle_tool_calls(
                tool_calls, active_agent._tool_registry, context_variables
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

    @overload
    def run(
        self,
        agent: Agent,
        *,
        stream: Literal[True],
        context_variables: dict[str, Any] | None = None,
        max_turns: int | None = None,
        execute_tools: bool = True,
        **kwargs: Unpack[CompletionCreateParamsBase],
    ) -> Iterable[
        dict[Literal["delim"], Literal["start", "end"]]
        | dict[Literal["response"], Response]
        | dict[str, Any]
    ]: ...

    @overload
    def run(
        self,
        agent: Agent,
        *,
        stream: Literal[False] = False,
        context_variables: dict[str, Any] | None = None,
        max_turns: int | None = None,
        execute_tools: bool = True,
        **kwargs: Unpack[CompletionCreateParamsBase],
    ) -> Response: ...

    def run(
        self,
        agent: Agent,
        *,
        stream: bool = False,
        context_variables: dict[str, Any] | None = None,
        max_turns: int | None = None,
        execute_tools: bool = True,
        **kwargs: Unpack[CompletionCreateParamsBase],
    ) -> (
        Response
        | Iterable[
            dict[Literal["delim"], Literal["start", "end"]]
            | dict[Literal["response"], Response]
            | dict[str, Any]
        ]
    ):
        if stream:
            return self.run_and_stream(
                agent=agent,
                context_variables=context_variables,
                max_turns=max_turns,
                execute_tools=execute_tools,
                **kwargs,
            )
        active_agent = agent
        context_variables = (
            copy.deepcopy(context_variables) if context_variables else {}
        )
        messages = list(copy.deepcopy(kwargs["messages"]))
        init_len = len(messages)

        while max_turns is None or len(messages) - init_len < max_turns:
            # get completion with current history, agent
            completion = self.get_chat_completion(
                agent=active_agent,
                context_variables=context_variables,
                stream=stream,
                **kwargs,
            )
            message = completion.choices[0].message
            logger.debug("Received completion:", message)
            m = cast(
                ChatCompletionAssistantMessageParam,
                message.model_dump(mode="json", exclude_none=True),
            )
            m["name"] = active_agent.name
            messages.append(m)

            if not message.tool_calls or not execute_tools:
                logger.debug("Ending turn.")
                break

            # handle function calls, updating context_variables, and switching agents
            partial_response = self.handle_tool_calls(
                message.tool_calls, active_agent._tool_registry, context_variables
            )
            messages.extend(partial_response.messages)
            context_variables |= partial_response.context_variables
            if partial_response.agent:
                active_agent = partial_response.agent

        return Response(
            messages=messages[init_len:],
            agent=active_agent,
            context_variables=context_variables,
        )


class MCPClient:
    def __init__(
        self,
        server_name: str,
        server_params: StdioServerParameters | str,
    ):
        self.server_name = server_name
        self.server_params = server_params
        self.exit_stack = AsyncExitStack()

    async def __aenter__(self):
        read_stream, write_stream = await self.exit_stack.enter_async_context(
            sse_client(self.server_params)
            if isinstance(self.server_params, str)
            else stdio_client(self.server_params)
        )
        self.session = cast(
            ClientSession,
            await self.exit_stack.enter_async_context(
                ClientSession(read_stream, write_stream)
            ),
        )
        await self.session.initialize()
        return self

    async def __aexit__(self, *exc_info):
        await self.exit_stack.aclose()

    async def list_tools(self) -> list[ChatCompletionToolParam]:
        """List tools available on the server.

        Returns:
            list[ChatCompletionToolParam]: The list of tools available on the server
        """
        return [
            ChatCompletionToolParam(
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description or "",
                        "parameters": tool.inputSchema,
                    },
                }
            )
            for tool in (await self.session.list_tools()).tools
        ]

    async def call_tool(
        self, name: str, arguments: dict | None = None
    ) -> mcp.types.CallToolResult:
        """Call a tool on the server.

        Parameters:
            name (str): The name of the tool to call
            arguments (dict): The arguments to pass to the tool

        Returns:
            mcp.types.CallToolResult: The result of the tool call
        """
        return await self.session.call_tool(name, arguments)


class AsyncSwarm:
    def __init__(
        self,
        mcp_servers: dict[str, StdioServerParameters | str] = {},
        client: AsyncOpenAI | None = None,
    ):
        self.mcp_servers = mcp_servers
        self.mcp_clients: dict[str, MCPClient] = {}
        self.exit_stack = AsyncExitStack()
        self.client = AsyncOpenAI() if client is None else client
        self.tool_registry: dict[str, tuple[str, ChatCompletionToolParam]] = {}

    async def __aenter__(self):
        for server_name, server_params in self.mcp_servers.items():
            client = await self.exit_stack.enter_async_context(
                MCPClient(server_name, server_params)
            )
            self.mcp_clients[server_name] = client
            self.tool_registry.update(
                {
                    tool["function"]["name"]: (server_name, tool)
                    for tool in await client.list_tools()
                }
            )
        return self

    async def __aexit__(self, *exc_info):
        await self.exit_stack.aclose()

    @overload
    async def get_chat_completion(
        self,
        agent: Agent,
        context_variables: dict,
        stream: Literal[True],
        **kwargs: Unpack[CompletionCreateParamsBase],
    ) -> AsyncStream[ChatCompletionChunk]: ...

    @overload
    async def get_chat_completion(
        self,
        agent: Agent,
        context_variables: dict,
        stream: Literal[False] = False,
        **kwargs: Unpack[CompletionCreateParamsBase],
    ) -> ChatCompletion: ...

    async def get_chat_completion(
        self,
        agent: Agent,
        context_variables: dict,
        stream: bool = False,
        **kwargs: Unpack[CompletionCreateParamsBase],
    ) -> ChatCompletion | AsyncStream[ChatCompletionChunk]:
        create_params = agent.preprocess(context_variables=context_variables, **kwargs)
        messages = create_params["messages"]
        logger.debug("Getting chat completion for...:", messages)

        # hide context_variables from model
        for tool in cast(list[ChatCompletionToolParam], agent.tools):
            params: dict[str, Any] = tool["function"].get(
                "parameters", {"properties": {}}
            )
            params["properties"].pop(__CTX_VARS_NAME__, None)
            if __CTX_VARS_NAME__ in params.get("required", []):
                params["required"].remove(__CTX_VARS_NAME__)

        if agent.tool_choice:
            create_params["tool_choice"] = agent.tool_choice
        create_params["parallel_tool_calls"] = agent.parallel_tool_calls
        create_params["tools"] = cast(
            list[ChatCompletionToolParam],
            chain(agent.tools, [tool for _, tool in self.tool_registry.values()]),
        )

        return await self.client.chat.completions.create(stream=stream, **create_params)

    async def handle_tool_calls(
        self,
        tool_calls: list[ChatCompletionMessageToolCall],
        tools: dict[str, AgentFunction | None],
        context_variables: dict,
    ) -> Response:
        partial_response = Response(messages=[], agent=None, context_variables={})

        for tool_call in tool_calls:
            name = tool_call.function.name
            # handle missing tool case, skip to next tool
            if tools.get(name) is None and name not in self.tool_registry:
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
            args = json.loads(tool_call.function.arguments)
            logger.debug(f"Processing tool call: {name} with arguments {args}")
            if tools.get(name) is None and name in self.tool_registry:
                server, _ = self.tool_registry[name]
                raw_result = await self.mcp_clients[server].call_tool(name, args)
            else:
                func = cast(AgentFunction, tools[name])
                # pass context_variables to agent functions
                if does_function_need_context(func):
                    args[__CTX_VARS_NAME__] = context_variables
                raw_result = func(**args)
                if inspect.isawaitable(raw_result):
                    raw_result = await raw_result

            result: Result = handle_function_result(raw_result)
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

    async def run_and_stream(
        self,
        agent: Agent,
        context_variables: dict[str, Any] | None = None,
        max_turns: int | None = None,
        execute_tools: bool = True,
        **kwargs: Unpack[CompletionCreateParamsBase],
    ) -> AsyncIterable[
        dict[Literal["delim"], Literal["start", "end"]]
        | dict[Literal["response"], Response]
        | dict[str, Any]
    ]:
        active_agent = agent
        context_variables = (
            copy.deepcopy(context_variables) if context_variables else {}
        )
        history = [m for m in copy.deepcopy(kwargs["messages"])]
        init_len = len(history)

        while max_turns is None or len(history) - init_len < max_turns:
            message: ChatCompletionMessageParam = {
                "content": "",
                "name": agent.name,
                "role": "assistant",
                "tool_calls": defaultdict(
                    lambda: {
                        "function": {"arguments": "", "name": ""},
                        "id": "",
                        "type": "",
                    }
                ),
            }

            # get completion with current history, agent
            completion = await self.get_chat_completion(
                agent=active_agent,
                context_variables=context_variables,
                stream=True,
                **kwargs,
            )

            yield {"delim": "start"}
            async for chunk in completion:
                delta = chunk.choices[0].delta.model_dump(mode="json")
                if delta["role"] == "assistant":
                    delta["name"] = active_agent.name
                yield delta
                delta.pop("role", None)
                delta.pop("name", None)
                merge_chunk(message, delta)
            yield {"delim": "end"}

            message["tool_calls"] = list(message.get("tool_calls", {}).values())  # type: ignore
            if not message["tool_calls"]:
                message.pop("tool_calls", None)
            logger.debug("Received completion:", message)
            history.append(message)

            if not message.get("tool_calls") or not execute_tools:
                logger.debug("Ending turn.")
                break

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
            partial_response = await self.handle_tool_calls(
                tool_calls, active_agent._tool_registry, context_variables
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

    @overload
    async def run(
        self,
        agent: Agent,
        *,
        stream: Literal[True],
        context_variables: dict[str, Any] | None = None,
        max_turns: int | None = None,
        execute_tools: bool = True,
        **kwargs: Unpack[CompletionCreateParamsBase],
    ) -> AsyncIterable[
        dict[Literal["delim"], Literal["start", "end"]]
        | dict[Literal["response"], Response]
        | dict[str, Any]
    ]: ...

    @overload
    async def run(
        self,
        agent: Agent,
        *,
        stream: Literal[False] = False,
        context_variables: dict[str, Any] | None = None,
        max_turns: int | None = None,
        execute_tools: bool = True,
        **kwargs: Unpack[CompletionCreateParamsBase],
    ) -> Response: ...

    async def run(
        self,
        agent: Agent,
        *,
        stream: bool = False,
        context_variables: dict[str, Any] | None = None,
        max_turns: int | None = None,
        execute_tools: bool = True,
        **kwargs: Unpack[CompletionCreateParamsBase],
    ) -> (
        Response
        | AsyncIterable[
            dict[Literal["delim"], Literal["start", "end"]]
            | dict[Literal["response"], Response]
            | dict[str, Any]
        ]
    ):
        if stream:
            return self.run_and_stream(
                agent=agent,
                context_variables=context_variables,
                max_turns=max_turns,
                execute_tools=execute_tools,
                **kwargs,
            )
        active_agent = agent
        context_variables = (
            copy.deepcopy(context_variables) if context_variables else {}
        )
        messages = list(copy.deepcopy(kwargs["messages"]))
        init_len = len(messages)

        while max_turns is None or len(messages) - init_len < max_turns:
            # get completion with current history, agent
            completion = await self.get_chat_completion(
                agent=active_agent,
                context_variables=context_variables,
                stream=stream,
                **kwargs,
            )
            message = completion.choices[0].message
            logger.debug("Received completion:", message)
            m = cast(
                ChatCompletionAssistantMessageParam,
                message.model_dump(mode="json", exclude_none=True),
            )  # exclude none to avoid validation error
            m["name"] = active_agent.name
            messages.append(m)

            if not message.tool_calls or not execute_tools:
                logger.debug("Ending turn.")
                break

            # handle function calls, updating context_variables, and switching agents
            partial_response = await self.handle_tool_calls(
                message.tool_calls, active_agent._tool_registry, context_variables
            )
            messages.extend(partial_response.messages)
            context_variables |= partial_response.context_variables
            if partial_response.agent:
                active_agent = partial_response.agent
            kwargs["messages"] = messages

        return Response(
            messages=messages[init_len:],
            agent=active_agent,
            context_variables=context_variables,
        )
