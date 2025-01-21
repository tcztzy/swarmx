# mypy: disable-error-code="misc"
import asyncio
import copy
import inspect
import json
import logging
import warnings
from collections import defaultdict
from contextlib import AsyncExitStack
from itertools import chain
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
    get_type_hints,
    overload,
)

import mcp.types
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
    Field,
    ImportString,
    PrivateAttr,
    TypeAdapter,
    computed_field,
    create_model,
)
from pydantic.json_schema import GenerateJsonSchema
from typing_extensions import Unpack

__CTX_VARS_NAME__ = "context_variables"

logger = logging.getLogger(__name__)

ReturnType: TypeAlias = "str | Agent | dict[str, Any] | Result"
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

    if delta.tool_calls:
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
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        arguments_model = create_model(func.__name__, **field_definitions)  # type: ignore[call-overload]
    name = TypeAdapter(ImportString).dump_json(func).decode()
    if name.startswith('"') and name.endswith('"'):
        name = name[1:-1]
    else:
        name = func.__name__
    return {
        "type": "function",
        "function": {
            "name": name,
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
    e = TypeError(
        "Agent function return type must be str, Agent, dict[str, Any], or Result"
    )
    match func:
        case func if callable(func):
            annotation = get_type_hints(func)
            if (return_anno := annotation.get("return")) is None:
                warnings.warn(
                    "Agent function return type is not annotated, assuming str. "
                    "This will be an error in a future version.",
                    FutureWarning,
                )
            if return_anno not in [str, Agent, dict[str, Any], Result, None]:
                raise e
            return cast(AgentFunction, func)
        case _:
            raise e


def check_functions(functions: list[object]) -> list[AgentFunction]:
    return [check_function(func) for func in functions]


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

    functions: Annotated[list[ImportString], AfterValidator(check_functions)] = Field(
        default_factory=list
    )
    """The tools available to the agent"""

    tool_choice: ChatCompletionToolChoiceOptionParam | None = None
    """The tool choice option for the agent"""

    parallel_tool_calls: bool = False
    """Whether to make tool calls in parallel"""

    _tool_registry: dict[str, AgentFunction] = PrivateAttr(default_factory=dict)

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

    @computed_field
    @property
    def tools(self) -> Iterable[ChatCompletionToolParam]:
        tools = []
        for func in self.functions:
            tools.append(function_to_json(func))
            self._tool_registry[tools[-1]["function"]["name"]] = func
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


async def handle_tool_calls(
    tool_calls: list[ChatCompletionMessageToolCall],
    functions: dict[str, AgentFunction],
    context_variables: dict[str, Any],
    mcp_tool_registry: dict[str, tuple[str, ChatCompletionToolParam]],
    mcp_clients: dict[str, ClientSession],
) -> Response:
    partial_response = Response(messages=[], agent=None, context_variables={})

    for tool_call in tool_calls:
        name = tool_call.function.name
        # handle missing tool case, skip to next tool
        if functions.get(name) is None and name not in mcp_tool_registry:
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
        if functions.get(name) is None and name in mcp_tool_registry:
            server, _ = mcp_tool_registry[name]
            raw_result = await mcp_clients[server].call_tool(name, args)
        else:
            func = cast(AgentFunction, functions[name])
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
            try:
                return Result(
                    content=[mcp.types.TextContent(type="text", text=str(result))]
                )
            except Exception as e:
                error_message = f"Failed to cast response to string: {result}. Make sure agent functions return a string or Result object. Error: {str(e)}"
                logger.debug(error_message)
                raise TypeError(error_message)


class BaseSwarm(BaseModel):
    mcp_servers: dict[str, StdioServerParameters | str] = Field(
        default_factory=dict,
        validation_alias=AliasChoices("mcp_servers", "mcpServers"),
        serialization_alias="mcpServers",
    )


class AsyncSwarm(BaseSwarm):
    _client: AsyncOpenAI = PrivateAttr()
    _mcp_clients: dict[str, ClientSession] = PrivateAttr()
    _exit_stack: AsyncExitStack = PrivateAttr()
    _tool_registry: dict[str, tuple[str, ChatCompletionToolParam]] = PrivateAttr()

    def model_post_init(self, __context: Any) -> None:
        self._mcp_clients = {}
        self._exit_stack = AsyncExitStack()
        self._client = AsyncOpenAI()
        self._tool_registry: dict[str, tuple[str, ChatCompletionToolParam]] = {}

    async def __aenter__(self):
        for server_name, server_params in self.mcp_servers.items():
            read_stream, write_stream = await self._exit_stack.enter_async_context(
                sse_client(server_params)
                if isinstance(server_params, str)
                else stdio_client(server_params)
            )
            client = cast(
                ClientSession,
                await self._exit_stack.enter_async_context(
                    ClientSession(read_stream, write_stream)
                ),
            )
            await client.initialize()
            self._mcp_clients[server_name] = client
            self._tool_registry.update(
                {
                    tool.name: (
                        server_name,
                        ChatCompletionToolParam(
                            {
                                "type": "function",
                                "function": {
                                    "name": tool.name,
                                    "description": tool.description or "",
                                    "parameters": tool.inputSchema,
                                },
                            }
                        ),
                    )
                    for tool in (await client.list_tools()).tools
                }
            )
        return self

    async def __aexit__(self, *exc_info):
        await self._exit_stack.aclose()

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
            chain(agent.tools, [tool for _, tool in self._tool_registry.values()]),
        )

        return await self._client.chat.completions.create(
            stream=stream, **create_params
        )

    async def run_and_stream(
        self,
        agent: Agent,
        context_variables: dict[str, Any] | None = None,
        max_turns: int | None = None,
        execute_tools: bool = True,
        **kwargs: Unpack[CompletionCreateParamsBase],
    ):
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
                delta = chunk.choices[0].delta
                if delta.role == "assistant":
                    delta.name = active_agent.name  # type: ignore
                yield delta.model_dump(mode="json")
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
            partial_response = await handle_tool_calls(
                tool_calls,
                active_agent._tool_registry,
                context_variables,
                self._tool_registry,
                self._mcp_clients,
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
    ) -> AsyncIterator[
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
        | AsyncIterator[
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
            partial_response = await handle_tool_calls(
                message.tool_calls,
                active_agent._tool_registry,
                context_variables,
                self._tool_registry,
                self._mcp_clients,
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


class Swarm(BaseSwarm):
    _swarm: AsyncSwarm = PrivateAttr()

    def model_post_init(self, __context: Any) -> None:
        self._swarm = AsyncSwarm(mcp_servers=self.mcp_servers)

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
            agenerator = self._swarm.run_and_stream(
                agent=agent,
                context_variables=context_variables,
                max_turns=max_turns,
                execute_tools=execute_tools,
                **kwargs,
            )

            def generator():
                while True:
                    try:
                        yield asyncio.run(anext(agenerator))
                    except StopAsyncIteration:
                        break

            return generator()
        return asyncio.run(
            self._swarm.run(
                agent=agent,
                stream=False,
                context_variables=context_variables,
                max_turns=max_turns,
                execute_tools=execute_tools,
                **kwargs,
            )
        )


async def main():
    import argparse

    parser = argparse.ArgumentParser(description="SwarmX Command Line Interface")

    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o",
        help="The model to use for the agent",
    )
    parser.add_argument(
        "--file",
        type=Path,
        default=None,
        help="The path to the swarmx file (JSON file containing `mcpServers` and `agent`)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="The path to the output file to save the conversation",
    )
    args = parser.parse_args()
    if args.file is None:
        data = {}
    else:
        data = json.loads(args.file.read_text())
    agent = Agent.model_validate(data.pop("agent", {}))
    client = AsyncSwarm.model_validate(data)
    messages: list[ChatCompletionMessageParam] = []
    while True:
        try:
            user_prompt = input(">>> ")
            messages.append(
                {
                    "role": "user",
                    "content": user_prompt,
                }
            )
            message = ""
            async for chunk in await client.run(
                agent, model=args.model, messages=messages, stream=True
            ):
                if (c := chunk.get("content")) is not None:  # type: ignore
                    message += c  # type: ignore
                    print(c, end="", flush=True)  # noqa: T201
                elif (response := chunk.get("response")) is not None:  # type: ignore
                    messages.extend(response.messages)  # type: ignore
            print()  # noqa: T201
        except KeyboardInterrupt:
            break
    if args.output is not None:
        args.output.write_text(json.dumps(messages, indent=2))


def repl():
    asyncio.run(main())


if __name__ == "__main__":
    repl()
