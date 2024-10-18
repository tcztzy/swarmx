import copy
import json
from collections import defaultdict
from typing import Any, Literal, cast, overload

from loguru import logger
from openai import OpenAI
from openai.resources.chat.completions import NOT_GIVEN, Stream

from .types import (
    Agent,
    AgentFunction,
    AgentFunctionReturnType,
    ChatCompletion,
    ChatCompletionChunk,
    ChatCompletionMessage,
    ChatCompletionMessageToolCall,
    FunctionParameters,
    Response,
    Result,
)
from .util import function_to_json

__CTX_VARS_NAME__ = "context_variables"


class Swarm:
    def __init__(self, client: OpenAI | None = None):
        if not client:
            client = OpenAI()
        self.client = client

    @overload
    def get_chat_completion(
        self,
        agent: Agent,
        history: list,
        context_variables: dict,
        model_override: str | None,
        stream: Literal[True],
    ) -> Stream[ChatCompletionChunk]: ...

    @overload
    def get_chat_completion(
        self,
        agent: Agent,
        history: list,
        context_variables: dict[str, Any],
        model_override: str | None,
        stream: Literal[False],
    ) -> ChatCompletion: ...

    def get_chat_completion(
        self,
        agent: Agent,
        history: list,
        context_variables: dict[str, Any],
        model_override: str | None,
        stream: bool,
    ) -> ChatCompletion | Stream[ChatCompletionChunk]:
        context_variables = defaultdict(str, context_variables)
        instructions = (
            agent.instructions(context_variables)
            if callable(agent.instructions)
            else agent.instructions
        )
        messages = [{"role": "system", "content": instructions}] + history
        logger.debug("Getting chat completion for...:", messages)

        tools = [function_to_json(f) for f in agent.functions]
        # hide context_variables from model
        for tool in tools:
            params = cast(FunctionParameters, tool["function"]["parameters"])
            params["properties"].pop(__CTX_VARS_NAME__, None)
            if __CTX_VARS_NAME__ in params["required"]:
                params["required"].remove(__CTX_VARS_NAME__)
        return self.client.chat.completions.create(
            model=model_override or agent.model,
            messages=messages,
            tools=tools or NOT_GIVEN,
            tool_choice=agent.tool_choice or NOT_GIVEN,
            stream=stream,
            parallel_tool_calls=len(tools) > 0 and agent.parallel_tool_calls,
        )

    def handle_function_result(self, result: AgentFunctionReturnType) -> Result:
        match result:
            case Result() as result:
                return result

            case Agent() as agent:
                return Result(
                    value=json.dumps({"assistant": agent.name}),
                    agent=agent,
                )
            case _:
                try:
                    return Result(value=str(result))
                except Exception as e:
                    error_message = f"Failed to cast response to string: {result}. Make sure agent functions return a string or Result object. Error: {str(e)}"
                    logger.debug(error_message)
                    raise TypeError(error_message)

    def handle_tool_calls(
        self,
        tool_calls: list[ChatCompletionMessageToolCall],
        functions: list[AgentFunction],
        context_variables: dict[str, Any],
    ) -> Response:
        function_map = {f.__name__: f for f in functions}
        partial_response = Response(messages=[], agent=None, context_variables={})

        for tool_call in tool_calls:
            name = tool_call.function.name
            # handle missing tool case, skip to next tool
            if name not in function_map:
                logger.debug(f"Tool {name} not found in function map.")
                partial_response.messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "tool_name": name,
                        "content": f"Error: Tool {name} not found.",
                    }
                )
                continue
            args = json.loads(tool_call.function.arguments)
            logger.debug(f"Processing tool call: {name} with arguments {args}")

            func = function_map[name]
            # pass context_variables to agent functions
            if __CTX_VARS_NAME__ in func.__code__.co_varnames:
                args[__CTX_VARS_NAME__] = context_variables
            raw_result = function_map[name](**args)

            result: Result = self.handle_function_result(raw_result)
            partial_response.messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "tool_name": name,
                    "content": result.value,
                }
            )
            partial_response.context_variables.update(result.context_variables)
            if result.agent:
                partial_response.agent = result.agent

        return partial_response

    def run_and_stream(
        self,
        agent: Agent,
        messages: list,
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
            completion = self.get_chat_completion(
                agent=active_agent,
                history=history,
                context_variables=context_variables,
                model_override=model_override,
                stream=True,
            )

            yield {"delim": "start"}
            first = True
            message: ChatCompletionMessage | None = None
            for chunk in completion:
                delta = chunk.choices[0].delta
                delta_json = delta.model_dump(mode="json")
                if delta.role == "assistant":
                    delta_json["sender"] = active_agent.name
                yield json.dumps(delta_json)
                if first:
                    first = False
                    message = ChatCompletionMessage.from_delta(delta)
                else:
                    message = cast(ChatCompletionMessage, message)
                    message += delta
            if message is None:
                break
            yield {"delim": "end"}

            logger.debug("Received completion:", message.model_dump_json())
            history.append(message.model_dump(mode="json"))

            if not message._tool_calls or not execute_tools:
                logger.debug("Ending turn.")
                break

            # convert tool_calls to list
            tool_calls = []
            for i, index in enumerate(sorted(message._tool_calls)):
                if i != index:
                    logger.warning(f"Tool call index mismatch: {i} != {index}")
                    continue
                tool_calls.append(message._tool_calls[index])

            # handle function calls, updating context_variables, and switching agents
            partial_response = self.handle_tool_calls(
                tool_calls, active_agent.functions, context_variables
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
        messages: list,
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
            completion = self.get_chat_completion(
                agent=active_agent,
                history=history,
                context_variables=context_variables,
                model_override=model_override,
                stream=stream,
            )
            message = completion.choices[0].message
            logger.debug("Received completion:", message)
            message_data = message.model_dump(mode="json")
            message_data["sender"] = active_agent.name
            history.append(message_data)

            if not message.tool_calls or not execute_tools:
                logger.debug("Ending turn.")
                break

            # handle function calls, updating context_variables, and switching agents
            partial_response = self.handle_tool_calls(
                message.tool_calls, active_agent.functions, context_variables
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
