import copy
from collections import defaultdict
from typing import Any, Literal, cast, overload

from loguru import logger
from openai import OpenAI
from openai.resources.chat.completions import NOT_GIVEN, Stream

from .types import (
    Agent,
    ChatCompletion,
    ChatCompletionChunk,
    ChatCompletionMessage,
    ChatCompletionMessageParam,
    Response,
)
from .util import (
    function_to_json,
    handle_tool_calls,
)


class Swarm:
    def __init__(self, client: OpenAI | None = None):
        if not client:
            client = OpenAI()
        self.client = client

    @overload
    def get_chat_completion(
        self,
        agent: Agent,
        history: list[ChatCompletionMessageParam],
        context_variables: dict,
        model_override: str | None,
        stream: Literal[True],
    ) -> Stream[ChatCompletionChunk]: ...

    @overload
    def get_chat_completion(
        self,
        agent: Agent,
        history: list[ChatCompletionMessageParam],
        context_variables: dict[str, Any],
        model_override: str | None,
        stream: Literal[False],
    ) -> ChatCompletion: ...

    def get_chat_completion(
        self,
        agent: Agent,
        history: list[ChatCompletionMessageParam],
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
        messages = [{"role": "system", "content": instructions}, *history]
        logger.debug("Getting chat completion for...:", messages)

        tools = [function_to_json(f) for f in agent.functions]
        return self.client.chat.completions.create(
            model=model_override or agent.model,
            messages=messages,
            tools=tools or NOT_GIVEN,
            tool_choice=agent.tool_choice or NOT_GIVEN,
            stream=stream,
            parallel_tool_calls=len(tools) > 0 and agent.parallel_tool_calls,
        )

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
                yield delta_json
                if first:
                    first = False
                    message = ChatCompletionMessage.from_delta(delta)
                else:
                    message = cast(ChatCompletionMessage, message) + delta
            yield {"delim": "end"}
            if message is None:
                break

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
            partial_response = handle_tool_calls(
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
            completion = self.get_chat_completion(
                agent=active_agent,
                history=history,
                context_variables=context_variables,
                model_override=model_override,
                stream=stream,
            )
            message = completion.choices[0].message
            logger.debug("Received completion:", message)
            message_data = message.model_dump(mode="json", exclude_none=True)
            message_data["name"] = active_agent.name
            history.append(message_data)

            if not message.tool_calls or not execute_tools:
                logger.debug("Ending turn.")
                break

            # handle function calls, updating context_variables, and switching agents
            partial_response = handle_tool_calls(
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
