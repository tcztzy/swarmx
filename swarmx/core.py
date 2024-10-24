import copy
from dataclasses import dataclass
from typing import Any, cast

from loguru import logger
from openai import OpenAI

from .config import settings
from .types import (
    Agent,
    ChatCompletionAssistantMessageParam,
    ChatCompletionMessage,
    ChatCompletionMessageParam,
    Response,
)
from .util import (
    handle_tool_calls,
)


@dataclass
class Swarm:
    client: OpenAI = settings.openai

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

            logger.debug(f"Received completion: {message.model_dump_json()}")
            history.append(
                cast(ChatCompletionMessageParam, message.model_dump(mode="json"))
            )

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
                tool_calls, active_agent.tools, context_variables
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
            message_data = cast(
                ChatCompletionAssistantMessageParam,
                message.model_dump(mode="json", exclude_none=True),
            )
            message_data["name"] = active_agent.name
            history.append(message_data)

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
