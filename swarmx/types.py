from typing import Any, Callable, MutableMapping, TypeAlias, TypedDict

from openai.types.chat.chat_completion import ChatCompletion as ChatCompletion
from openai.types.chat.chat_completion_chunk import (
    ChatCompletionChunk as ChatCompletionChunk,
)
from openai.types.chat.chat_completion_chunk import (
    ChoiceDelta,
    ChoiceDeltaToolCall,
    ChoiceDeltaToolCallFunction,
)
from openai.types.chat.chat_completion_message import (
    ChatCompletionMessage as _ChatCompletionMessage,
)
from openai.types.chat.chat_completion_message_tool_call import (
    ChatCompletionMessageToolCall as ChatCompletionMessageToolCall,
)
from openai.types.chat.chat_completion_message_tool_call import (
    Function as Function,
)
from openai.types.chat.chat_completion_tool_choice_option_param import (
    ChatCompletionToolChoiceOptionParam,
)
from openai.types.chat.chat_completion_tool_param import (
    ChatCompletionToolParam as ChatCompletionToolParam,
)

# Third-party imports
from pydantic import BaseModel

AgentFunctionReturnType: TypeAlias = "str | dict | Agent | Result"

AgentFunction = Callable[..., AgentFunctionReturnType]


class Agent(BaseModel):
    name: str = "Agent"
    model: str = "gpt-4o"
    instructions: str | Callable[..., str] = "You are a helpful agent."
    functions: list[AgentFunction] = []
    tool_choice: ChatCompletionToolChoiceOptionParam | None = None
    parallel_tool_calls: bool = True


class Response(BaseModel):
    messages: list = []
    agent: Agent | None = None
    context_variables: dict = {}


class Result(BaseModel):
    """
    Encapsulates the possible return values for an agent function.

    Attributes:
        value (str): The result value as a string.
        agent (Agent): The agent instance, if applicable.
        context_variables (dict): A dictionary of context variables.
    """

    value: str = ""
    agent: Agent | None = None
    context_variables: dict = {}


class FunctionParameters(TypedDict, total=False):
    properties: dict[str, Any]
    required: list[str]


ToolCalls = MutableMapping[int, ChatCompletionMessageToolCall]


def _function(f: ChoiceDeltaToolCallFunction | None) -> Function:
    return Function(
        arguments=(f.arguments or "") if f else "",
        name=(f.name or "") if f else "",
    )


class ChatCompletionMessage(_ChatCompletionMessage):
    _tool_calls: ToolCalls | None = None

    @staticmethod
    def from_delta(delta: ChoiceDelta) -> "ChatCompletionMessage":
        return ChatCompletionMessage(
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

    def __add__(self, other: ChoiceDelta) -> "ChatCompletionMessage":
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
        return ChatCompletionMessage(
            content=((self.content or "") + (other.content or "")) or None,
            refusal=((self.refusal or "") + (other.refusal or "")) or None,
            role="assistant",
            _tool_calls=tool_calls,
        )
