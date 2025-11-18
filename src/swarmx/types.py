"""SwarmX types."""

from typing import Any, Literal, Required, TypedDict

from mcp.client.stdio import StdioServerParameters
from openai.types.chat import (
    ChatCompletionMessageParam,
    ChatCompletionToolParam,
)
from openai.types.chat.chat_completion_stream_options_param import (
    ChatCompletionStreamOptionsParam,
)
from openai.types.chat.chat_completion_tool_choice_option_param import (
    ChatCompletionToolChoiceOptionParam,
)
from openai.types.chat.completion_create_params import ResponseFormat
from pydantic import BaseModel


class SSEServer(BaseModel):
    """SSE server configuration.

    This is used to configure the MCP server for the agent.

    """

    type: Literal["sse"]
    url: str
    headers: dict[str, str] | None = None


MCPServer = StdioServerParameters | SSEServer


class CompletionCreateParams(TypedDict, total=False):
    """Parameters for Agent.run method.

    This combines chat completion parameters with SwarmX-specific options.
    """

    messages: Required[list[ChatCompletionMessageParam]]
    """Required. The messages to start the conversation with."""

    stream: bool
    """Whether to stream the response."""

    frequency_penalty: float | None
    """Number between -2.0 and 2.0. Positive values penalize new tokens based on their
    existing frequency in the text so far."""

    logprobs: bool | None
    """Whether to return log probabilities of the output tokens or not."""

    max_tokens: int | None
    """The maximum number of tokens that can be generated in the chat completion."""

    presence_penalty: float | None
    """Number between -2.0 and 2.0. Positive values penalize new tokens based on whether
    they appear in the text so far."""

    response_format: ResponseFormat | Any | None
    """An object specifying the format that the model must output."""

    seed: int | None
    """If specified, our system will make a best effort to sample deterministically."""

    stop: str | list[str] | None
    """Up to 4 sequences where the API will stop generating further tokens."""

    stream_options: ChatCompletionStreamOptionsParam | None
    """Options for streaming response. Only set this when you set stream: true."""

    temperature: float | None
    """What sampling temperature to use, between 0 and 2."""

    tool_choice: ChatCompletionToolChoiceOptionParam | None
    """Controls which (if any) tool is called by the model."""

    top_logprobs: int | None
    """An integer between 0 and 20 specifying the number of most likely tokens to return."""

    top_p: float | None
    """An alternative to sampling with temperature, called nucleus sampling."""

    tools: list[ChatCompletionToolParam]


class MessagesState(TypedDict):
    """Messages state."""

    messages: list[ChatCompletionMessageParam]
