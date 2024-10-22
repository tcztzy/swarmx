import json
from unittest.mock import MagicMock

import pytest
from openai.types.chat.chat_completion import (
    ChatCompletion,
    ChatCompletionMessage,
    Choice,
)

from swarmx.types import ChatCompletionMessageToolCall, Function


def create_mock_response(message, function_calls=[], model="gpt-4o"):
    role = message.get("role", "assistant")
    content = message.get("content", "")
    tool_calls = (
        [
            ChatCompletionMessageToolCall(
                id="mock_tc_id",
                type="function",
                function=Function(
                    name=call.get("name", ""),
                    arguments=json.dumps(call.get("args", {})),
                ),
            )
            for call in function_calls
        ]
        if function_calls
        else None
    )

    return ChatCompletion(
        id="mock_cc_id",
        created=1234567890,
        model=model,
        object="chat.completion",
        choices=[
            Choice(
                message=ChatCompletionMessage(
                    role=role, content=content, tool_calls=tool_calls
                ),
                finish_reason="stop",
                index=0,
            )
        ],
    )


class MockOpenAIClient:
    def __init__(self):
        self.chat = MagicMock()
        self.chat.completions = MagicMock()

    def set_response(self, response: ChatCompletion):
        """
        Set the mock to return a specific response.
        :param response: A ChatCompletion response to return.
        """
        self.chat.completions.create.return_value = response

    def set_sequential_responses(self, responses: list[ChatCompletion]):
        """
        Set the mock to return different responses sequentially.
        :param responses: A list of ChatCompletion responses to return in order.
        """
        self.chat.completions.create.side_effect = responses

    def assert_create_called_with(self, **kwargs):
        self.chat.completions.create.assert_called_with(**kwargs)


@pytest.fixture
def mock_openai_client(DEFAULT_RESPONSE_CONTENT):
    m = MockOpenAIClient()
    m.set_response(
        create_mock_response({"role": "assistant", "content": DEFAULT_RESPONSE_CONTENT})
    )
    return m


@pytest.fixture
def DEFAULT_RESPONSE_CONTENT():
    return "sample response content"
