import json
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from openai.types.chat.chat_completion import ChatCompletion, Choice
from openai.types.chat.chat_completion_chunk import (
    ChatCompletionChunk,
    ChoiceDelta,
    ChoiceDeltaToolCall,
    ChoiceDeltaToolCallFunction,
)
from openai.types.chat.chat_completion_chunk import Choice as ChunkChoice
from openai.types.chat.chat_completion_message import ChatCompletionMessage

from swarmx import Swarm


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--deepeval", action="store_true", default=False, help="Run deepeval tests"
    )
    parser.addoption(
        "--openai",
        action="store_true",
        default=False,
        help="Run tests with actual OpenAI API calls instead of mocks",
    )
    parser.addoption(
        "--model", action="store", default="gpt-4o", help="Model to use for tests"
    )


@pytest.fixture
def skip_deepeval(pytestconfig: pytest.Config):
    return not pytestconfig.getoption("deepeval")


@pytest.fixture
def is_mocking(pytestconfig: pytest.Config):
    return not pytestconfig.getoption("openai")


@pytest.fixture
def anyio_backend():
    return "asyncio"


def create_mock_streaming_response(
    message,
    function_calls=[],
    model="gpt-4o",
):
    def _generator():
        tokens = message.get("content", "").split()
        for i, token in enumerate(tokens):
            yield ChatCompletionChunk(
                id="mock_cc_id",
                created=int(datetime.now().timestamp()),
                model=model,
                object="chat.completion.chunk",
                choices=[
                    ChunkChoice(
                        delta=ChoiceDelta(
                            content=token,
                            tool_calls=[
                                ChoiceDeltaToolCall(
                                    index=0,
                                    id="mock_tc_id",
                                    type="function",
                                    function=ChoiceDeltaToolCallFunction(
                                        name=call.get("name", ""),
                                        arguments=json.dumps(call.get("args", {})),
                                    ),
                                )
                                for call in function_calls
                            ]
                            if len(function_calls) > 0
                            else None,
                            role=message.get("role", "assistant") if i == 0 else None,
                        ),
                        finish_reason="stop" if i == len(tokens) - 1 else None,
                        index=0,
                    )
                ],
            )

    return _generator()


def create_mock_response(message, model="gpt-4o"):
    message = ChatCompletionMessage.model_validate(message)

    return ChatCompletion(
        id="mock_cc_id",
        created=int(datetime.now().timestamp()),
        model=model,
        object="chat.completion",
        choices=[
            Choice(
                message=message,
                finish_reason="tool_calls" if message.tool_calls else "stop",
                index=0,
            )
        ],
    )


class MockAsyncOpenAI:
    def __init__(self):
        self.chat = MagicMock()
        self.chat.completions = MagicMock()
        self.chat.completions.create = AsyncMock()

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


@pytest.fixture
def mock_openai():
    return MockAsyncOpenAI()


@pytest.fixture
def client(
    mock_openai: MockAsyncOpenAI, is_mocking: bool, request: pytest.FixtureRequest
):
    c = Swarm()
    if is_mocking:
        match request.node.name:
            case "test_mcp_tool_call":
                messages = json.loads(
                    (
                        Path(__file__).parent / "threads" / "mcp_tool_call.json"
                    ).read_text()
                )
                mock_openai.set_sequential_responses(
                    [create_mock_response(message) for message in messages]
                    + [
                        create_mock_response(
                            {
                                "content": datetime.now(timezone.utc).strftime(
                                    "%H:%M:%S"
                                ),
                                "role": "assistant",
                            }
                        )
                    ]
                )
            case name if name.startswith("test_"):
                messages = json.loads(
                    (
                        Path(__file__).parent
                        / "threads"
                        / f"{name.replace('test_', '')}.json"
                    ).read_text()
                )
                mock_openai.set_sequential_responses(
                    [create_mock_response(message) for message in messages]
                )
        c._client = mock_openai  # type: ignore
    return c


@pytest.fixture
def model(pytestconfig: pytest.Config):
    return pytestconfig.getoption("model")
