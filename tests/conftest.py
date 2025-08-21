import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncGenerator
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
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
from openai.types.chat.chat_completion_chunk import Choice as ChunkChoice
from openai.types.chat.chat_completion_message import ChatCompletionMessage


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--openai",
        action="store_true",
        default=False,
        help="Run tests with actual OpenAI API calls instead of mocks",
    )
    parser.addoption(
        "--model",
        action="store",
        default="openai/gpt-oss-120b",
        help="Model to use for tests",
    )


@pytest.fixture
def is_mocking(pytestconfig: pytest.Config):
    return not pytestconfig.getoption("openai")


@pytest.fixture
def anyio_backend():
    return "asyncio"


def create_mock_stream_response(
    message: ChatCompletionAssistantMessageParam,
    model: str,
):
    content = message.get("content")
    match content:
        case str():
            tokens = re.split(r"(\s+)", content)
        case None:
            tokens = []
        case _:
            tokens = [part["text"] for part in content if part["type"] == "text"]

    tool_calls = list(message.get("tool_calls", []))
    message_id = str(uuid4())

    async def _generator():
        for i, token in enumerate(tokens):
            yield ChatCompletionChunk(
                id=message_id,
                created=int(datetime.now().timestamp()),
                model=model,
                object="chat.completion.chunk",
                choices=[
                    ChunkChoice(
                        delta=ChoiceDelta(
                            content=token,
                            role="assistant" if i == 0 else None,
                        ),
                        finish_reason="stop"
                        if i == len(tokens) - 1 and not tool_calls
                        else None,
                        index=0,
                    )
                ],
            )
        for i, call in enumerate(tool_calls):
            yield ChatCompletionChunk(
                id=message_id,
                created=int(datetime.now().timestamp()),
                model=model,
                object="chat.completion.chunk",
                choices=[
                    ChunkChoice(
                        index=0,
                        delta=ChoiceDelta(
                            role="assistant" if i == 0 else None,
                            tool_calls=[
                                ChoiceDeltaToolCall(
                                    index=i,
                                    id=f"{message_id}_{i}",
                                    type="function",
                                    function=ChoiceDeltaToolCallFunction(
                                        name=call["function"]["name"],
                                        arguments=call["function"]["arguments"],
                                    ),
                                )
                            ],
                        ),
                    )
                ],
            )

    return _generator()


def create_mock_response(message, model):
    message = ChatCompletionMessage.model_validate(message)

    return ChatCompletion.model_validate(
        {
            "id": str(uuid4()),
            "created": int(datetime.now().timestamp()),
            "model": model,
            "object": "chat.completion",
            "choices": [
                {
                    "message": message,
                    "finish_reason": "tool_calls" if message.tool_calls else "stop",
                    "index": 0,
                }
            ],
        }
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

    def set_sequential_responses(
        self,
        responses: list[ChatCompletion]
        | list[AsyncGenerator[ChatCompletionChunk, Any]],
    ):
        """
        Set the mock to return different responses sequentially.
        :param responses: A list of ChatCompletion responses to return in order.
        """
        self.chat.completions.create.side_effect = responses


@pytest.fixture
def mock_openai():
    return MockAsyncOpenAI()


@pytest.fixture(autouse=True)
def client(
    mock_openai: MockAsyncOpenAI,
    is_mocking: bool,
    request: pytest.FixtureRequest,
    model: str,
):
    if is_mocking:
        match request.node.name:
            case "test_mcp_tool_call":
                messages = json.loads(
                    (
                        Path(__file__).parent / "threads" / "mcp_tool_call.json"
                    ).read_text()
                )
                mock_openai.set_sequential_responses(
                    [
                        create_mock_response(message, model)
                        for message in messages
                        if message["role"] == "assistant"
                    ]
                    + [
                        create_mock_response(
                            {
                                "content": datetime.now(timezone.utc).strftime(
                                    "%H:%M:%S"
                                ),
                                "role": "assistant",
                            },
                            model,
                        )
                    ]
                )
            case name if name.startswith("test_"):
                n = name.replace("test_", "")
                if n.endswith("_stream"):
                    n = n.replace("_stream", "")
                if (
                    json_file := Path(__file__).parent / "threads" / f"{n}.json"
                ).exists():
                    messages = json.loads(json_file.read_text())
                    if name.endswith("_stream"):
                        mock_openai.set_sequential_responses(
                            [
                                create_mock_stream_response(message, model)
                                for message in messages
                                if message["role"] == "assistant"
                            ]
                        )
                    else:
                        mock_openai.set_sequential_responses(
                            [
                                create_mock_response(message, model)
                                for message in messages
                                if message["role"] == "assistant"
                            ]
                        )
                else:
                    messages = [
                        {
                            "role": "assistant",
                            "content": "Hello World",
                        }
                    ]
                    mock_openai.set_sequential_responses(
                        [
                            (
                                create_mock_response
                                if not name.endswith("_stream")
                                else create_mock_stream_response
                            )(message, model)  # type: ignore
                            for message in messages
                        ]
                    )
        # Only patch when mocking
        with patch("swarmx.agent.DEFAULT_CLIENT", mock_openai):
            yield
    else:
        yield


@pytest.fixture
def model(pytestconfig: pytest.Config):
    return pytestconfig.getoption("model")
