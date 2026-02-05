import copy
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, AsyncGenerator, Unpack
from unittest.mock import AsyncMock

import httpx
import openai
import pytest
from mcp.client.stdio import StdioServerParameters
from openai import AsyncOpenAI, AsyncStream
from openai._types import (
    Body,
    Headers,
    NotGiven,
    Query,
    not_given,
)
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionChunk,
    CompletionCreateParams,
)
from openai.types.chat.completion_create_params import CompletionCreateParamsBase
from pydantic import BaseModel


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--model",
        action="store",
        default="gpt-oss:20b",
        help="Model to use for tests",
    )


@pytest.fixture
def anyio_backend():
    return "asyncio"


class Thread(BaseModel):
    parameters: dict  # pydantic/pydantic#10640 & #9266
    completion: ChatCompletion | list[ChatCompletionChunk]


class GetCurrentTimeResult(BaseModel):
    timezone: str
    datetime: datetime
    is_dst: bool


def message_eq(message1, message2):
    if message1["role"] == message2["role"] == "tool":
        try:
            result1 = GetCurrentTimeResult.model_validate_json(message1["content"])
            result2 = GetCurrentTimeResult.model_validate_json(message2["content"])
            return (
                message1["tool_call_id"] == message2["tool_call_id"]
                and result1.timezone == result2.timezone
                and result1.is_dst == result2.is_dst
            )
        except Exception:
            return False
    else:
        return message1 == message2


def cache_hit(
    cached_parameter: CompletionCreateParams,
    parameter: CompletionCreateParams,
):
    cached_parameter = copy.deepcopy(cached_parameter)
    parameter = copy.deepcopy(parameter)
    cached_messages = cached_parameter.pop("messages")
    messages = parameter.pop("messages")
    return (
        cached_parameter == parameter
        and len(cached_messages) == len(messages)  # type: ignore
        and all(
            message_eq(m1, m2)
            for m1, m2 in zip(cached_messages, messages)  # type: ignore
        )
    )


class CachedAsyncOpenAI(AsyncOpenAI):
    def __init__(self, **params):
        super().__init__(**params)
        # Load fixtures from a single JSON file if it exists.
        self._threads_path = Path(__file__).parent / "threads.jsonl"
        self._threads: list[Thread] = []
        try:
            self._threads = [
                Thread.model_validate_json(line)
                for line in self._threads_path.read_text().splitlines()
                if len(line.strip()) > 0
            ]
        except Exception:
            pass
        create = self.chat.completions.create

        async def _create(
            stream: bool = False,
            extra_headers: Headers | None = None,
            extra_query: Query | None = None,
            extra_body: Body | None = None,
            timeout: float | httpx.Timeout | None | NotGiven = not_given,
            **kw: Unpack[CompletionCreateParamsBase],
        ) -> Any:
            # Call the actual API.
            parameters = kw | {"stream": stream}
            for thread in self._threads:
                if cache_hit(thread.parameters, parameters):  # type: ignore
                    if isinstance(thread.completion, list):

                        async def gen():
                            for chunk in thread.completion:
                                yield chunk

                        return gen()
                    return thread.completion
            if stream:
                # ``response`` is an async generator of chunks.
                response = await create(
                    stream=stream,
                    **kw,
                    extra_headers=extra_headers,
                    extra_query=extra_query,
                    extra_body=extra_body,
                    timeout=timeout,
                )

                async def _gen(
                    response: AsyncStream[ChatCompletionChunk],
                ) -> AsyncGenerator[ChatCompletionChunk, None]:
                    chunks = []
                    async for chunk in response:
                        yield chunk
                        chunks.append(chunk)
                    thread = Thread(parameters=parameters, completion=chunks)
                    with self._threads_path.open(mode="a") as f:
                        f.write(thread.model_dump_json(exclude_unset=True) + "\n")

                completion = _gen(response)
            else:
                completion = await create(
                    **kw,
                    extra_headers=extra_headers,
                    extra_query=extra_query,
                    extra_body=extra_body,
                    timeout=timeout,
                )
                thread = Thread(parameters=parameters, completion=completion)
                with self._threads_path.open(mode="a") as f:
                    f.write(thread.model_dump_json(exclude_unset=True) + "\n")
            return completion

        self.chat.completions.create = AsyncMock(side_effect=_create)


@pytest.fixture(autouse=True)
def cached_openai(monkeypatch: pytest.MonkeyPatch):
    """Patch openai.AsyncOpenAI with the mock.

    The new ``MockAsyncOpenAI`` reads responses from ``tests/threads.json``
    and matches calls based on the supplied parameters. No per-test JSON
    handling is required.
    """
    monkeypatch.setattr(openai, "AsyncOpenAI", CachedAsyncOpenAI)

    agent_module = sys.modules.get("swarmx.agent")
    if agent_module is not None:
        monkeypatch.setattr(agent_module, "AsyncOpenAI", CachedAsyncOpenAI)


@pytest.fixture
def model(pytestconfig: pytest.Config):
    return pytestconfig.getoption("model")


@pytest.fixture
async def hooks_params():
    script = Path(__file__).parent / "hooks.py"
    return StdioServerParameters(
        command=sys.executable,
        args=[str(script)],
        env=dict(os.environ),
        cwd=str(Path(__file__).parent.parent),
        encoding="utf-8",
        encoding_error_handler="strict",
    )
