import os
import sys
from pathlib import Path
from typing import Any, AsyncGenerator, Unpack
from unittest.mock import AsyncMock

import openai
import pytest
from mcp.client.stdio import StdioServerParameters
from openai import AsyncOpenAI, AsyncStream
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionChunk,
)
from openai.types.chat.completion_create_params import CompletionCreateParamsBase
from pydantic import BaseModel

from swarmx.mcp_client import CLIENT_REGISTRY


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
            stream: bool = False, **kw: Unpack[CompletionCreateParamsBase]
        ) -> Any:
            # Call the actual API.
            parameters = kw | {"stream": stream}
            for thread in self._threads:
                if thread.parameters == parameters:
                    if isinstance(thread.completion, list):

                        async def gen():
                            for chunk in thread.completion:
                                yield chunk

                        return gen()
                    return thread.completion
            if stream:
                # ``response`` is an async generator of chunks.
                response = await create(stream=stream, **kw)

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
                completion = await create(stream=stream, **kw)
                thread = Thread(parameters=parameters, completion=completion)
                with self._threads_path.open(mode="a") as f:
                    f.write(thread.model_dump_json(exclude_unset=True) + "\n")
            return completion

        self.chat.completions.create = AsyncMock(side_effect=_create)


@pytest.fixture(autouse=True)
def cached_openai(monkeypatch: pytest.MonkeyPatch):
    """Patch ``swarmx.agent.DEFAULT_CLIENT`` with the mock.

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
    try:
        yield StdioServerParameters(
            command=sys.executable,
            args=[str(script)],
            env=dict(os.environ),
            cwd=str(Path(__file__).parent.parent),
            encoding="utf-8",
            encoding_error_handler="strict",
        )
    finally:
        await CLIENT_REGISTRY.close()
