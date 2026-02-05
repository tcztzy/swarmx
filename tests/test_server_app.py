"""Tests for the server app (OpenAI-compatible endpoints)."""

import asyncio
import json
from typing import Any
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from swarmx import Swarm, settings
from swarmx.node import Node
from swarmx.server import create_server_app
from swarmx.types import MessagesState

pytestmark = pytest.mark.anyio


class DummyNode(Node):
    async def __call__(
        self,
        arguments: Any,
        *,
        context: dict[str, Any] | None = None,  # noqa: ARG002
        timeout: float | None = None,  # noqa: ARG002
        progress_callable: Any = None,
    ) -> MessagesState:
        message = {"role": "assistant", "content": "Hello!"}
        if arguments.get("stream"):
            for i, token in enumerate(["Hello", "!"]):
                if progress_callable is not None:
                    await progress_callable(
                        float(i),
                        None,
                        json.dumps(
                            {
                                "id": "chatcmpl-test",
                                "object": "chat.completion.chunk",
                                "created": 0,
                                "model": self.name,
                                "choices": [
                                    {
                                        "index": 0,
                                        "delta": {
                                            "content": token,
                                            "role": "assistant",
                                        },
                                        "finish_reason": None,
                                    }
                                ],
                            }
                        ),
                    )
                await asyncio.sleep(0)
        return {"messages": [message]}


@pytest.fixture
async def hello_agent():
    settings.agents_md = []
    return DummyNode(name="hello-agent", parameters={})


@pytest.fixture
async def hello_swarm(hello_agent):
    settings.agents_md = []
    sub_agent = DummyNode(name="sub_agent", parameters={})
    return Swarm(
        name="swarm",
        parameters={},
        nodes={hello_agent.name: hello_agent, sub_agent.name: sub_agent},
        edges=[],
        root=hello_agent.name,
    )


async def test_create_server_app_lists_models(hello_swarm):
    """Test create_server_app function."""
    app = create_server_app(hello_swarm)
    with TestClient(app) as client:
        response = client.get("/models")
    payload = response.json()
    model_names = [m["id"] for m in payload["data"]]
    assert model_names == ["hello-agent", "sub_agent", "swarm"]


async def test_server_app_non_streaming(hello_agent):
    """Test that non-streaming requests."""
    app = create_server_app(hello_agent)
    with TestClient(app) as client:
        response = client.post(
            "/chat/completions",
            json={
                "model": "hello-agent",
                "messages": [{"role": "user", "content": "Hello"}],
                "stream": False,
            },
        )
    payload = response.json()
    assert payload["choices"][0]["message"]["content"]


async def test_server_app_streaming(hello_agent):
    """Test streaming chat completions."""
    app = create_server_app(hello_agent)
    with TestClient(app) as client:
        with client.stream(
            "POST",
            "/chat/completions",
            json={
                "model": "hello-agent",
                "messages": [{"role": "user", "content": "Hello"}],
                "stream": True,
            },
        ) as response:
            payloads = [chunk.decode() for chunk in response.iter_raw()]

    content_parts: list[str] = []
    done_seen = False
    for payload in payloads:
        for line in payload.splitlines():
            if not line.startswith("data: "):
                continue
            data = line.removeprefix("data: ").strip()
            if data == "[DONE]":
                done_seen = True
                continue
            chunk = json.loads(data)
            if chunk.get("choices"):
                delta = chunk["choices"][0].get("delta", {})
                if delta.get("content"):
                    content_parts.append(delta["content"])

    assert done_seen
    assert "".join(content_parts).strip()


async def test_server_app_streaming_error_returns_done(hello_agent):
    """Ensure streaming endpoint surfaces errors and still terminates the stream."""

    settings.agents_md = []

    async def fail_call(self, *args, **kwargs):
        raise RuntimeError("boom")

    with patch.object(DummyNode, "__call__", new=fail_call):
        app = create_server_app(hello_agent)
        with TestClient(app) as client:
            with client.stream(
                "POST",
                "/chat/completions",
                json={
                    "model": "hello-agent",
                    "messages": [{"role": "user", "content": "hi"}],
                    "stream": True,
                },
            ) as response:
                payloads = [chunk.decode() for chunk in response.iter_raw()]

    assert any("boom" in payload for payload in payloads)
    assert any("[DONE]" in payload for payload in payloads)


async def test_server_app_missing_model_returns_404():
    """Missing agent names should produce a 404 error."""

    settings.agents_md = []
    app = create_server_app(DummyNode(name="only", parameters={}))
    with TestClient(app) as client:
        response = client.post(
            "/chat/completions",
            json={
                "model": "unknown",
                "messages": [{"role": "user", "content": "hi"}],
            },
        )
    assert response.status_code == 404
