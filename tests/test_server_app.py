"""Tests for the server app (OpenAI-compatible endpoints)."""

from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient
from openai import OpenAI

from swarmx import Agent, settings
from swarmx.server import create_server_app

pytestmark = pytest.mark.anyio


@pytest.fixture
async def hello_agent(model):
    settings.agents_md = []
    async with Agent(
        name="hello-agent", instructions="You are a helpful AI assistant.", model=model
    ) as agent:
        yield agent


@pytest.fixture
def client(hello_agent) -> OpenAI:
    http_client = TestClient(create_server_app(hello_agent))
    return OpenAI(
        api_key="swarmx", base_url=http_client.base_url, http_client=http_client
    )


async def test_create_server_app(model, hello_agent):
    """Test create_server_app function."""

    settings.agents_md = []
    hello_agent.nodes.add(
        Agent(name="sub_agent", instructions="Sub agent.", model=model)
    )
    http_client = TestClient(create_server_app(hello_agent))
    client = OpenAI(
        api_key="swarmx", base_url=http_client.base_url, http_client=http_client
    )
    models = client.models.list()
    model_names = [m.id for m in models]
    assert model_names == ["hello-agent", "sub_agent"]


async def test_server_app_non_streaming(client: OpenAI):
    """Test that non-streaming requests."""
    response = client.chat.completions.create(
        messages=[{"role": "user", "content": "Hello"}], model="hello-agent"
    )
    assert len(response.choices[0].message.content or "") > 0


async def test_server_app_streaming(client: OpenAI):
    """Test streaming chat completions."""
    response = client.chat.completions.create(
        messages=[{"role": "user", "content": "Hello"}],
        model="hello-agent",
        stream=True,
    )
    chunks = []
    for chunk in response:
        if len(chunk.choices) > 0:
            assert chunk.choices[0].delta.content is not None
        chunks.append(chunk)
    assert len(chunks) > 0


async def test_server_app_streaming_error_returns_done(model):
    """Ensure streaming endpoint surfaces errors and still terminates the stream."""

    settings.agents_md = []
    async with Agent(
        name="failing",
        instructions="You fail.",
        model=model,
    ) as agent:

        async def fail_run(self, *args, **kwargs):
            raise RuntimeError("boom")

        with patch.object(Agent, "run", new=fail_run):
            app = create_server_app(agent)
            with TestClient(app) as client:
                with client.stream(
                    "POST",
                    "/chat/completions",
                    json={"model": "failing", "messages": [], "stream": True},
                ) as response:
                    payloads = [chunk.decode() for chunk in response.iter_raw()]

    assert any("boom" in payload for payload in payloads)
    assert any("[DONE]" in payload for payload in payloads)


async def test_server_app_missing_model_returns_404(model):
    """Missing agent names should produce a 404 error."""

    settings.agents_md = []
    async with Agent(
        name="only",
        instructions="Respond politely.",
        model=model,
    ) as agent:
        app = create_server_app(agent)
        with TestClient(app) as client:
            response = client.post(
                "/chat/completions",
                json={
                    "model": "unknown",
                    "messages": [{"role": "user", "content": "hi"}],
                },
            )
    assert response.status_code == 404
