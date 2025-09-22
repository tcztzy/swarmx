"""Tests for the server app (OpenAI-compatible endpoints)."""

from fastapi.testclient import TestClient


def test_create_server_app(model):
    """Test create_server_app function."""
    from openai import OpenAI

    from swarmx import Agent, settings
    from swarmx.server import create_server_app

    settings.agents_md = []
    agent = Agent(
        name="test_agent",
        instructions="Test instructions",
        model=model,
        nodes={Agent(name="sub_agent", instructions="Sub agent.", model=model)},
    )
    http_client = TestClient(create_server_app(agent))
    client = OpenAI(
        api_key="swarmx", base_url=http_client.base_url, http_client=http_client
    )
    models = client.models.list()
    model_names = [m.id for m in models]
    assert model_names == ["test_agent", "sub_agent"]


def test_server_app_non_streaming(model):
    """Test that non-streaming requests."""
    from openai import OpenAI

    from swarmx import Agent, settings
    from swarmx.server import create_server_app

    settings.agents_md = []
    agent = Agent(name="test_agent", instructions="Test instructions", model=model)
    http_client = TestClient(create_server_app(agent))
    client = OpenAI(
        api_key="swarmx", base_url=http_client.base_url, http_client=http_client
    )
    response = client.chat.completions.create(
        messages=[{"role": "user", "content": "Hello"}], model="test_agent"
    )
    assert len(response.choices[0].message.content or "") > 0


def test_server_app_streaming(model):
    """Test streaming chat completions."""
    from openai import OpenAI

    from swarmx import Agent, settings
    from swarmx.server import create_server_app

    settings.agents_md = []
    agent = Agent(name="test_agent", instructions="Test instructions", model=model)
    http_client = TestClient(create_server_app(agent))
    client = OpenAI(
        api_key="swarmx", base_url=http_client.base_url, http_client=http_client
    )
    response = client.chat.completions.create(
        messages=[{"role": "user", "content": "Hello"}], model="test_agent", stream=True
    )
    chunks = []
    for chunk in response:
        if len(chunk.choices) > 0:
            assert chunk.choices[0].delta.content is not None
        chunks.append(chunk)
    assert len(chunks) > 0
