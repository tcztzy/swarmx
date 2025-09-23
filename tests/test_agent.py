import pytest

from swarmx import Agent, settings

pytestmark = pytest.mark.anyio


@pytest.fixture
def hello_agent(model):
    # disable local AGENTS.md for context length.
    settings.agents_md = []
    return Agent(
        name="hello-agent", instructions="You are a helpful AI assistant.", model=model
    )


async def test_agent_run(hello_agent):
    response = await hello_agent.run(messages=[{"role": "user", "content": "Hello"}])
    assert len(response) >= 1 and response[0]["role"] == "assistant"


async def test_agent_run_stream(hello_agent):
    response = await hello_agent.run(
        messages=[{"role": "user", "content": "Hello"}], stream=True
    )
    first_id = None
    async for chunk in response:
        if first_id is None:
            first_id = chunk.id
        else:
            assert chunk.id == first_id
