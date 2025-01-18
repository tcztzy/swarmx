import datetime

import pytest
from deepeval import assert_test
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from mcp.client.stdio import StdioServerParameters
from pytest import fixture

from swarmx import Agent, AsyncSwarm, Swarm


def no_openai_available():
    try:
        Swarm().client.models.list()
        return False
    except Exception:
        return True


@fixture
def client():
    return Swarm()


@fixture
def spanish_agent():
    return Agent(
        name="Spanish Agent",
        instructions="You only speak Spanish.",
    )


@fixture
def english_agent(spanish_agent):
    def transfer_to_spanish_agent():
        """Transfer spanish speaking users immediately."""
        return spanish_agent

    return Agent(
        name="English Agent",
        instructions="You only speak English.",
        functions=[transfer_to_spanish_agent],
    )


@pytest.mark.skipif(no_openai_available(), reason="OpenAI API not available.")
def test_handoff(client: Swarm, english_agent: Agent):
    message_input = "Hola. ¿Como estás?"
    reponse = client.run(
        english_agent,
        messages=[{"role": "user", "content": message_input}],
        model="llama3.2",
    )
    test_case = LLMTestCase(
        message_input,
        reponse.messages[-1]["content"],
    )
    spanish_detection = GEval(
        name="Spanish Detection",
        criteria="Spanish Detection - the likelihood of the agent responding in Spanish.",
        evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT],
        threshold=0.85,  # interesting, Llama rarely generate likelihoods above 0.9
    )
    assert_test(test_case, [spanish_detection])
    assert reponse.agent != english_agent


@pytest.mark.parametrize("anyio_backend", ["asyncio"])
async def test_mcp_tool_call(anyio_backend):
    async with AsyncSwarm(
        mcp_servers={
            "time": StdioServerParameters(
                command="uvx",
                args=["mcp-server-time", "--local-timezone", "UTC"],
            )
        }
    ) as client:
        agent = Agent()
        response = await client.run(
            agent=agent,
            model="llama3.2",
            messages=[
                {"role": "user", "content": "What time is it now? UTC time is okay."}
            ],
        )
        message = response.messages[-1]
        assert message.get("name") == "Agent"
        now = datetime.datetime.now(datetime.timezone.utc)
        content = message.get("content")
        assert isinstance(content, str)
        assert now.strftime("%H:%M") in content
