import datetime

import pytest
from deepeval import assert_test
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from mcp.client.stdio import StdioServerParameters
from pytest import fixture

from swarmx import Agent, AsyncSwarm, Swarm

pytestmark = pytest.mark.anyio


def openai_available():
    try:
        Swarm().client.models.list()
        return True
    except Exception:
        return False


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


@pytest.mark.skipif(not openai_available(), reason="OpenAI API not available.")
def test_handoff(client: Swarm, english_agent: Agent):
    message_input = "Hola. ¿Como estás?"
    response = client.run(
        english_agent,
        messages=[{"role": "user", "content": message_input}],
        model="llama3.2",
    )
    last_message = response.messages[-1]
    content = last_message.get("content")
    if isinstance(content, str):
        actual_output = content
    elif content is None:
        actual_output = ""
    else:
        # Handle case where content is an iterable of content parts
        actual_output = " ".join(part["text"] for part in content if "text" in part)

    test_case = LLMTestCase(message_input, actual_output)
    spanish_detection = GEval(
        name="Spanish Detection",
        criteria="Spanish Detection - the likelihood of the agent responding in Spanish.",
        evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT],
        threshold=0.85,  # interesting, Llama rarely generate likelihoods above 0.9
    )
    assert_test(test_case, [spanish_detection])
    assert response.agent != english_agent


async def test_mcp_tool_call():
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
                {
                    "role": "user",
                    "content": "What time is it now? UTC time is okay. "
                    "You should only answer time in %H:%M:%S format without "
                    "any other characters, e.g. 12:34:56",
                }
            ],
        )
        message = response.messages[-1]
        assert message.get("name") == "Agent"
        now = datetime.datetime.now(datetime.timezone.utc)
        content = message.get("content")
        assert isinstance(content, str)
        answer_time = datetime.datetime.strptime(content, "%H:%M:%S").replace(
            tzinfo=datetime.timezone.utc
        )
        assert answer_time - now < datetime.timedelta(seconds=1)
