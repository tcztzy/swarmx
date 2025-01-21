import pytest
from deepeval import assert_test
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams

from swarmx import Agent, Swarm

pytestmark = pytest.mark.anyio


def test_handoff(handoff_client: Swarm, skip_deepeval: bool, model: str):
    english_agent = Agent(
        name="English Agent",
        instructions="You only speak English.",
        functions=["tests.functions.transfer_to_spanish_agent"],
    )
    message_input = "Hola. ¿Como estás?"
    response = handoff_client.run(
        english_agent,
        messages=[{"role": "user", "content": message_input}],
        model=model,
    )
    assert response.agent is not None and response.agent.name == "Spanish Agent"
    if skip_deepeval:
        return
    content = response.messages[-1].get("content")
    if isinstance(content, str):
        actual_output = content
    elif content is None:
        actual_output = ""
    else:
        # Handle case where content is an iterable of content parts
        actual_output = "".join(part["text"] for part in content if "text" in part)

    test_case = LLMTestCase(message_input, actual_output)
    spanish_detection = GEval(
        name="Spanish Detection",
        criteria="Spanish Detection - the likelihood of the agent responding in Spanish.",
        evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT],
        threshold=0.85,  # interesting, Llama rarely generate likelihoods above 0.9
    )
    assert_test(test_case, [spanish_detection])
