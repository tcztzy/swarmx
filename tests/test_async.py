import datetime
import json
import re

import pytest
from deepeval import assert_test
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from mcp.client.stdio import StdioServerParameters

from swarmx import TOOL_REGISTRY, Agent, AsyncSwarm

pytestmark = pytest.mark.anyio


async def test_handoff(client: AsyncSwarm, skip_deepeval: bool, model: str):
    english_agent = Agent(
        name="English Agent",
        instructions="You only speak English.",
        functions=["tests.functions.transfer_to_spanish_agent"],
    )
    message_input = "Hola. ¿Como estás?"
    response = await client.run(
        agent=english_agent,
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


async def test_mcp_tool_call(client: AsyncSwarm):
    client.mcp_servers = {
        "time": StdioServerParameters(
            command="uv",
            args=["run", "mcp-server-time", "--local-timezone", "UTC"],
        )
    }
    await TOOL_REGISTRY.add_mcp_server("time", client.mcp_servers["time"])
    assert TOOL_REGISTRY.tools[0] == {
        "function": {
            "description": "Get current time in a specific timezones",
            "name": "get_current_time",
            "parameters": {
                "properties": {
                    "timezone": {
                        "description": "IANA timezone name (e.g., 'America/New_York', 'Europe/London'). Use 'UTC' as local timezone if no timezone provided by the user.",
                        "type": "string",
                    }
                },
                "required": ["timezone"],
                "type": "object",
            },
        },
        "type": "function",
    }
    result = await TOOL_REGISTRY.call_tool("get_current_time", {"timezone": "UTC"})
    assert result.content[0].type == "text"
    json_result = json.loads(result.content[0].text)
    assert (
        "timezone" in json_result
        and "datetime" in json_result
        and "is_dst" in json_result
    )
    assert datetime.datetime.fromisoformat(
        json_result["datetime"]
    ) - datetime.datetime.now(datetime.timezone.utc) < datetime.timedelta(seconds=1)
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
    mo = re.search(r"\d{2}:\d{2}:\d{2}", content)
    assert mo is not None
    answer_time = datetime.datetime.strptime(mo.group(), "%H:%M:%S").replace(
        tzinfo=datetime.timezone.utc
    )
    assert answer_time - now < datetime.timedelta(seconds=1)
