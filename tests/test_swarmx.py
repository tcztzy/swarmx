from typing import Annotated

from deepeval import assert_test
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from openai.types.chat.chat_completion_chunk import (
    ChoiceDelta,
    ChoiceDeltaToolCall,
    ChoiceDeltaToolCallFunction,
)

from swarmx import Agent, Result, Swarm, function_to_json, merge_chunk


def test_merge_content_string():
    message = {"role": "assistant", "content": "Hello"}
    delta = ChoiceDelta(content=" world")
    merge_chunk(message, delta)
    assert message["content"] == "Hello world"


def test_merge_content_list():
    message = {"role": "assistant", "content": [{"type": "text", "text": "Hi"}]}
    delta = ChoiceDelta(content=" there")
    merge_chunk(message, delta)
    assert message["content"] == [
        {"type": "text", "text": "Hi"},
        {"type": "text", "text": " there"},
    ]


def test_merge_refusal():
    message = {"role": "assistant", "refusal": "No"}
    delta = ChoiceDelta(refusal=" way")
    merge_chunk(message, delta)
    assert message["refusal"] == "No way"


def test_new_tool_call():
    message = {"role": "assistant", "tool_calls": []}
    delta = ChoiceDelta(
        tool_calls=[
            ChoiceDeltaToolCall(
                index=0,
                id="call_1",
                function=ChoiceDeltaToolCallFunction(arguments="arg", name="func"),
            )
        ]
    )
    merge_chunk(message, delta)
    assert message["tool_calls"] == [
        {
            "id": "call_1",
            "function": {"arguments": "arg", "name": "func"},
            "type": "function",
        }
    ]


def test_update_existing_tool_call():
    message = {
        "role": "assistant",
        "tool_calls": [
            {
                "id": "call_1",
                "function": {"arguments": "arg1", "name": "func1"},
                "type": "function",
            }
        ],
    }
    delta = ChoiceDelta(
        tool_calls=[
            ChoiceDeltaToolCall(
                index=0,
                function=ChoiceDeltaToolCallFunction(arguments="arg2", name="func2"),
            )
        ]
    )
    merge_chunk(message, delta)
    assert message["tool_calls"][0]["function"]["arguments"] == "arg1arg2"
    assert message["tool_calls"][0]["function"]["name"] == "func2"


def test_update_tool_call_id():
    message = {
        "role": "assistant",
        "tool_calls": [
            {
                "id": "call_1",
                "function": {"arguments": "", "name": ""},
                "type": "function",
            }
        ],
    }
    delta = ChoiceDelta(
        tool_calls=[
            ChoiceDeltaToolCall(
                index=0, id="call_2", function=ChoiceDeltaToolCallFunction()
            )
        ]
    )
    merge_chunk(message, delta)
    assert message["tool_calls"][0]["id"] == "call_2"


def test_multiple_tool_calls():
    message = {"role": "assistant", "tool_calls": []}
    delta = ChoiceDelta(
        tool_calls=[
            ChoiceDeltaToolCall(
                index=0,
                id="call_1",
                function=ChoiceDeltaToolCallFunction(arguments="arg1", name="func1"),
            ),
            ChoiceDeltaToolCall(
                index=1,
                id="call_2",
                function=ChoiceDeltaToolCallFunction(arguments="arg2", name="func2"),
            ),
        ]
    )
    merge_chunk(message, delta)
    assert len(message["tool_calls"]) == 2
    assert message["tool_calls"][0]["id"] == "call_1"
    assert message["tool_calls"][1]["id"] == "call_2"


def test_handoff(client: Swarm, skip_deepeval: bool, model: str):
    english_agent = Agent(
        name="English Agent",
        instructions="You only speak English.",
        functions=["tests.functions.transfer_to_spanish_agent"],
    )
    message_input = "Hola. ¿Como estás?"
    response = client.run(
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


def sample(a: int, b: str) -> str:
    """Sample function"""
    return f"{a}{b}"


async def async_func(c: float) -> Agent:
    return Agent()


def no_params() -> dict:
    return {}


class TestFunctionToJson:
    def test_basic_function_conversion(self):
        tool = function_to_json(sample)
        assert tool["type"] == "function"
        assert tool["function"]["name"] == "tests.test_swarmx.sample"
        assert tool["function"]["description"] == "Sample function"
        assert tool["function"]["parameters"] == {
            "type": "object",
            "properties": {"a": {"type": "integer"}, "b": {"type": "string"}},
            "required": ["a", "b"],
        }

    def test_context_variables_exclusion(self):
        def with_context(a: int, context_variables: dict) -> str:
            return ""

        tool = function_to_json(with_context)
        assert "context_variables" not in tool["function"]["parameters"]["properties"]

    def test_async_function_handling(self):
        tool = function_to_json(async_func)
        assert tool["function"]["name"] == "tests.test_swarmx.async_func"
        assert tool["function"]["parameters"] == {
            "type": "object",
            "properties": {"c": {"type": "number"}},
            "required": ["c"],
        }

    def test_function_with_complex_types(self):
        def complex_types(
            d: Annotated[str, "metadata"], e: list[dict[str, int]]
        ) -> Result:
            """Complex types"""
            return Result(content=[])

        tool = function_to_json(complex_types)
        assert tool["function"]["parameters"] == {
            "type": "object",
            "properties": {
                "d": {"type": "string"},
                "e": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": {"type": "integer"},
                    },
                },
            },
            "required": ["d", "e"],
        }

    def test_function_with_no_parameters(self):
        tool = function_to_json(no_params)
        assert tool["function"]["parameters"] == {"type": "object", "properties": {}}
