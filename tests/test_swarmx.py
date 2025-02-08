from typing import Annotated, Any

import mcp.types
import pytest
from deepeval import assert_test
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from openai.types.chat.chat_completion_chunk import (
    ChoiceDelta,
    ChoiceDeltaToolCall,
    ChoiceDeltaToolCallFunction,
)

from swarmx import (
    Agent,
    Result,
    Swarm,
    check_function,
    check_instructions,
    function_to_json,
    handle_function_result,
    merge_chunk,
)


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


class TestCheckFunction:
    def test_valid_string_return(self):
        def valid_str() -> str:
            return "valid"

        assert check_function(valid_str) == valid_str

    def test_valid_agent_return(self):
        def valid_agent() -> Agent:
            return Agent()

        assert check_function(valid_agent) == valid_agent

    def test_valid_dict_return(self):
        def valid_dict() -> dict[str, Any]:
            return {"key": "value"}

        assert check_function(valid_dict) == valid_dict

    def test_valid_result_return(self):
        def valid_result() -> Result:
            return Result(content=[])

        assert check_function(valid_result) == valid_result

    def test_unannotated_return(self):
        def unannotated():
            return "no annotation"

        with pytest.warns(FutureWarning):
            checked = check_function(unannotated)
        assert checked == unannotated

    def test_invalid_return_type(self):
        def invalid_return() -> int:
            return 42

        with pytest.raises(TypeError) as excinfo:
            check_function(invalid_return)
        assert "must be str, Agent, dict[str, Any], or Result" in str(excinfo.value)

    def test_non_callable_input(self):
        with pytest.raises(TypeError) as excinfo:
            check_function("not a function")
        assert "must be str, Agent, dict[str, Any], or Result" in str(excinfo.value)

    def test_none_return_annotation(self):
        def none_return() -> None:
            return None

        with pytest.raises(TypeError) as excinfo:
            check_function(none_return)
        assert "must be str, Agent, dict[str, Any], or Result" in str(excinfo.value)


class TestCheckInstructions:
    def test_valid_string(self):
        instructions = "You are a helpful assistant"
        result = check_instructions(instructions)
        assert result == instructions

    def test_valid_callable(self):
        def valid_func(context_variables: dict) -> str:
            return "Hello"

        result = check_instructions(valid_func)
        assert result == valid_func

    def test_callable_missing_context_var(self):
        def invalid_func(a: int) -> str:
            return str(a)

        with pytest.raises(ValueError) as excinfo:
            check_instructions(invalid_func)
        assert "context_variable" in str(excinfo.value)

    def test_callable_wrong_param_name(self):
        def wrong_name(params: dict) -> str:
            return ""

        with pytest.raises(ValueError) as excinfo:
            check_instructions(wrong_name)
        assert "context_variables" in str(excinfo.value)

    def test_callable_bad_annotation(self):
        def bad_anno(context_variables: int) -> str:
            return ""

        with pytest.raises(ValueError) as excinfo:
            check_instructions(bad_anno)
        assert str(excinfo.value).startswith(
            "Instructions should be a string or a callable takes "
        )

    def test_non_callable_non_string(self):
        with pytest.raises(ValueError) as excinfo:
            check_instructions(42)
        assert "a string or a callable" in str(excinfo.value)

    def test_generic_dict_annotation(self):
        def generic_anno(context_variables: dict[str, str]) -> str:
            return ""

        result = check_instructions(generic_anno)
        assert result == generic_anno

    def test_no_annotation_callable(self):
        def no_anno(context_variables):
            return ""

        result = check_instructions(no_anno)
        assert result == no_anno


class TestHandleFunctionResult:
    def test_result_instance(self):
        result = Result(content=[])
        assert handle_function_result(result) == result

    def test_agent_instance(self):
        agent = Agent()
        result = handle_function_result(agent)
        assert isinstance(result, Result)
        assert result.agent == agent

    def test_dict_result(self):
        d = {"key": "value"}
        result = handle_function_result(d)
        assert result.meta == d
        assert result.content == []

    def test_mcp_result(self):
        mcp_result = mcp.types.CallToolResult(content=[])
        result = handle_function_result(mcp_result)
        assert isinstance(result, Result)
        assert result.model_dump(exclude={"agent"}) == mcp_result.model_dump()

    def test_string_result(self):
        s = "test"
        result = handle_function_result(s)
        assert len(result.content) == 1
        assert result.content[0].text == "test"
