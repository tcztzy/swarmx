import pytest
from pydantic import ValidationError

from swarmx.types import ChatCompletionMessageToolCall, Function, Tool
from swarmx.util import handle_tool_calls


def test_basic_tool_call():
    def test_tool(arg1, arg2):
        return f"Tool called with {arg1} and {arg2}"

    response = handle_tool_calls(
        [
            ChatCompletionMessageToolCall(
                id="mock_tc_id",
                type="function",
                function=Function(
                    name="test_tool",
                    arguments='{"arg1": "value1", "arg2": "value2"}',
                ),
            )
        ],
        [Tool(test_tool)],
        {},
    )
    assert len(response.messages) == 1
    assert response.messages[0]["content"] == "Tool called with value1 and value2"

    invalid_function = Function(
        name="test_tool",
        arguments="{}",
    )
    with pytest.raises(ValidationError):
        response = handle_tool_calls(
            [
                ChatCompletionMessageToolCall(
                    id="mock_tc_id",
                    type="function",
                    function=invalid_function,
                )
            ],
            [Tool(test_tool)],
            {},
        )
