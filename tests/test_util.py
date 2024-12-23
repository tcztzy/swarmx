import pytest

from swarmx import (
    ChatCompletionMessageToolCall,
    Function,
    Swarm,
    function_to_json,
)


def test_basic_tool_call():
    def test_tool(arg1, arg2):
        return f"Tool called with {arg1} and {arg2}"

    client = Swarm()

    response = client.handle_tool_calls(
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
        {"test_tool": test_tool},
        {},
    )
    assert len(response.messages) == 1
    assert response.messages[0]["content"] == "Tool called with value1 and value2"

    invalid_function = Function(
        name="test_tool",
        arguments="{}",
    )
    with pytest.raises(TypeError):
        response = client.handle_tool_calls(
            [
                ChatCompletionMessageToolCall(
                    id="mock_tc_id",
                    type="function",
                    function=invalid_function,
                )
            ],
            {"test_tool": test_tool},
            {},
        )


def test_function_to_openai_tool():
    def print_account_details(context_variables: dict):
        """Simple function to print account details."""
        user_id = context_variables.get("user_id", None)
        name = context_variables.get("name", None)
        return f"Account Details: {name} {user_id}"

    t = function_to_json(print_account_details)  # type: ignore
    assert t == {
        "function": {
            "description": "Simple function to print account details.",
            "name": "print_account_details",
            "parameters": {"properties": {}, "type": "object"},
        },
        "type": "function",
    }
