from swarmx import function_to_json

from .functions import print_account_details


def test_function_to_openai_tool():
    assert function_to_json(print_account_details) == {
        "function": {
            "description": "Simple function to print account details.",
            "name": "print_account_details",
            "parameters": {"properties": {}, "type": "object"},
        },
        "type": "function",
    }
