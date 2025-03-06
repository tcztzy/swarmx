from typing import Any

import pytest

from swarmx import (
    __CTX_VARS_NAME__,
    check_instructions,
    does_function_need_context,
    validate_tool,
)


def test_validate_tool_warning():
    """Test that validate_tool issues a warning for unannotated functions."""

    # Define a function without return annotation
    def unannotated_func():
        return "Hello"

    # Capture the warning
    with pytest.warns(
        FutureWarning, match="Agent function return type is not annotated"
    ):
        tool = validate_tool(unannotated_func)

    # Check that the tool was still created properly
    assert tool["function"]["name"] == "unannotated_func"


def test_does_function_need_context_edge_cases():
    """Test edge cases for the does_function_need_context function."""
    # Test with a function that has a parameter named exactly __CTX_VARS_NAME__
    assert __CTX_VARS_NAME__ == "context_variables"

    def func_with_exact_ctx_name(context_variables: dict):
        return "Hello"

    assert does_function_need_context(func_with_exact_ctx_name)

    # Test with a function that has no parameters
    def func_no_params():
        return "Hello"

    assert not does_function_need_context(func_no_params)

    # Test with a class method that takes self
    class TestClass:
        def method(self, name: str):
            return f"Hello {name}"

    assert not does_function_need_context(TestClass().method)

    # Test with a static method
    class StaticClass:
        @staticmethod
        def static_method(context_variables: dict):
            return "Hello"

    assert does_function_need_context(StaticClass.static_method)


def test_check_instructions_edge_cases():
    """Test edge cases for the check_instructions function."""

    # Test with a complex callable that uses type hints
    def complex_instructions(context_variables: dict[str, Any]) -> str:
        return f"Hello {context_variables.get('name', 'world')}"

    result = check_instructions(complex_instructions)
    assert callable(result)
    assert result({"name": "test"}) == "Hello test"

    # Test with a callable that takes **kwargs
    def kwargs_instructions(name: str):
        return f"Hello {name}"

    result = check_instructions(kwargs_instructions)
    assert callable(result)
    assert result(**{"name": "test"}) == "Hello test"


def test_check_instructions_wrong_signature():
    """Test that check_instructions validates the signature properly."""

    # Test with a function that has the wrong parameter type
    def wrong_type_func(context_variables: int) -> str:
        return "Hello"

    with pytest.raises(ValueError):
        check_instructions(wrong_type_func)

    # Test with a function that has too many required parameters
    def too_many_params(context_variables: dict, extra_param: str) -> str:
        return f"Hello {extra_param}"

    with pytest.raises(ValueError):
        check_instructions(too_many_params)
