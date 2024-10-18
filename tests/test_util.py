from swarmx.util import SwarmXGenerateJsonSchema, function_to_json, function_to_model


def test_basic_function():
    def basic_function(arg1, arg2):
        return arg1 + arg2

    model = function_to_model(basic_function)
    assert model.__name__ == "basic_function"
    assert model.__annotations__ == {"arg1": str, "arg2": str}
    assert model.model_json_schema(schema_generator=SwarmXGenerateJsonSchema) == {
        "type": "object",
        "properties": {
            "arg1": {"type": "string"},
            "arg2": {"type": "string"},
        },
        "required": ["arg1", "arg2"],
    }

    result = function_to_json(basic_function)
    assert result == {
        "type": "function",
        "function": {
            "name": "basic_function",
            "description": "",
            "parameters": {
                "type": "object",
                "properties": {
                    "arg1": {"type": "string"},
                    "arg2": {"type": "string"},
                },
                "required": ["arg1", "arg2"],
            },
        },
    }


def test_complex_function():
    def complex_function_with_types_and_descriptions(
        arg1: int, arg2: str, arg3: float = 3.14, arg4: bool = False
    ):
        """This is a complex function with a docstring."""
        pass

    result = function_to_json(complex_function_with_types_and_descriptions)
    assert result == {
        "type": "function",
        "function": {
            "name": "complex_function_with_types_and_descriptions",
            "description": "This is a complex function with a docstring.",
            "parameters": {
                "type": "object",
                "properties": {
                    "arg1": {"type": "integer"},
                    "arg2": {"type": "string"},
                    "arg3": {"type": "number", "default": 3.14},
                    "arg4": {"type": "boolean", "default": False},
                },
                "required": ["arg1", "arg2"],
            },
        },
    }
