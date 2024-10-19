import inspect
import json
from typing import Any

from loguru import logger
from pydantic import BaseModel, create_model
from pydantic.json_schema import GenerateJsonSchema

from .types import (
    Agent,
    AgentFunction,
    AgentFunctionReturnType,
    ChatCompletionMessageToolCall,
    ChatCompletionToolParam,
    Response,
    Result,
)

__CTX_VARS_NAME__ = "context_variables"


class SwarmXGenerateJsonSchema(GenerateJsonSchema):
    """hide context_variables from model"""

    def generate(self, schema, mode="validation"):
        json_schema = super().generate(schema, mode=mode)
        properties = {
            name: {k: v for k, v in property.items() if k != "title"}
            for name, property in json_schema.pop("properties", {}).items()
            if name != __CTX_VARS_NAME__
        }
        required = [
            r for r in json_schema.pop("required", []) if r != __CTX_VARS_NAME__
        ]
        return {
            **({k: v for k, v in json_schema.items() if k != "title"}),
            "properties": properties,
            **({"required": required} if required else {}),
        }


def function_to_model(func: AgentFunction) -> BaseModel:
    """
    Converts a Python function into a Pydantic model that describes
    the function's signature, including its name, description, and parameters.

    Args:
        func: The function to be converted.

    Returns:
        A Pydantic model representing the function's signature.
    """

    try:
        signature = inspect.signature(func)
    except ValueError as e:
        raise ValueError(
            f"Failed to get signature for function {func.__name__}: {str(e)}"
        )

    parameters = {}
    for param in signature.parameters.values():
        parameters[param.name] = (
            param.annotation if param.annotation is not param.empty else str,
            param.default if param.default is not param.empty else ...,
        )
    return create_model(func.__name__, **parameters)  # type: ignore[call-overload]


def function_to_json(func: AgentFunction) -> ChatCompletionToolParam:
    """
    Converts a Python function into a JSON-serializable dictionary
    that describes the function's signature, including its name,
    description, and parameters.

    Args:
        func: The function to be converted.

    Returns:
        A dictionary representing the function's signature in JSON format.
    """
    return {
        "type": "function",
        "function": {
            "name": func.__name__,
            "description": func.__doc__ or "",
            "parameters": function_to_model(func).model_json_schema(
                schema_generator=SwarmXGenerateJsonSchema
            ),
        },
    }


def handle_function_result(result: AgentFunctionReturnType) -> Result:
    match result:
        case Result() as result:
            return result

        case Agent() as agent:
            return Result(
                value=json.dumps({"assistant": agent.name}),
                agent=agent,
            )
        case _:
            try:
                return Result(value=str(result))
            except Exception as e:
                error_message = f"Failed to cast response to string: {result}. Make sure agent functions return a string or Result object. Error: {str(e)}"
                logger.debug(error_message)
                raise TypeError(error_message)


def handle_tool_calls(
    tool_calls: list[ChatCompletionMessageToolCall],
    functions: list[AgentFunction],
    context_variables: dict[str, Any],
) -> Response:
    function_map = {f.__name__: f for f in functions}
    partial_response = Response(messages=[], agent=None, context_variables={})

    for tool_call in tool_calls:
        name = tool_call.function.name
        # handle missing tool case, skip to next tool
        if name not in function_map:
            logger.debug(f"Tool {name} not found in function map.")
            partial_response.messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": f"Error: Tool {name} not found.",
                }
            )
            continue

        func = function_map[name]
        args = (
            function_to_model(func)
            .model_validate_json(tool_call.function.arguments)
            .model_dump(mode="json")
        )
        logger.debug(f"Processing tool call: {name} with arguments {args}")
        # pass context_variables to agent functions
        if __CTX_VARS_NAME__ in func.__code__.co_varnames:
            args[__CTX_VARS_NAME__] = context_variables
        raw_result = func(**args)

        result: Result = handle_function_result(raw_result)
        partial_response.messages.append(
            {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result.value,
            }
        )
        partial_response.context_variables.update(result.context_variables)
        if result.agent:
            partial_response.agent = result.agent

    return partial_response
