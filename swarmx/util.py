from typing import TYPE_CHECKING, Any

from loguru import logger
from pydantic.json_schema import GenerateJsonSchema

from .types import ChatCompletionMessageToolCall

if TYPE_CHECKING:
    from .types import Tool

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


def handle_tool_calls(
    tool_calls: list[ChatCompletionMessageToolCall],
    tools: "list[Tool]",
    context_variables: dict[str, Any],
):
    from .types import Response

    tool_map = {f.__name__: f for f in tools}
    partial_response = Response(messages=[], agent=None, context_variables={})

    for tool_call in tool_calls:
        name = tool_call.function.name
        # handle missing tool case, skip to next tool
        if name not in tool_map:
            logger.debug(f"Tool {name} not found in function map.")
            partial_response.messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": f"Error: Tool {name} not found.",
                }
            )
            continue

        tool = tool_map[name]
        args = tool.arguments_model.model_validate_json(
            tool_call.function.arguments
        ).model_dump(mode="json")
        logger.debug(f"Processing tool call: {name} with arguments {args}")
        # pass context_variables to agent functions
        if __CTX_VARS_NAME__ in tool.function.__code__.co_varnames:
            args[__CTX_VARS_NAME__] = context_variables
        result = tool(**args)
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
