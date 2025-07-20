"""MCP client related."""

import inspect
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from typing import Any, Callable, cast

from mcp import ClientSession, StdioServerParameters
from mcp.client.sse import sse_client
from mcp.client.stdio import stdio_client
from openai.types.chat.chat_completion_tool_param import ChatCompletionToolParam
from pydantic import create_model
from pydantic.json_schema import GenerateJsonSchema

__CTX_VARS_NAME__ = "context_variables"


class SwarmXGenerateJsonSchema(GenerateJsonSchema):
    """Remove the title field from the JSON schema."""

    def field_title_should_be_set(self, schema) -> bool:
        """No title for all fields."""
        return False


def function_to_json(func: Any) -> ChatCompletionToolParam:
    """Convert a function to a JSON schema."""
    if not callable(func):
        raise ValueError("Function is not callable")
    signature = inspect.signature(func)
    field_definitions = {}
    for param in signature.parameters.values():
        if param.name == __CTX_VARS_NAME__:
            continue
        field_definitions[param.name] = (
            param.annotation if param.annotation is not param.empty else str,
            param.default if param.default is not param.empty else ...,
        )
    arguments_model = create_model(func.__name__, **field_definitions)  # type: ignore[call-overload]
    function: ChatCompletionToolParam = {
        "type": "function",
        "function": {
            "name": func.__name__,
            "parameters": {
                k: v
                for k, v in arguments_model.model_json_schema(
                    schema_generator=SwarmXGenerateJsonSchema
                ).items()
                if k != "title"
            },
        },
    }
    if func.__doc__:
        function["function"]["description"] = func.__doc__
    return function


# SECTION 4: Tool registry
@dataclass
class ToolRegistry:
    """Registry for tools."""

    functions: dict[str, ChatCompletionToolParam] = field(default_factory=dict)
    mcp_tools: dict[tuple[str, str], ChatCompletionToolParam] = field(
        default_factory=dict
    )
    exit_stack: AsyncExitStack = field(default_factory=AsyncExitStack)
    mcp_clients: dict[str, ClientSession] = field(default_factory=dict)
    _tools: dict[str, "Callable[..., Any] | str"] = field(default_factory=dict)

    @property
    def tools(self) -> dict[str, ChatCompletionToolParam]:
        """Return all tools, both local and MCP."""
        return {
            **self.functions,
            **{tool_name: tool for (_, tool_name), tool in self.mcp_tools.items()},
        }

    async def call_tool(
        self,
        name: str,
        arguments: dict[str, Any],
        context_variables: dict[str, Any] | None = None,
    ):
        """Call a tool.

        Args:
            name: The name of the tool
            arguments: The arguments to pass to the tool
            context_variables: The context variables to pass to the tool

        """
        callable_func = self._tools.get(name)
        if callable_func is None:
            raise ValueError(f"Tool {name} not found")
        if isinstance(callable_func, str):
            return await self.mcp_clients[callable_func].call_tool(name, arguments)
        signature = inspect.signature(callable_func)
        if __CTX_VARS_NAME__ in signature.parameters:
            arguments[__CTX_VARS_NAME__] = context_variables or {}
        result = callable_func(**arguments)
        if inspect.isawaitable(result):
            result = await result
        return result

    async def add_mcp_server(
        self, name: str, server_params: StdioServerParameters | str
    ):
        """Add an MCP server to the registry.

        Args:
            name: The name of the server
            server_params: The parameters to connect to the server

        """
        if name in self.mcp_clients:
            return
        read_stream, write_stream = await self.exit_stack.enter_async_context(
            sse_client(server_params)
            if isinstance(server_params, str)
            else stdio_client(server_params)
        )
        client = cast(
            ClientSession,
            await self.exit_stack.enter_async_context(
                ClientSession(read_stream, write_stream)
            ),
        )
        await client.initialize()
        self.mcp_clients[name] = client
        for tool in (await client.list_tools()).tools:
            self._tools[tool.name] = name
            function_schema = tool.model_dump(exclude_none=True)
            function_schema["parameters"] = function_schema.pop("inputSchema")
            self.mcp_tools[name, tool.name] = ChatCompletionToolParam(
                type="function",
                function=function_schema,  # type: ignore
            )

    def add_function(self, func: Callable[..., Any]):
        """Add a function to the registry.

        Args:
            func: The function to add

        """
        func_json = function_to_json(func)
        name = func_json["function"]["name"]
        self.functions[name] = func_json
        self._tools[name] = func

    async def close(self):
        """Close all clients."""
        await self.exit_stack.aclose()


TOOL_REGISTRY = ToolRegistry()
