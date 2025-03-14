import inspect

import pytest

from swarmx import (
    __CTX_VARS_NAME__,
    Agent,
    check_instructions,
    validate_tool,
    validate_tools,
)

pytestmark = pytest.mark.anyio


async def test_agent_creation():
    agent = Agent(
        name="test_agent",
        model="deepseek-r1",
        instructions="You are a fantasy writer.",
    )
    assert agent.name == "test_agent"
    assert agent.model == "deepseek-r1"
    assert agent.instructions == "You are a fantasy writer."


async def test_agent_with_default_values():
    agent = Agent()
    assert agent.name == "Agent"
    assert agent.model == "deepseek-reasoner"
    assert agent.instructions == "You are a helpful agent."
    assert agent.tools == []
    assert agent.client is None


async def test_agent_with_custom_client():
    client_config = {"api_key": "test_key", "organization": "test_org"}
    agent = Agent(client=client_config)
    assert agent.client is not None
    assert agent.client.api_key == "test_key"
    assert agent.client.organization == "test_org"

    # Test serialization
    serialized = agent.model_dump(mode="json")
    assert "client" in serialized
    assert serialized["client"]["organization"] == "test_org"
    assert "api_key" not in serialized["client"]  # Should not serialize the API key


async def test_agent_with_callable_instructions():
    def get_instructions(ctx):
        return f"You are a helpful assistant with context: {ctx.get('user_info', '')}"

    agent = Agent(instructions=get_instructions)
    assert callable(agent.instructions)

    # Test _with_instructions method
    messages = [{"role": "user", "content": "Hello"}]
    ctx_vars = {"user_info": "John Doe"}
    result = agent._with_instructions(messages=messages, context_variables=ctx_vars)

    assert len(result) == 2
    assert result[0]["role"] == "system"
    assert "John Doe" in result[0]["content"]
    assert result[1] == messages[0]


async def test_agent_with_jinja_template_instructions():
    agent = Agent(instructions="You are a helpful assistant for {{ user_name }}.")

    messages = [{"role": "user", "content": "Hello"}]
    ctx_vars = {"user_name": "Alice"}
    result = agent._with_instructions(messages=messages, context_variables=ctx_vars)

    assert len(result) == 2
    assert result[0]["role"] == "system"
    assert "Alice" in result[0]["content"]


async def test_agent_with_tools():
    def tool_function(x: int, y: int) -> str:
        """Add two numbers.

        Args:
            x: First number
            y: Second number

        Returns:
            The sum as a string
        """
        return str(x + y)

    tool_dict = {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the weather for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The location to get weather for",
                    }
                },
                "required": ["location"],
            },
        },
    }

    agent = Agent(name="tool_agent", tools=[tool_function, tool_dict])

    assert len(agent.tools) == 2
    assert agent.tools[0]["function"]["name"] == "tool_function"
    assert agent.tools[1]["function"]["name"] == "get_weather"


def test_check_instructions_with_string():
    """Test check_instructions function with a string input"""
    instructions = "You are a helpful assistant."
    result = check_instructions(instructions)
    assert result == instructions


def test_check_instructions_with_valid_callable_ctx():
    """Test check_instructions function with a valid callable that accepts __CTX_VARS_NAME__"""
    # Using a dynamic function name to match __CTX_VARS_NAME__
    func_code = f"def instructions_func({__CTX_VARS_NAME__}):\n    return f'You are a helpful assistant with context: {{{__CTX_VARS_NAME__}}}'"
    namespace = {}
    exec(func_code, namespace)
    instructions_func = namespace["instructions_func"]

    result = check_instructions(instructions_func)
    assert callable(result)


def test_check_instructions_with_valid_callable_params():
    """Test check_instructions function with a valid callable that has multiple parameters"""

    def instructions_func(name, age):
        return f"You are a helpful assistant for {name} who is {age} years old."

    result = check_instructions(instructions_func)
    assert callable(result)


def test_check_instructions_with_invalid_callable():
    """Test check_instructions function with an invalid callable"""
    # Create a function that has both context and __CTX_VARS_NAME__ parameters
    func_code = f"""\
def invalid_func(context, {__CTX_VARS_NAME__}):
    return f'Invalid function with both context and {{{__CTX_VARS_NAME__}}}'
"""
    namespace = {}
    exec(func_code, namespace)
    invalid_func = namespace["invalid_func"]

    # Ensure the signature has both parameters
    sig = inspect.signature(invalid_func)
    assert "context" in sig.parameters
    assert __CTX_VARS_NAME__ in sig.parameters

    with pytest.raises(ValueError):
        check_instructions(invalid_func)


def test_validate_tool_with_dict():
    """Test validate_tool function with a dictionary input"""
    tool_dict = {
        "type": "function",
        "function": {
            "name": "test_tool",
            "description": "A test tool",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    }

    result = validate_tool(tool_dict)
    assert result["type"] == "function"
    assert result["function"]["name"] == "test_tool"


def test_validate_tool_with_function():
    """Test validate_tool function with a callable input"""

    def test_func(x: int) -> str:
        """A test function"""
        return str(x)

    result = validate_tool(test_func)
    assert result["type"] == "function"
    assert result["function"]["name"] == "test_func"


def test_validate_tools():
    """Test validate_tools function"""

    def test_func(x: int) -> str:
        """A test function"""
        return str(x)

    tool_dict = {
        "type": "function",
        "function": {
            "name": "test_tool",
            "description": "A test tool",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    }

    result = validate_tools([test_func, tool_dict])
    assert len(result) == 2
    assert result[0]["function"]["name"] == "test_func"
    assert result[1]["function"]["name"] == "test_tool"


async def test_agent_run_with_max_turns_error():
    agent = Agent()

    with pytest.raises(RuntimeError, match="Reached max turns"):
        await agent.run(messages=[{"role": "user", "content": "Hello"}], max_turns=0)
