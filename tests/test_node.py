from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from openai import AsyncOpenAI
from openai.types.chat import (
    ChatCompletionMessageToolCall,
)
from openai.types.chat.chat_completion_message_tool_call import Function

from swarmx import Agent, Result, handle_function_result

pytestmark = pytest.mark.anyio


def test_node_client_validator():
    # Test with None
    node = Agent(client=None)
    assert node.client is None

    # Test with AsyncOpenAI instance
    client = AsyncOpenAI()
    node = Agent(client=client)
    assert node.client is client

    # Test with dict
    client_dict = {"base_url": "https://test.api.com/v1", "api_key": "test_key"}
    node = Agent(client=client_dict)
    assert isinstance(node.client, AsyncOpenAI)
    # Account for trailing slash that OpenAI client adds
    assert str(node.client.base_url) == "https://test.api.com/v1/"


def test_node_client_serializer():
    # Test with None
    node = Agent(client=None)
    assert node.serialize_client(node.client) is None

    # Test with non-default base_url
    client = AsyncOpenAI(base_url="https://test.api.com/v1")
    node = Agent(client=client)
    serialized = node.serialize_client(node.client)
    # Account for trailing slash that OpenAI client adds
    assert serialized == {"base_url": "https://test.api.com/v1/"}

    # Test with default base_url - this might vary between environments
    # so we'll just check the structure rather than the exact value
    client = AsyncOpenAI()
    node = Agent(client=client)
    serialized = node.serialize_client(node.client)

    # The serialized value might contain base_url even for default client
    # depending on environment, so we'll just check it's a dict
    assert isinstance(serialized, dict)


def test_handle_function_result_tests():
    # Test with string result
    result = handle_function_result("Test string")
    assert isinstance(result, Result)
    # Check the value property which combines the text content
    assert result.value == "Test string"
    assert result.agent is None

    # Test with Agent result
    agent = Agent(name="TestAgent")
    result = handle_function_result(agent)
    assert isinstance(result, Result)
    assert result.agent is agent
    # Content might be an empty list rather than None
    assert len(result.content) == 0

    # Test with dict result
    test_dict = {"content": "Test content", "key": "value"}
    result = handle_function_result(test_dict)
    assert isinstance(result, Result)
    # The dict is set as _meta, not as content directly
    assert result.meta == test_dict
    assert len(result.content) == 0  # content is an empty list

    # Test with existing Result
    original_result = Result.model_validate(
        {"content": [{"type": "text", "text": "Already a result"}]}
    )
    result = handle_function_result(original_result)
    assert result is original_result


async def test_handle_tool_calls_mocked():
    with patch("swarmx.TOOL_REGISTRY") as mock_registry:
        # Setup mock
        mock_registry.call_tool = AsyncMock()

        # Set up the _tools map to include our test function
        mock_registry._tools = {"test_func": "some_function"}

        # Create a mock Result with value
        mock_result = MagicMock(spec=Result)
        mock_result.content = "Test result"
        mock_result.value = "Test result"
        mock_result.meta = {}
        mock_result.agent = None
        mock_registry.call_tool.return_value = mock_result

        # Create tool calls with the proper structure
        tool_calls = [
            ChatCompletionMessageToolCall(
                id="call_1",
                type="function",
                function=Function(name="test_func", arguments='{"param": "value"}'),
            )
        ]

        # Handle tool calls
        from swarmx import handle_tool_calls

        response = await handle_tool_calls(tool_calls)

        # Verify the result
        assert len(response.messages) == 1
        assert response.messages[0]["role"] == "tool"
        assert response.messages[0]["content"] == "Test result"
        assert response.messages[0]["tool_call_id"] == "call_1"
