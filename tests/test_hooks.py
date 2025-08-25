"""Tests for Hook functionality."""

from typing import cast
from unittest.mock import AsyncMock, patch

import pytest
from openai.types.chat import ChatCompletionMessageParam

from swarmx import Agent, Hook
from swarmx.mcp_client import CLIENT_REGISTRY

pytestmark = pytest.mark.anyio


@pytest.fixture
def mock_client_registry():
    """Mock the CLIENT_REGISTRY for testing."""
    with patch.object(CLIENT_REGISTRY, "call_tool", new_callable=AsyncMock) as mock:
        yield mock


async def test_hook_creation():
    """Test that Hook can be created with all fields."""
    hook = Hook(
        on_start="start_tool",
        on_end="end_tool",
        on_tool_start="tool_start_tool",
        on_tool_end="tool_end_tool",
    )

    assert hook.on_start == "start_tool"
    assert hook.on_end == "end_tool"
    assert hook.on_tool_start == "tool_start_tool"
    assert hook.on_tool_end == "tool_end_tool"


async def test_hook_serialization():
    """Test that Hook can be serialized and deserialized."""
    hook = Hook(
        on_start="start_tool",
        on_end="end_tool",
    )

    # Test serialization
    hook_dict = hook.model_dump(exclude_none=True)
    expected = {
        "on_start": "start_tool",
        "on_end": "end_tool",
    }
    assert hook_dict == expected

    # Test deserialization
    hook_restored = Hook.model_validate(hook_dict)
    assert hook_restored.on_start == "start_tool"
    assert hook_restored.on_end == "end_tool"
    assert hook_restored.on_tool_start is None


async def test_agent_with_hooks():
    """Test that Agent can be created with hooks."""
    hook = Hook(on_start="start_tool")
    agent = Agent(hooks=[hook])

    assert len(agent.hooks) == 1
    assert agent.hooks[0].on_start == "start_tool"


async def test_execute_hooks_with_valid_tool(mock_client_registry):
    """Test that _execute_hooks calls the correct tool."""
    hook = Hook(on_start="test_tool")
    agent = Agent(hooks=[hook])

    messages = cast(
        list[ChatCompletionMessageParam], [{"role": "user", "content": "test"}]
    )
    result_messages, result_context = await agent._execute_hooks(
        "on_start", messages, {"test": "context"}
    )

    expected_input = {"messages": messages, "context": {"test": "context"}}
    mock_client_registry.assert_called_once_with("test_tool", expected_input)

    # Should return the same messages and context if no structured output
    assert result_messages == messages
    assert result_context == {"test": "context"}


async def test_execute_hooks_with_none_tool(mock_client_registry):
    """Test that _execute_hooks skips None tools."""
    hook = Hook(on_end="end_tool")
    agent = Agent(hooks=[hook])

    messages = cast(
        list[ChatCompletionMessageParam], [{"role": "user", "content": "test"}]
    )
    result_messages, result_context = await agent._execute_hooks(
        "on_start", messages, {"test": "context"}
    )

    mock_client_registry.assert_not_called()

    # Should return the same messages and context unchanged
    assert result_messages == messages
    assert result_context == {"test": "context"}


async def test_execute_hooks_with_multiple_hooks(mock_client_registry):
    """Test that _execute_hooks calls all hooks of the same type."""
    hook1 = Hook(on_start="tool1")
    hook2 = Hook(on_start="tool2")
    agent = Agent(hooks=[hook1, hook2])

    messages = cast(
        list[ChatCompletionMessageParam], [{"role": "user", "content": "test"}]
    )
    result_messages, result_context = await agent._execute_hooks(
        "on_start", messages, {"test": "context"}
    )

    assert mock_client_registry.call_count == 2
    expected_input = {"messages": messages, "context": {"test": "context"}}
    mock_client_registry.assert_any_call("tool1", expected_input)
    mock_client_registry.assert_any_call("tool2", expected_input)


async def test_execute_hooks_with_exception(mock_client_registry):
    """Test that _execute_hooks handles exceptions gracefully."""
    mock_client_registry.side_effect = Exception("Tool failed")

    hook = Hook(on_start="failing_tool")
    agent = Agent(hooks=[hook])

    messages = cast(
        list[ChatCompletionMessageParam], [{"role": "user", "content": "test"}]
    )

    # Should not raise an exception
    result_messages, result_context = await agent._execute_hooks(
        "on_start", messages, {"test": "context"}
    )

    expected_input = {"messages": messages, "context": {"test": "context"}}
    mock_client_registry.assert_called_once_with("failing_tool", expected_input)

    # Should return original values when exception occurs
    assert result_messages == messages
    assert result_context == {"test": "context"}


async def test_execute_hooks_with_basemodel_context(mock_client_registry):
    """Test that _execute_hooks properly handles BaseModel context."""
    from pydantic import BaseModel

    class TestContext(BaseModel):
        value: str

    hook = Hook(on_start="test_tool")
    agent = Agent(hooks=[hook])
    context = TestContext(value="test")

    messages = cast(
        list[ChatCompletionMessageParam], [{"role": "user", "content": "test"}]
    )
    result_messages, result_context = await agent._execute_hooks(
        "on_start", messages, context
    )

    expected_input = {"messages": messages, "context": {"value": "test"}}
    mock_client_registry.assert_called_once_with("test_tool", expected_input)


async def test_execute_hooks_with_structured_output(mock_client_registry):
    """Test that _execute_hooks handles structured output correctly."""
    from unittest.mock import MagicMock

    # Mock the tool result with structured output
    mock_result = MagicMock()
    mock_result.structuredContent = {
        "messages": [{"role": "assistant", "content": "modified response"}],
        "context": {"modified": True, "value": "updated"},
    }
    mock_client_registry.return_value = mock_result

    hook = Hook(on_start="modifying_tool")
    agent = Agent(hooks=[hook])

    messages = cast(
        list[ChatCompletionMessageParam], [{"role": "user", "content": "test"}]
    )
    original_context = {"original": True}

    result_messages, result_context = await agent._execute_hooks(
        "on_start", messages, original_context
    )

    # Verify the tool was called with correct input
    expected_input = {"messages": messages, "context": original_context}
    mock_client_registry.assert_called_once_with("modifying_tool", expected_input)

    # Verify the output was modified
    assert result_messages == [{"role": "assistant", "content": "modified response"}]
    assert result_context == {"modified": True, "value": "updated"}


async def test_hook_integration_with_agent_run(mock_client_registry, model):
    """Test that hooks are executed during agent run."""
    hook = Hook(on_start="start_tool", on_end="end_tool")
    agent = Agent(hooks=[hook], model=model)

    messages = cast(
        list[ChatCompletionMessageParam], [{"role": "user", "content": "test"}]
    )
    # Consume the async generator
    async for _ in agent._run(messages=messages):
        pass

    # Verify hooks were called with the new signature
    assert mock_client_registry.call_count >= 2

    # Check that start_tool was called with initial messages
    start_call_found = False
    end_call_found = False

    for call in mock_client_registry.call_args_list:
        args, _ = call
        tool_name, tool_input = args

        if tool_name == "start_tool" and tool_input["messages"] == messages:
            start_call_found = True
        elif tool_name == "end_tool" and "messages" in tool_input:
            end_call_found = True

    assert start_call_found, "start_tool hook was not called correctly"
    assert end_call_found, "end_tool hook was not called correctly"


async def test_hook_modifies_context_and_messages(mock_client_registry):
    """Test that hooks can modify both context and messages through structured output."""
    from unittest.mock import MagicMock

    # Mock the tool result with structured output that modifies both messages and context
    mock_result = MagicMock()
    mock_result.structuredContent = {
        "messages": [
            {"role": "user", "content": "test"},
            {"role": "system", "content": "Added by hook"},
        ],
        "context": {"hook_executed": True, "original_count": 1},
    }
    mock_client_registry.return_value = mock_result

    hook = Hook(on_start="context_modifier")
    agent = Agent(hooks=[hook])

    messages = cast(
        list[ChatCompletionMessageParam], [{"role": "user", "content": "test"}]
    )
    original_context = {"original": True}

    result_messages, result_context = await agent._execute_hooks(
        "on_start", messages, original_context
    )

    # Verify the tool was called with correct input
    expected_input = {"messages": messages, "context": original_context}
    mock_client_registry.assert_called_once_with("context_modifier", expected_input)

    # Verify both messages and context were modified
    assert len(result_messages) == 2
    assert result_messages[0] == {"role": "user", "content": "test"}
    assert result_messages[1] == {"role": "system", "content": "Added by hook"}

    assert result_context == {"hook_executed": True, "original_count": 1}


def test_hook_json_schema():
    """Test that Hook can be included in JSON schema generation."""
    hook = Hook(on_start="test_tool")
    schema = hook.model_json_schema()

    assert "properties" in schema
    assert "on_start" in schema["properties"]
    assert "on_end" in schema["properties"]
    assert "on_tool_start" in schema["properties"]
    assert "on_tool_end" in schema["properties"]


async def test_execute_hooks_with_basemodel_context_modification(mock_client_registry):
    """Test that _execute_hooks modifies BaseModel context attributes (covers lines 118-119)."""
    from unittest.mock import MagicMock

    from pydantic import BaseModel

    class TestContext(BaseModel):
        user_id: str
        session_id: str
        modified: bool = False

    # Mock the tool result with structured output that modifies context
    mock_result = MagicMock()
    mock_result.structuredContent = {
        "context": {
            "user_id": "updated_user",
            "modified": True,
            # Note: session_id is not included, so it should remain unchanged
        }
    }
    mock_client_registry.return_value = mock_result

    hook = Hook(on_start="context_modifier")
    agent = Agent(hooks=[hook])

    messages = cast(
        list[ChatCompletionMessageParam], [{"role": "user", "content": "test"}]
    )
    original_context = TestContext(user_id="original_user", session_id="session_123")

    result_messages, result_context = await agent._execute_hooks(
        "on_start", messages, original_context
    )

    # Verify the tool was called with correct input
    expected_input = {
        "messages": messages,
        "context": {
            "user_id": "original_user",
            "session_id": "session_123",
            "modified": False,
        },
    }
    mock_client_registry.assert_called_once_with("context_modifier", expected_input)

    # Verify the BaseModel context was modified correctly
    assert isinstance(result_context, TestContext)
    assert result_context.user_id == "updated_user"  # Modified by hook
    assert result_context.session_id == "session_123"  # Unchanged
    assert result_context.modified is True  # Modified by hook


def test_typeddict_import_python312():
    """Test TypedDict import for Python 3.12+ (covers line 14)."""
    import sys
    from unittest.mock import patch

    # Mock sys.version_info to simulate Python 3.12+
    with patch.object(sys, "version_info", (3, 12, 0)):
        # Force reimport of the hook module to trigger the Python 3.12+ import path
        import importlib

        import swarmx.hook

        importlib.reload(swarmx.hook)

        # Verify that the module was imported successfully
        # The fact that no ImportError was raised means line 14 was executed
        assert hasattr(swarmx.hook, "TypedDict")

    # Restore the original module state
    importlib.reload(swarmx.hook)
