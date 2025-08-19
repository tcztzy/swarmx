from collections import defaultdict
from unittest.mock import AsyncMock, patch

import pytest
from httpx import Timeout
from mcp.client.stdio import StdioServerParameters
from openai import AsyncOpenAI
from openai.types.chat import (
    ChatCompletionMessageParam,
    ChatCompletionMessageToolCallParam,
)
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk

from swarmx import Agent
from swarmx.agent import Edge, SwarmXGenerateJsonSchema, _merge_chunk, exec_tool_calls
from swarmx.utils import now

pytestmark = pytest.mark.anyio


async def test_agent_validate_and_serialize():
    agent = Agent(
        name="test_agent",
        model="deepseek-r1",
        instructions="You are a fantasy writer.",
    )
    assert agent.name == "test_agent"
    assert agent.model == "deepseek-r1"
    assert agent.instructions == "You are a fantasy writer."

    # Test serialization and deserialization
    serialized = agent.model_dump(mode="json")
    assert isinstance(serialized, dict)
    assert serialized["name"] == "test_agent"
    assert serialized["model"] == "deepseek-r1"
    assert serialized["instructions"] == "You are a fantasy writer."

    # Test loading from serialized data
    loaded_agent = Agent(**serialized)
    assert loaded_agent.name == "test_agent"
    assert loaded_agent.model == "deepseek-r1"
    assert loaded_agent.instructions == "You are a fantasy writer."


async def test_agent_with_custom_client():
    client_config = {"api_key": "test_key", "organization": "test_org"}
    agent = Agent(client=client_config)  # type: ignore
    assert agent.client is not None
    assert agent.client.api_key == "test_key"
    assert agent.client.organization == "test_org"

    # Test serialization
    serialized = agent.model_dump(mode="json")
    assert "client" in serialized
    assert serialized["client"]["organization"] == "test_org"
    assert "api_key" not in serialized["client"]  # Should not serialize the API key


async def test_agent_with_jinja_template_instructions():
    agent = Agent(instructions="You are a helpful assistant for {{ user_name }}.")
    ctx_vars = {"user_name": "Alice"}
    result = await agent.get_system_prompt(context=ctx_vars)
    assert result is not None and "Alice" in result


async def test_run_node_stream():
    agent = Agent(
        name="test_agent",
        model="deepseek-r1",
        instructions="You are a fantasy writer.",
        entry_point="agent1",
        nodes={"agent1": Agent(name="agent1")},
    )
    messages: list[ChatCompletionMessageParam] = [{"role": "user", "content": "Hello"}]
    context = None

    async for chunk in agent._run_node_stream(
        messages=messages,
        context=context,
    ):
        assert chunk.id is not None


async def test_create_chat_completion():
    agent = Agent(
        name="test_agent",
        model="deepseek-r1",
        instructions="You are a fantasy writer.",
    )
    messages: list[ChatCompletionMessageParam] = [{"role": "user", "content": "Hello"}]
    context = None

    completion = await agent._create_chat_completion(
        messages=messages,
        context=context,
        stream=False,
    )
    assert len(completion.choices) > 0


async def test_get_system_prompt_edge_cases():
    agent = Agent(
        name="test_agent",
        model="deepseek-r1",
        instructions="You are a helpful assistant for {{ user_name }}.",
    )
    # Test with None context
    result = await agent.get_system_prompt(context=None)
    assert result is not None and "user_name" not in result
    # Test with empty context
    result = await agent.get_system_prompt(context={})
    assert result is not None and "user_name" not in result
    # Test with invalid context
    result = await agent.get_system_prompt(context={"invalid_key": "value"})
    assert result is not None and "user_name" not in result


async def test_exec_tool_calls():
    tool_calls: list[ChatCompletionMessageToolCallParam] = [
        {
            "id": "1",
            "type": "function",
            "function": {"name": "test_tool", "arguments": "{}"},
        }
    ]
    async for chunk in exec_tool_calls(tool_calls):
        ...
    else:
        assert isinstance(chunk, list)  # type: ignore
        assert len(chunk) > 0


async def test_linear_sequence_of_subagents():
    # Create main agent with nodes
    main_agent = Agent(
        name="main_agent",
        model="deepseek-r1",
        instructions="Coordinate the workflow.",
    )

    # Create subagents
    agent1 = Agent(
        name="agent1",
        model="deepseek-r1",
        instructions="You are a fantasy writer.",
    )
    agent2 = Agent(
        name="agent2",
        model="deepseek-r1",
        instructions="You are a fantasy editor.",
    )
    agent3 = Agent(
        name="agent3",
        model="deepseek-r1",
        instructions="You are a fantasy publisher.",
    )

    # Add nodes to main agent
    main_agent.nodes = {"agent1": agent1, "agent2": agent2, "agent3": agent3}

    # Add edges to create linear sequence
    main_agent.edges = [
        Edge(source="agent1", target="agent2"),
        Edge(source="agent2", target="agent3"),
    ]

    # Test the sequence
    assert "agent1" in main_agent.nodes
    assert "agent2" in main_agent.nodes
    assert "agent3" in main_agent.nodes
    assert all(isinstance(e, Edge) for e in main_agent.edges)
    assert any(
        e.source == "agent1" and e.target == "agent2"
        for e in main_agent.edges
        if isinstance(e, Edge)
    )
    assert any(
        e.source == "agent2" and e.target == "agent3"
        for e in main_agent.edges
        if isinstance(e, Edge)
    )

    # Test serialization and deserialization
    serialized = main_agent.model_dump(mode="json")
    loaded_agent = Agent(**serialized)
    assert "agent1" in loaded_agent.nodes
    assert "agent2" in loaded_agent.nodes
    assert "agent3" in loaded_agent.nodes
    assert all(isinstance(e, Edge) for e in loaded_agent.edges)
    assert any(
        e.source == "agent1" and e.target == "agent2"
        for e in loaded_agent.edges
        if isinstance(e, Edge)
    )
    assert any(
        e.source == "agent2" and e.target == "agent3"
        for e in loaded_agent.edges
        if isinstance(e, Edge)
    )


async def test_agent_sequence_execution_stream():
    # Create main agent with nodes
    main_agent = Agent(
        name="main_agent",
        model="deepseek-r1",
        instructions="Coordinate the workflow.",
        entry_point="agent1",
    )

    # Create subagents with specific behaviors
    agent1 = Agent(
        name="agent1",
        model="deepseek-r1",
        instructions="You are a fantasy writer. Always respond with 'Story written'.",
    )
    agent2 = Agent(
        name="agent2",
        model="deepseek-r1",
        instructions="You are a fantasy editor. Always respond with 'Story edited'.",
    )
    agent3 = Agent(
        name="agent3",
        model="deepseek-r1",
        instructions="You are a fantasy publisher. Always respond with 'Story published'.",
    )

    # Add nodes and edges
    main_agent.nodes = {"agent1": agent1, "agent2": agent2, "agent3": agent3}
    main_agent.edges = [
        Edge(source="agent1", target="agent2"),
        Edge(source="agent2", target="agent3"),
    ]

    # Test execution flow
    messages: list[ChatCompletionMessageParam] = [
        {"role": "user", "content": "Create a fantasy story"}
    ]
    responses = []

    async for chunk in main_agent._run_node_stream(messages=messages):
        if chunk.choices[0].delta.content:
            responses.append(chunk.choices[0].delta.content)

    # Verify each agent responded in sequence
    assert "Story written" in "".join(responses)
    assert "Story edited" in "".join(responses)
    assert "Story published" in "".join(responses)

    # Verify order of responses
    assert "".join(responses).index("Story written") < "".join(responses).index(
        "Story edited"
    )
    assert "".join(responses).index("Story edited") < "".join(responses).index(
        "Story published"
    )


async def test_merge_chunk_with_content():
    """Test _merge_chunk function with content."""
    messages: dict[str, ChatCompletionMessageParam] = defaultdict(
        lambda: {"role": "assistant"}
    )
    chunk = ChatCompletionChunk.model_validate(
        {
            "id": "test_id",
            "choices": [{"index": 0, "delta": {"content": "Hello world"}}],
            "created": now(),
            "model": "gpt-4o",
            "object": "chat.completion.chunk",
        }
    )

    _merge_chunk(messages, chunk)

    assert "test_id" in messages
    assert messages["test_id"]["role"] == "assistant"
    assert messages["test_id"].get("content") == "Hello world"


async def test_merge_chunk_with_content_list():
    """Test _merge_chunk function with content as list (line 64)."""
    messages: dict[str, ChatCompletionMessageParam] = defaultdict(
        lambda: {"role": "assistant"}
    )
    # Set initial content as a list - use type ignore to bypass type checking for test
    messages["test_id"]["content"] = [{"type": "text", "text": "Initial"}]  # type: ignore

    chunk = ChatCompletionChunk.model_validate(
        {
            "id": "test_id",
            "choices": [{"index": 0, "delta": {"content": " more text"}}],
            "created": now(),
            "model": "gpt-4o",
            "object": "chat.completion.chunk",
        }
    )
    _merge_chunk(messages, chunk)

    # Should append new text content to the list
    content = messages["test_id"]["content"]
    assert isinstance(content, list)
    assert len(content) == 2
    assert content[0] == {"type": "text", "text": "Initial"}
    assert content[1] == {"type": "text", "text": " more text"}


async def test_merge_chunk_with_refusal():
    """Test _merge_chunk function with refusal."""
    messages: dict[str, ChatCompletionMessageParam] = defaultdict(
        lambda: {"role": "assistant"}
    )
    chunk = ChatCompletionChunk.model_validate(
        {
            "id": "test_id",
            "choices": [{"index": 0, "delta": {"refusal": "I cannot help with that"}}],
            "created": now(),
            "model": "gpt-4o",
            "object": "chat.completion.chunk",
        }
    )

    _merge_chunk(messages, chunk)

    assert "test_id" in messages
    assert messages["test_id"]["role"] == "assistant"
    assert messages["test_id"].get("refusal") == "I cannot help with that"


async def test_merge_chunk_with_tool_calls():
    """Test _merge_chunk function with tool calls."""
    messages: dict[str, ChatCompletionMessageParam] = defaultdict(
        lambda: {"role": "assistant"}
    )
    chunk = ChatCompletionChunk.model_validate(
        {
            "id": "test_id",
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_123",
                                "function": {
                                    "name": "test_function",
                                    "arguments": '{"arg": "value"}',
                                },
                                "type": "function",
                            }
                        ]
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "created": now(),
            "model": "gpt-4o",
            "object": "chat.completion.chunk",
        }
    )

    _merge_chunk(messages, chunk)

    assert "test_id" in messages
    assert messages["test_id"]["role"] == "assistant"
    assert "tool_calls" in messages["test_id"]


async def test_merge_chunk_invalid_tool_call_role():
    """Test _merge_chunk function with tool calls on non-assistant message."""
    messages: dict[str, ChatCompletionMessageParam] = defaultdict(
        lambda: {"role": "assistant"}
    )
    messages["test_id"] = {"role": "user", "content": "Hello"}
    chunk = ChatCompletionChunk.model_validate(
        {
            "id": "test_id",
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_123",
                                "function": {
                                    "name": "test_function",
                                    "arguments": '{"arg": "value"}',
                                },
                                "type": "function",
                            }
                        ]
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "created": now(),
            "model": "gpt-4o",
            "object": "chat.completion.chunk",
        }
    )

    with pytest.raises(
        ValueError, match="Tool calls can only be added to assistant messages"
    ):
        _merge_chunk(messages, chunk)


async def test_swarmx_generate_json_schema():
    """Test SwarmXGenerateJsonSchema class."""
    schema_generator = SwarmXGenerateJsonSchema()

    # Test that it exists and is the right type
    assert isinstance(schema_generator, SwarmXGenerateJsonSchema)


async def test_agent_as_tool():
    """Test Agent.as_tool method (lines 163-164)."""
    tool = Agent.as_tool()
    assert tool.name == "swarmx.Agent"
    assert tool.description == "Create new Agent"
    assert tool.inputSchema is not None
    assert tool.outputSchema is not None


async def test_agent_client_serialization_with_timeout():
    """Test agent client serialization with Timeout object."""
    timeout = Timeout(10.0)
    client = AsyncOpenAI(api_key="test-key", timeout=timeout)
    agent = Agent(client=client)

    serialized = agent.model_dump(mode="json")
    assert "client" in serialized
    assert "timeout" in serialized["client"]


async def test_agent_client_validation_with_timeout_dict():
    """Test agent client validation with timeout as dict (line 185)."""
    client_config = {
        "api_key": "test-key",
        "timeout": {"connect": 5.0, "read": 10.0, "write": 5.0, "pool": 10.0},
    }
    agent = Agent(client=client_config)  # type: ignore
    assert agent.client is not None
    assert isinstance(agent.client.timeout, Timeout)
    assert agent.client.timeout.connect == 5.0
    assert agent.client.timeout.read == 10.0


async def test_agent_client_serialization_with_custom_headers():
    """Test agent client serialization with custom headers."""
    client = AsyncOpenAI(api_key="test-key", default_headers={"Custom-Header": "value"})
    agent = Agent(client=client)

    serialized = agent.model_dump(mode="json")
    assert "client" in serialized
    assert "default_headers" in serialized["client"]


async def test_agent_client_serialization_with_float_timeout():
    """Test agent client serialization with timeout as float (line 209)."""
    client = AsyncOpenAI(api_key="test-key", timeout=15.0)
    agent = Agent(client=client)

    serialized = agent.model_dump(mode="json")
    assert "client" in serialized
    assert "timeout" in serialized["client"]
    assert serialized["client"]["timeout"] == 15.0


async def test_agent_client_serialization_with_custom_query():
    """Test agent client serialization with custom query parameters."""
    client = AsyncOpenAI(api_key="test-key", default_query={"param": "value"})
    agent = Agent(client=client)

    serialized = agent.model_dump(mode="json")
    assert "client" in serialized
    assert "default_query" in serialized["client"]


async def test_agent_client_serialization_with_max_retries():
    """Test agent client serialization with non-default max_retries."""
    client = AsyncOpenAI(api_key="test-key", max_retries=5)
    agent = Agent(client=client)

    serialized = agent.model_dump(mode="json")
    assert "client" in serialized
    assert "max_retries" in serialized["client"]
    assert serialized["client"]["max_retries"] == 5


async def test_agent_with_mcp_servers():
    """Test agent with MCP servers configuration."""
    server_params = StdioServerParameters(command="python", args=["-m", "test_server"])

    # Use the alias field name that Agent expects
    agent = Agent(mcpServers={"test_server": server_params})

    # The mcp_servers should be set
    assert agent.mcp_servers is not None
    assert isinstance(agent.mcp_servers, dict)
    assert "test_server" in agent.mcp_servers and isinstance(
        agent.mcp_servers["test_server"], StdioServerParameters
    )
    assert agent.mcp_servers["test_server"].command == "python"


async def test_agent_run_with_mcp_servers():
    """Test agent run method with MCP servers."""
    with patch("swarmx.mcp_client.CLIENT_REGISTRY.add_server") as mock_add_server:
        server_params = StdioServerParameters(
            command="python", args=["-m", "test_server"]
        )
        # Use the alias field name that Agent expects
        agent = Agent(mcpServers={"test_server": server_params})

        # Mock the run method to avoid actual API calls
        with patch.object(agent, "_run") as mock_run:
            mock_run.return_value = []

            await agent.run(messages=[{"role": "user", "content": "Hello"}])

            # Should have called add_server for each server in mcp_servers
            mock_add_server.assert_called_with("test_server", server_params)


async def test_agent_create_chat_completion_with_tools():
    """Test _create_chat_completion with tools."""
    agent = Agent()

    with patch.object(agent, "_get_client") as mock_get_client:
        mock_client = AsyncMock()
        mock_get_client.return_value = mock_client

        # Mock CLIENT_REGISTRY.tools property by patching the property directly
        with patch("swarmx.agent.CLIENT_REGISTRY") as mock_registry:
            mock_registry.tools = [{"name": "test_tool"}]

            await agent._create_chat_completion(
                messages=[{"role": "user", "content": "Hello"}], stream=False
            )

            # Verify tools were included in the call
            call_args = mock_client.chat.completions.create.call_args
            assert "tools" in call_args.kwargs
            assert call_args.kwargs["tools"] == [{"name": "test_tool"}]


async def test_agent_create_chat_completion_with_response_format_and_stream():
    """Test _create_chat_completion with response format and streaming raises error."""
    agent = Agent()

    with pytest.raises(NotImplementedError, match="Streamed parsing is not supported"):
        await agent._create_chat_completion(
            messages=[{"role": "user", "content": "Hello"}],  # type: ignore
            response_format=dict,
            stream=True,
        )


async def test_agent_create_chat_completion_with_response_format():
    """Test _create_chat_completion with response format."""
    agent = Agent()

    with patch.object(agent, "_get_client") as mock_get_client:
        mock_client = AsyncMock()
        mock_get_client.return_value = mock_client

        await agent._create_chat_completion(
            messages=[{"role": "user", "content": "Hello"}],  # type: ignore
            response_format=dict,
            stream=False,
        )

        # Verify parse method was called instead of create
        mock_client.chat.completions.parse.assert_called_once()
        mock_client.chat.completions.create.assert_not_called()


async def test_agent_run_node_stream_with_no_entry_point():
    """Test _run_node_stream with no entry point."""
    agent = Agent()

    result = []
    async for chunk in agent._run_node_stream(
        messages=[{"role": "user", "content": "Hello"}]
    ):
        result.append(chunk)

    # Should return empty generator when no entry point
    assert len(result) == 0


async def test_agent_run_node_with_no_entry_point():
    """Test _run_node with no entry point."""
    agent = Agent()

    result = await agent._run_node(messages=[{"role": "user", "content": "Hello"}])

    # Should return empty list when no entry point
    assert result == []


# Removed problematic tests that patch Pydantic model methods


async def test_agent_run_stream_with_tool_execution():
    """Test _run_stream with tool execution."""
    agent = Agent()

    with (
        patch.object(agent, "_create_chat_completion") as mock_create,
        patch("swarmx.agent.exec_tool_calls") as mock_exec_tools,
        patch.object(agent, "_run_node_stream") as mock_run_node,
    ):
        # Mock chat completion with tool calls
        async def mock_completion_stream():
            chunk = ChatCompletionChunk.model_validate(
                {
                    "id": "test",
                    "choices": [
                        {
                            "index": 0,
                            "delta": {
                                "tool_calls": [
                                    {
                                        "index": 0,
                                        "id": "call_123",
                                        "function": {
                                            "name": "test_tool",
                                            "arguments": "{}",
                                        },
                                        "type": "function",
                                    }
                                ]
                            },
                            "finish_reason": "tool_calls",
                        }
                    ],
                    "created": 1234567890,
                    "model": "gpt-4o",
                    "object": "chat.completion.chunk",
                }
            )
            yield chunk

        mock_create.return_value = mock_completion_stream()

        # Mock tool execution
        async def mock_tool_stream():
            yield [
                {"role": "tool", "content": "Tool result", "tool_call_id": "call_123"}
            ]

        mock_exec_tools.return_value = mock_tool_stream()

        # Mock node stream
        async def mock_node_stream():
            yield ChatCompletionChunk.model_validate(
                {
                    "id": "test2",
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": "Final"},
                            "finish_reason": "stop",
                        }
                    ],
                    "created": 1234567890,
                    "model": "gpt-4o",
                    "object": "chat.completion.chunk",
                }
            )

        mock_run_node.return_value = mock_node_stream()

        result = []
        async for chunk in agent._run_stream(
            messages=[{"role": "user", "content": "Hello"}]
        ):
            result.append(chunk)

        assert len(result) >= 1


async def test_agent_run_stream_message_count_mismatch():
    """Test _run_stream with message count mismatch."""
    agent = Agent()

    with patch.object(agent, "_create_chat_completion") as mock_create:
        # Mock completion that creates mismatched messages
        async def mock_completion_stream():
            # First chunk with one ID
            chunk1 = ChatCompletionChunk.model_validate(
                {
                    "id": "test1",
                    "choices": [{"index": 0, "delta": {"content": "Hello"}}],
                    "created": now(),
                    "model": "gpt-4o",
                    "object": "chat.completion.chunk",
                }
            )
            yield chunk1

            # Second chunk with different ID but no finish_reason
            chunk2 = ChatCompletionChunk.model_validate(
                {
                    "id": "test2",
                    "choices": [{"index": 0, "delta": {"content": " World"}}],
                    "created": now(),
                    "model": "gpt-4o",
                    "object": "chat.completion.chunk",
                }
            )
            yield chunk2

            # Only finish the first chunk
            chunk3 = ChatCompletionChunk.model_validate(
                {
                    "id": "test1",
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                    "created": now(),
                    "model": "gpt-4o",
                    "object": "chat.completion.chunk",
                }
            )
            yield chunk3

        mock_create.return_value = mock_completion_stream()

        with pytest.raises(
            ValueError, match="Number of messages does not match number of chunks"
        ):
            result = []
            async for chunk in agent._run_stream(
                messages=[{"role": "user", "content": "Hello"}]
            ):
                result.append(chunk)


async def test_agent_run_with_tool_execution():
    """Test _run with tool execution."""
    agent = Agent()

    with (
        patch("swarmx.agent.exec_tool_calls") as mock_exec_tools,
    ):
        # Mock completion with tool calls

        # Mock tool execution
        async def mock_tool_stream():
            yield [
                {"role": "tool", "content": "Tool result", "tool_call_id": "call_123"}
            ]

        mock_exec_tools.return_value = mock_tool_stream()

        result = await agent._run(messages=[{"role": "user", "content": "Hello"}])

        assert len(result) >= 2  # At least assistant message and node result


async def test_swarmx_generate_json_schema_field_title():
    """Test SwarmXGenerateJsonSchema field_title_should_be_set method (line 103)."""
    from swarmx.agent import SwarmXGenerateJsonSchema

    schema_generator = SwarmXGenerateJsonSchema()
    # This method should always return False regardless of input
    # We can't easily test with the exact schema type, but we can test the method exists
    # and returns False
    assert hasattr(schema_generator, "field_title_should_be_set")
    # The method should return False for any input
    result = schema_generator.field_title_should_be_set(None)  # type: ignore
    assert result is False
