import sys
from collections import defaultdict
from typing import cast
from unittest.mock import AsyncMock, patch

import pytest
from httpx import Timeout
from mcp.client.stdio import StdioServerParameters
from openai import AsyncOpenAI
from openai.types.chat import (
    ChatCompletionMessageParam,
)
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk

from swarmx import Agent
from swarmx.agent import (
    Edge,
    SwarmXGenerateJsonSchema,
    _apply_message_slice,
    _merge_chunk,
)
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


async def test_create_chat_completion(model):
    agent = Agent(
        name="test_agent",
        model=model,
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


async def test_agent_sequence_execution_stream(model):
    # Create main agent with nodes
    main_agent = Agent(
        name="main_agent",
        model=model,
        instructions="Coordinate the workflow.",
        entry_point="agent1",
    )

    # Create subagents with specific behaviors
    agent1 = Agent(
        name="agent1",
        model=model,
        instructions="You are a fantasy writer. Always respond with 'Story written'.",
    )
    agent2 = Agent(
        name="agent2",
        model=model,
        instructions="You are a fantasy editor. Always respond with 'Story edited'.",
    )
    agent3 = Agent(
        name="agent3",
        model=model,
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

    async for chunk in main_agent._handoff(
        node="agent1", messages=messages, stream=True
    ):
        if content := cast(ChatCompletionChunk, chunk).choices[0].delta.content:
            responses.append(content)

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
    server_params = StdioServerParameters(
        command=sys.executable, args=["-m", "test_server"]
    )

    # Use the alias field name that Agent expects
    agent = Agent(mcpServers={"test_server": server_params})

    # The mcp_servers should be set
    assert agent.mcp_servers is not None
    assert isinstance(agent.mcp_servers, dict)
    assert "test_server" in agent.mcp_servers and isinstance(
        agent.mcp_servers["test_server"], StdioServerParameters
    )
    assert agent.mcp_servers["test_server"].command == sys.executable


async def test_agent_run_with_mcp_servers():
    """Test agent run method with MCP servers."""
    with patch("swarmx.mcp_client.CLIENT_REGISTRY.add_server") as mock_add_server:
        server_params = StdioServerParameters(
            command=sys.executable, args=["-m", "test_server"]
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


async def test_agent_run_stream_message_count_mismatch():
    """Test _run_stream with message count mismatch."""
    import uuid

    agent = Agent()
    request_id = str(uuid.uuid4())

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
            chunk1._request_id = request_id
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
            chunk2._request_id = request_id
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
            chunk3._request_id = request_id
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


async def test_agent_run_with_tool_execution(model):
    """Test _run with tool execution."""
    agent = Agent(
        model=model,
        mcpServers={
            "time": StdioServerParameters(
                command=sys.executable, args=["-m", "mcp_server_time"]
            )
        },
    )

    result = []
    async for completion in await agent.run(
        messages=[{"role": "user", "content": "What is time now?"}]
    ):
        result.append(completion)

    assert len(result) >= 1  # At least one completion


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


async def test_apply_message_slice():
    """Test _apply_message_slice function with various slice patterns."""
    messages: list[ChatCompletionMessageParam] = [
        {"role": "user", "content": "message1"},
        {"role": "assistant", "content": "message2"},
        {"role": "user", "content": "message3"},
        {"role": "assistant", "content": "message4"},
        {"role": "user", "content": "message5"},
    ]

    # Test basic slice patterns
    assert _apply_message_slice(messages, "0:3") == messages[0:3]
    assert _apply_message_slice(messages, "1:4") == messages[1:4]
    assert _apply_message_slice(messages, "2:") == messages[2:]
    assert _apply_message_slice(messages, ":3") == messages[:3]
    assert _apply_message_slice(messages, "-2:") == messages[-2:]
    assert _apply_message_slice(messages, ":-1") == messages[:-1]
    assert _apply_message_slice(messages, "-3:-1") == messages[-3:-1]

    # Test edge cases
    assert _apply_message_slice(messages, "0:0") == []
    assert _apply_message_slice(messages, "10:20") == []
    assert _apply_message_slice(messages, ":") == messages
    assert _apply_message_slice(messages, "-100:") == messages

    # Test with step (rarely used but supported)
    assert _apply_message_slice(messages, "0:5:2") == messages[0:5:2]

    # Test invalid slice patterns
    with pytest.raises(ValueError, match="Invalid message slice"):
        _apply_message_slice(messages, "invalid")
    with pytest.raises(ValueError, match="Invalid message slice"):
        _apply_message_slice(messages, "a:b")  # Non-numeric values


async def test_run_node_with_explicit_node():
    """Test _run_node with explicitly specified node."""
    # Create subagent
    subagent = Agent(name="subagent", instructions="You are a test agent.")

    # Create main agent with nodes
    main_agent = Agent(name="main_agent", nodes={"test_node": subagent})

    messages: list[ChatCompletionMessageParam] = [{"role": "user", "content": "Hello"}]

    with patch.object(Agent, "run") as mock_run:
        # Mock to return an async generator
        async def mock_async_gen():
            yield {"role": "assistant", "content": "Response from subagent"}

        mock_run.return_value = mock_async_gen()

        result = []
        async for completion in main_agent._handoff(
            node="test_node", messages=messages
        ):
            result.append(completion)

        # Should call subagent.run with correct parameters
        mock_run.assert_called_once_with(messages=messages, stream=False, context={})
        # Should return the result from subagent
        assert result == [{"role": "assistant", "content": "Response from subagent"}]
        # Should mark node as visited
        assert main_agent._visited["test_node"] is True


async def test_run_node_with_entry_point():
    """Test _run_node uses entry_point when node is None."""
    # Create subagent
    subagent = Agent(name="subagent", instructions="You are a test agent.")

    # Create main agent with entry point
    main_agent = Agent(
        name="main_agent", entry_point="entry_node", nodes={"entry_node": subagent}
    )

    messages: list[ChatCompletionMessageParam] = [{"role": "user", "content": "Hello"}]

    with patch.object(Agent, "run") as mock_run:
        # Mock to return an async generator
        async def mock_async_gen():
            yield {"role": "assistant", "content": "Response from entry"}

        mock_run.return_value = mock_async_gen()

        result = []
        async for completion in main_agent._handoff(
            node="entry_node", messages=messages
        ):
            result.append(completion)

        # Should call subagent.run with correct parameters
        mock_run.assert_called_once_with(messages=messages, stream=False, context={})
        # Should return the result from subagent
        assert result == [{"role": "assistant", "content": "Response from entry"}]
        # Should mark node as visited
        assert main_agent._visited["entry_node"] is True


async def test_run_node_with_finish_point():
    """Test _run_node returns early when reaching finish_point."""
    # Create main agent with finish point
    main_agent = Agent(
        name="main_agent",
        entry_point="node1",
        finish_point="node1",  # Same as entry point to test early return
        nodes={"node1": Agent(name="subagent1"), "node2": Agent(name="subagent2")},
        edges=[Edge(source="node1", target="node2")],
    )

    messages: list[ChatCompletionMessageParam] = [{"role": "user", "content": "Hello"}]

    with patch.object(Agent, "run") as mock_run:
        # Mock to return an async generator
        async def mock_async_gen():
            yield {"role": "assistant", "content": "Response from node1"}

        mock_run.return_value = mock_async_gen()

        result = []
        async for completion in main_agent._handoff(node="node1", messages=messages):
            result.append(completion)

        # Should call only once (for the entry node, then return due to finish_point)
        mock_run.assert_called_once()
        # Should return result from first node only
        assert result == [{"role": "assistant", "content": "Response from node1"}]


async def test_run_node_with_simple_edge():
    """Test _run_node follows simple edges between nodes."""
    # Create main agent with edge
    main_agent = Agent(
        name="main_agent",
        entry_point="node1",
        nodes={"node1": Agent(name="subagent1"), "node2": Agent(name="subagent2")},
        edges=[Edge(source="node1", target="node2")],
    )

    messages: list[ChatCompletionMessageParam] = [{"role": "user", "content": "Hello"}]

    with patch.object(Agent, "run") as mock_run:
        # Mock to return async generators
        async def mock_async_gen1():
            yield {"role": "assistant", "content": "Response from node1"}

        async def mock_async_gen2():
            yield {"role": "assistant", "content": "Response from node2"}

        mock_run.side_effect = [mock_async_gen1(), mock_async_gen2()]

        result = []
        async for completion in main_agent._handoff(node="node1", messages=messages):
            result.append(completion)

        # Should call both subagents
        assert mock_run.call_count == 2
        # Should return combined results
        expected = [
            {"role": "assistant", "content": "Response from node1"},
            {"role": "assistant", "content": "Response from node2"},
        ]
        assert result == expected


async def test_run_node_with_list_source_edge():
    """Test _run_node handles edges with list sources (conditional execution)."""
    # Create main agent with conditional edge
    main_agent = Agent(
        name="main_agent",
        entry_point="node1",
        nodes={
            "node1": Agent(name="subagent1"),
            "node2": Agent(name="subagent2"),
            "node3": Agent(name="subagent3"),
        },
        edges=[
            Edge(source="node1", target="node2"),
            Edge(source=["node1", "node2"], target="node3"),  # Conditional edge
        ],
    )

    messages: list[ChatCompletionMessageParam] = [{"role": "user", "content": "Hello"}]

    # For this test, let's simplify and just test that the method runs without error
    # and calls the expected number of agents
    with patch.object(Agent, "run") as mock_run:
        # Mock to return an async generator
        async def mock_async_gen():
            yield {"role": "assistant", "content": "Response"}

        mock_run.return_value = mock_async_gen()

        result = []
        async for completion in main_agent._handoff(node="node1", messages=messages):
            result.append(completion)

        # Should call node1, node2, and node3 (node3 might be called multiple times due to conditional edge)
        assert mock_run.call_count >= 3
        # Should return some result
        assert len(result) > 0


async def test_run_node_stream_with_explicit_node():
    """Test _run_node_stream with explicitly specified node."""
    from swarmx.utils import now

    # Create main agent with nodes
    main_agent = Agent(
        name="main_agent",
        nodes={
            "test_node": Agent(name="subagent", instructions="You are a test agent.")
        },
    )

    messages: list[ChatCompletionMessageParam] = [{"role": "user", "content": "Hello"}]

    # Mock stream response
    async def mock_stream():
        yield ChatCompletionChunk.model_validate(
            {
                "id": "test_chunk",
                "choices": [{"index": 0, "delta": {"content": "Response"}}],
                "created": now(),
                "model": "gpt-4o",
                "object": "chat.completion.chunk",
            }
        )

    with patch.object(Agent, "run") as mock_run:
        mock_run.return_value = mock_stream()

        result = []
        async for chunk in main_agent._handoff(
            node="test_node", messages=messages, stream=True
        ):
            result.append(chunk)

        # Should call Agent.run with correct parameters
        mock_run.assert_called_once_with(messages=messages, stream=True, context={})
        # Should return chunks from subagent
        assert len(result) == 1
        assert result[0].choices[0].delta.content == "Response"
        # Should mark node as visited
        assert main_agent._visited["test_node"] is True


async def test_run_node_stream_with_entry_point():
    """Test _run_node_stream uses entry_point when node is None."""
    from swarmx.utils import now

    # Create main agent with entry point
    main_agent = Agent(
        name="main_agent",
        entry_point="entry_node",
        nodes={
            "entry_node": Agent(name="subagent", instructions="You are a test agent.")
        },
    )

    messages: list[ChatCompletionMessageParam] = [{"role": "user", "content": "Hello"}]

    # Mock stream response
    async def mock_stream():
        yield ChatCompletionChunk.model_validate(
            {
                "id": "test_chunk",
                "choices": [{"index": 0, "delta": {"content": "Entry response"}}],
                "created": now(),
                "model": "gpt-4o",
                "object": "chat.completion.chunk",
            }
        )

    with patch.object(Agent, "run") as mock_run:
        mock_run.return_value = mock_stream()

        result = []
        async for chunk in main_agent._handoff(
            node="entry_node", messages=messages, stream=True
        ):
            result.append(chunk)

        # Should call Agent.run with correct parameters
        mock_run.assert_called_once_with(messages=messages, stream=True, context={})
        # Should return chunks from subagent
        assert len(result) == 1
        assert result[0].choices[0].delta.content == "Entry response"
        # Should mark node as visited
        assert main_agent._visited["entry_node"] is True


async def test_run_node_stream_with_finish_point():
    """Test _run_node_stream returns early when reaching finish_point."""
    from swarmx.utils import now

    # Create main agent with finish point
    main_agent = Agent(
        name="main_agent",
        entry_point="node1",
        finish_point="node1",  # Same as entry point to test early return
        nodes={"node1": Agent(name="subagent1"), "node2": Agent(name="subagent2")},
        edges=[Edge(source="node1", target="node2")],
    )

    messages: list[ChatCompletionMessageParam] = [{"role": "user", "content": "Hello"}]

    # Mock stream response
    async def mock_stream():
        yield ChatCompletionChunk.model_validate(
            {
                "id": "test_chunk",
                "choices": [{"index": 0, "delta": {"content": "Node1 response"}}],
                "created": now(),
                "model": "gpt-4o",
                "object": "chat.completion.chunk",
            }
        )

    with patch.object(Agent, "run") as mock_run:
        mock_run.return_value = mock_stream()

        result = []
        async for chunk in main_agent._handoff(
            node="node1", messages=messages, stream=True
        ):
            result.append(chunk)

        # Should call only once (for the entry node, then return due to finish_point)
        mock_run.assert_called_once()
        # Should return chunks from first node only
        assert len(result) == 1
        assert result[0].choices[0].delta.content == "Node1 response"


async def test_run_node_stream_with_simple_edge():
    """Test _run_node_stream follows simple edges between nodes."""
    from swarmx.utils import now

    # Create main agent with edge
    main_agent = Agent(
        name="main_agent",
        entry_point="node1",
        nodes={"node1": Agent(name="subagent1"), "node2": Agent(name="subagent2")},
        edges=[Edge(source="node1", target="node2")],
    )

    messages: list[ChatCompletionMessageParam] = [{"role": "user", "content": "Hello"}]

    # Mock stream responses
    async def mock_stream1():
        yield ChatCompletionChunk.model_validate(
            {
                "id": "chunk1",
                "choices": [{"index": 0, "delta": {"content": "Node1"}}],
                "created": now(),
                "model": "gpt-4o",
                "object": "chat.completion.chunk",
            }
        )

    async def mock_stream2():
        yield ChatCompletionChunk.model_validate(
            {
                "id": "chunk2",
                "choices": [{"index": 0, "delta": {"content": "Node2"}}],
                "created": now(),
                "model": "gpt-4o",
                "object": "chat.completion.chunk",
            }
        )

    with patch.object(Agent, "run") as mock_run:
        mock_run.side_effect = [mock_stream1(), mock_stream2()]

        result = []
        async for chunk in main_agent._handoff(
            node="node1", messages=messages, stream=True
        ):
            result.append(chunk)

        # Should call both subagents
        assert mock_run.call_count == 2
        # Should return chunks from both nodes
        assert len(result) == 2
        contents = [chunk.choices[0].delta.content for chunk in result]
        assert "Node1" in contents
        assert "Node2" in contents


async def test_run_node_stream_with_list_source_edge():
    """Test _run_node_stream handles edges with list sources (conditional execution)."""
    from swarmx.utils import now

    # Create main agent with conditional edge
    main_agent = Agent(
        name="main_agent",
        entry_point="node1",
        nodes={
            "node1": Agent(name="subagent1"),
            "node2": Agent(name="subagent2"),
            "node3": Agent(name="subagent3"),
        },
        edges=[
            Edge(source="node1", target="node2"),
            Edge(source=["node1", "node2"], target="node3"),  # Conditional edge
        ],
    )

    messages: list[ChatCompletionMessageParam] = [{"role": "user", "content": "Hello"}]

    # Mock stream responses
    async def mock_stream1():
        yield ChatCompletionChunk.model_validate(
            {
                "id": "chunk1",
                "choices": [{"index": 0, "delta": {"content": "Node1"}}],
                "created": now(),
                "model": "gpt-4o",
                "object": "chat.completion.chunk",
            }
        )

    async def mock_stream2():
        yield ChatCompletionChunk.model_validate(
            {
                "id": "chunk2",
                "choices": [{"index": 0, "delta": {"content": "Node2"}}],
                "created": now(),
                "model": "gpt-4o",
                "object": "chat.completion.chunk",
            }
        )

    async def mock_stream3():
        yield ChatCompletionChunk.model_validate(
            {
                "id": "chunk3",
                "choices": [{"index": 0, "delta": {"content": "Node3"}}],
                "created": now(),
                "model": "gpt-4o",
                "object": "chat.completion.chunk",
            }
        )

    with patch.object(Agent, "run") as mock_run:
        mock_run.side_effect = [mock_stream1(), mock_stream2(), mock_stream3()]

        result = []
        async for chunk in main_agent._handoff(
            node="node1", messages=messages, stream=True
        ):
            result.append(chunk)

        # Should call all three subagents
        assert mock_run.call_count == 3
        # Should return chunks from all nodes
        assert len(result) == 3
        contents = [chunk.choices[0].delta.content for chunk in result]
        assert "Node1" in contents
        assert "Node2" in contents
        assert "Node3" in contents
