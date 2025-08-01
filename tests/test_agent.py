import pytest
from openai.types.chat import (
    ChatCompletionMessageParam,
    ChatCompletionMessageToolCallParam,
)

from swarmx import Agent
from swarmx.agent import Edge, exec_tool_calls

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
    result = await exec_tool_calls(tool_calls)
    assert isinstance(result, list)
    assert len(result) > 0


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
    assert any(e.source == "agent1" and e.target == "agent2" for e in main_agent.edges)
    assert any(e.source == "agent2" and e.target == "agent3" for e in main_agent.edges)

    # Test serialization and deserialization
    serialized = main_agent.model_dump(mode="json")
    loaded_agent = Agent(**serialized)
    assert "agent1" in loaded_agent.nodes
    assert "agent2" in loaded_agent.nodes
    assert "agent3" in loaded_agent.nodes
    assert any(
        e.source == "agent1" and e.target == "agent2" for e in loaded_agent.edges
    )
    assert any(
        e.source == "agent2" and e.target == "agent3" for e in loaded_agent.edges
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
    messages = [{"role": "user", "content": "Create a fantasy story"}]
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
