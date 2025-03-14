import datetime
import operator
import sys

import pytest
from mcp.client.stdio import StdioServerParameters

from swarmx import Agent, Swarm

pytestmark = pytest.mark.anyio


async def test_swarm_creation():
    """Test creating a basic swarm with a single agent node."""
    agent = Agent(name="test_agent", instructions="You are a helpful assistant.")

    swarm = Swarm()
    swarm.add_node(0, type="agent", agent=agent)

    assert 0 in swarm.nodes
    assert swarm.root == 0
    assert isinstance(swarm.nodes[0]["agent"], Agent)
    assert swarm.nodes[0]["agent"].name == "test_agent"


async def test_swarm_with_multiple_agents():
    """Test creating a swarm with multiple agents in a directed graph."""
    agent1 = Agent(name="agent1")
    agent2 = Agent(name="agent2")
    agent3 = Agent(name="agent3")

    swarm = Swarm()
    swarm.add_node(0, type="agent", agent=agent1)
    swarm.add_node(1, type="agent", agent=agent2)
    swarm.add_node(2, type="agent", agent=agent3)

    swarm.add_edge(0, 1)
    swarm.add_edge(0, 2)

    assert list(swarm.successors(0)) == [1, 2]
    assert swarm.root == 0
    assert len(swarm.nodes) == 3
    assert len(swarm.edges) == 2


async def test_swarm_serialization():
    """Test serializing and deserializing a swarm."""
    agent1 = Agent(name="agent1")
    agent2 = Agent(name="agent2")

    # Create a swarm
    swarm = Swarm()
    swarm.add_node(0, type="agent", agent=agent1)
    swarm.add_node(1, type="agent", agent=agent2)
    swarm.add_edge(0, 1, priority=1)

    # Serialize
    serialized = swarm.model_dump(mode="json")

    # Check serialization result
    assert "nodes" in serialized
    assert "edges" in serialized
    assert "graph" in serialized
    assert len(serialized["nodes"]) == 2
    assert len(serialized["edges"]) == 1
    assert serialized["edges"][0]["priority"] == 1

    # Deserialize
    deserialized = Swarm.model_validate(serialized)

    # Check graph structure is preserved
    assert list(deserialized.nodes) == [0, 1]
    assert list(deserialized.edges) == [(0, 1)]
    assert deserialized.edges[0, 1]["priority"] == 1
    assert deserialized.nodes[0]["agent"].name == "agent1"
    assert deserialized.nodes[1]["agent"].name == "agent2"


async def test_swarm_with_mcp_servers():
    """Test swarm with MCP servers configuration."""
    agent = Agent(name="agent")

    # Create a swarm with MCP server configuration
    swarm = Swarm(
        mcpServers={
            "time": StdioServerParameters(
                command=sys.executable,
                args=["-m", "mcp_server_time", "--local-timezone", "UTC"],
            )
        }
    )
    swarm.add_node(0, type="agent", agent=agent)

    # Check server config is present
    assert "time" in swarm.mcp_servers
    assert swarm.mcp_servers["time"].command == sys.executable
    assert swarm.mcp_servers["time"].args == [
        "-m",
        "mcp_server_time",
        "--local-timezone",
        "UTC",
    ]

    # Test serialization with MCP servers
    serialized = swarm.model_dump(mode="json")
    assert "mcpServers" in serialized
    assert "time" in serialized["mcpServers"]


async def test_swarm_root_detection():
    """Test root node detection in a swarm."""
    agent1 = Agent(name="agent1")
    agent2 = Agent(name="agent2")
    agent3 = Agent(name="agent3")

    # Create a graph with one root
    swarm = Swarm()
    swarm.add_node(0, type="agent", agent=agent1)
    swarm.add_node(1, type="agent", agent=agent2)
    swarm.add_node(2, type="agent", agent=agent3)
    swarm.add_edge(0, 1)
    swarm.add_edge(1, 2)

    assert swarm.root == 0

    # Test error when multiple roots
    swarm2 = Swarm()
    swarm2.add_node(0, type="agent", agent=agent1)
    swarm2.add_node(1, type="agent", agent=agent2)

    with pytest.raises(ValueError, match="Swarm must have exactly one root node"):
        _ = swarm2.root


async def test_swarm_run_with_tool_execution():
    """Test swarm run with tool execution."""

    def calculator(a: int, b: int, operation: str) -> str:
        """Calculate the result of the operation."""
        return str(
            {
                "add": operator.add,
                "subtract": operator.sub,
                "multiply": operator.mul,
                "divide": operator.truediv,
            }[operation](a, b)
        )

    agent = Agent(name="agent", tools=[calculator])
    swarm = Swarm()
    swarm.add_node(0, type="agent", agent=agent)

    # Execute the run method
    messages = [
        {
            "role": "user",
            "content": "Calculate 5+3, just answer with the result without any other text",
        }
    ]
    new_messages = await swarm.run(
        messages=messages, context_variables={"initial": "context"}, stream=False
    )

    # Verify the result
    assert len(new_messages) == 3  # Assistant message + tool + assistant
    assert new_messages[0]["role"] == "assistant"
    assert new_messages[1]["role"] == "tool"
    assert new_messages[1]["content"] == "8"
    assert new_messages[2]["role"] == "assistant"
    assert new_messages[2]["content"] == "8"


async def test_mcp_tool_call():
    """Test swarm with MCP server integration."""
    agent = Agent(name="agent")

    # Create a swarm with MCP server
    swarm = Swarm(
        mcpServers={
            "time": StdioServerParameters(
                command=sys.executable,
                args=["-m", "mcp_server_time", "--local-timezone", "UTC"],
            )
        }
    )
    swarm.add_node(0, type="agent", agent=agent)

    messages = await swarm.run(
        messages=[
            {
                "role": "user",
                "content": "What's the time now? Answer me in 'HH:MM:SS' format",
            }
        ]
    )
    t = datetime.datetime.strptime(messages[-1]["content"], "%H:%M:%S")
    n = datetime.datetime.now(datetime.UTC)
    # make t as UTC
    t = t.astimezone(datetime.UTC)
    assert t - n < datetime.timedelta(seconds=2)


async def test_swarm_of_swarms():
    """Test swarm that contains another swarm."""
    # Create inner swarm
    inner_agent = Agent(name="inner_agent")
    inner_swarm = Swarm()
    inner_swarm.add_node(0, type="agent", agent=inner_agent)

    # Create outer swarm with inner swarm as a node
    agent = Agent(name="agent")
    outer_swarm = Swarm()
    outer_swarm.add_node(0, type="agent", agent=agent)
    outer_swarm.add_node(1, type="swarm", swarm=inner_swarm)
    outer_swarm.add_edge(0, 1)

    # Verify the structure
    assert "agent" in outer_swarm.nodes[0]
    assert "swarm" in outer_swarm.nodes[1]
    assert outer_swarm.nodes[1]["swarm"] == inner_swarm

    # Test serialization of nested swarm
    serialized = outer_swarm.model_dump(mode="json")
    assert len(serialized["nodes"]) == 2
    assert serialized["nodes"][1]["type"] == "swarm"
    assert "nodes" in serialized["nodes"][1]["swarm"]

    # Test deserialization of nested swarm
    deserialized = Swarm.model_validate(serialized)
    assert "swarm" in deserialized.nodes[1]
    assert isinstance(deserialized.nodes[1]["swarm"], Swarm)
    assert deserialized.nodes[1]["swarm"].nodes[0]["agent"].name == "inner_agent"


async def test_swarm_run_with_max_turns():
    """Test swarm run with max_turns limit."""
    agent = Agent(name="agent")
    swarm = Swarm()
    swarm.add_node(0, type="agent", agent=agent)

    # Run with max_turns=1
    messages = await swarm.run(
        messages=[{"role": "user", "content": "Test max turns"}], max_turns=1
    )

    # Verify only one turn was executed
    assert len(messages) == 1
    assert messages[0]["content"] == "First response"


async def test_swarm_validation_errors():
    """Test error handling in Swarm validation."""
    swarm = Swarm()

    # Test adding edge with non-existent nodes
    with pytest.raises(ValueError, match="Node 99 not found"):
        swarm.add_edge(99, 100)

    # Add a node first
    agent = Agent(name="agent")
    swarm.add_node(0, type="agent", agent=agent)

    # Test adding edge with one non-existent node
    with pytest.raises(ValueError, match="Node 99 not found"):
        swarm.add_edge(0, 99)

    # Test deserializing invalid data
    with pytest.raises(Exception):
        Swarm.model_validate({"nodes": [{"invalid": "data"}]})


async def test_handoff():
    client = Swarm()
    english_agent = Agent(
        name="English Agent",
        instructions="You only speak English.",
        functions=["tests.functions.transfer_to_spanish_agent"],
    )
    message_input = "Hola. ¿Como estás?"
    client.add_node(0, type="agent", agent=english_agent)
    messages = await client.run(
        messages=[{"role": "user", "content": message_input}],
    )
    assert messages[-1].get("name").startswith("Spanish Agent")


@pytest.fixture
def simple_swarm():
    swarm = Swarm()
    agent1 = Agent(name="Agent1", instructions="You are Agent 1")
    agent2 = Agent(name="Agent2", instructions="You are Agent 2")

    swarm.add_node(0, type="agent", agent=agent1)
    swarm.add_node(1, type="agent", agent=agent2)
    swarm.add_edge(0, 1)

    return swarm


def test_swarm_root(simple_swarm):
    assert simple_swarm.root == 0


def test_next_node(simple_swarm):
    assert simple_swarm._next_node == 2


def test_swarm_add_edge_validation():
    swarm = Swarm()
    agent1 = Agent(name="Agent1")
    swarm.add_node(0, type="agent", agent=agent1)

    # Test with non-existent source node
    with pytest.raises(ValueError, match="Node 999 not found"):
        swarm.add_edge(999, 0)

    # Test with non-existent target node
    with pytest.raises(ValueError, match="Node 999 not found"):
        swarm.add_edge(0, 999)
