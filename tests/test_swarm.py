"""Tests for Swarm graph functionality."""

import networkx as nx
import pytest

from swarmx.agent import Agent
from swarmx.node import Node
from swarmx.swarm import Swarm, swarm_serializer

pytestmark = pytest.mark.anyio


class MockNode:
    """Mock Node for testing."""

    def __init__(self, name: str):
        self.name = name

    def __hash__(self):
        return hash(self.name)

    async def __call__(self, arguments, *args, **kwargs):
        return arguments


def test_swarm_initialization():
    """Test Swarm can be initialized as a DiGraph."""
    swarm = Swarm(name="swarm")
    assert isinstance(swarm, nx.DiGraph)
    assert len(swarm.nodes) == 0
    assert len(swarm.edges) == 0
    assert swarm.name == "swarm"


def test_swarm_initialization_with_name():
    """Test Swarm can be initialized with a custom name."""
    swarm = Swarm(name="my_workflow")
    assert swarm.name == "my_workflow"
    assert swarm.graph["name"] == "my_workflow"


def test_swarm_is_hashable():
    """Test Swarm is hashable using its name."""
    swarm1 = Swarm(name="swarm1")
    swarm2 = Swarm(name="swarm2")
    swarm3 = Swarm(name="swarm1")

    # Can be used in sets
    swarms = {swarm1, swarm2, swarm3}
    assert len(swarms) == 2  # swarm1 and swarm3 have same hash

    # Can be used in dicts
    swarm_dict = {swarm1: "first", swarm2: "second"}
    assert len(swarm_dict) == 2


def test_swarm_is_node_protocol():
    """Test Swarm implements Node protocol."""
    swarm = Swarm(name="test_swarm")

    # Has required name attribute
    assert hasattr(swarm, "name")
    assert swarm.name == "test_swarm"

    # Is hashable
    assert isinstance(swarm, Node)
    hash(swarm)

    # Has __call__ method
    assert hasattr(swarm, "__call__")
    assert callable(swarm)


async def test_swarm_call_not_implemented():
    """Test Swarm __call__ raises NotImplementedError (until implemented)."""
    swarm = Swarm(name="test_swarm")

    with pytest.raises(
        NotImplementedError, match="Swarm workflow execution is not yet implemented"
    ):
        await swarm({"messages": []})


def test_swarm_can_be_nested():
    """Test Swarm can be added as a node to another Swarm."""
    parent_swarm = Swarm(name="parent")
    child_swarm = Swarm(name="child_workflow")

    # Add child swarm as a node
    parent_swarm.add_node(child_swarm)

    # Verify child was added with its name as ID
    assert "child_workflow" in parent_swarm.nodes
    assert parent_swarm.nodes["child_workflow"]["swarm"] == child_swarm


def test_swarm_add_node_with_node_object():
    """Test adding a Node object (Agent/Tool/Swarm) to Swarm."""

    swarm = Swarm(name="test_swarm")
    agent = Agent(name="agent_a", instructions="Test agent", model="gpt-4")

    swarm.add_node(agent)

    # Verify node was added with name as ID
    assert "agent_a" in swarm.nodes
    assert swarm.nodes["agent_a"]["type"] == "agent"
    assert swarm.nodes["agent_a"]["agent"] == agent


def test_swarm_add_node_with_id_and_attributes():
    """Test adding a node with ID and attributes."""
    swarm = Swarm(name="swarm")

    swarm.add_node("agent_b", type="agent", model="gpt-4", temperature=0.7)

    # Verify node and attributes
    assert "agent_b" in swarm.nodes
    assert swarm.nodes["agent_b"]["type"] == "agent"
    assert swarm.nodes["agent_b"]["model"] == "gpt-4"
    assert swarm.nodes["agent_b"]["temperature"] == 0.7


def test_swarm_add_node_with_node_and_attrs_raises_error():
    """Test that providing attrs with Node object raises error."""

    swarm = Swarm(name="test_swarm")
    agent = Agent(name="agent_a", instructions="Test agent", model="gpt-4")

    with pytest.raises(
        ValueError, match="Cannot provide attributes when adding a Node object"
    ):
        swarm.add_node(agent, type="agent")


def test_swarm_add_multiple_nodes():
    """Test adding multiple nodes of different types."""

    swarm = Swarm(name="test_swarm")

    agent1 = Agent(name="agent_a", instructions="Agent A", model="gpt-4")
    agent2 = Agent(name="agent_b", instructions="Agent B", model="gpt-4")

    swarm.add_node(agent1)
    swarm.add_node("agent_c", type="agent")
    swarm.add_node(agent2)
    swarm.add_node("router", type="router")

    assert len(swarm.nodes) == 4
    assert "agent_a" in swarm.nodes
    assert "agent_b" in swarm.nodes
    assert "agent_c" in swarm.nodes
    assert "router" in swarm.nodes

    # Verify Agent objects are stored with type and agent keys
    assert swarm.nodes["agent_a"]["type"] == "agent"
    assert swarm.nodes["agent_a"]["agent"] == agent1

    # Verify regular nodes have attributes
    assert swarm.nodes["agent_c"]["type"] == "agent"


def test_swarm_dag_with_conditional_edges():
    """Test Swarm graph as DAG with conditional edge routing."""
    # TODO: Implement GraphModel for graph serialization/deserialization
    pytest.skip("GraphModel not yet implemented")


def test_swarm_dag_prevents_unconditional_cycles():
    """Test that we can represent DAG invariant: cycles must have conditional edges."""
    # TODO: Implement GraphModel for graph serialization/deserialization
    pytest.skip("GraphModel not yet implemented")


def test_swarm_serializer_with_tuple_source():
    """Test swarm_serializer properly handles edges with tuple sources."""

    swarm = Swarm(name="test_workflow")
    agent_a = Agent(name="agent_a", instructions="Agent A", model="gpt-4")
    agent_b = Agent(name="agent_b", instructions="Agent B", model="gpt-4")
    agent_c = Agent(name="agent_c", instructions="Agent C", model="gpt-4")

    swarm.add_node(agent_a)
    swarm.add_node(agent_b)
    swarm.add_node(agent_c)

    # Add edge with simple source
    swarm.add_edge("agent_a", "agent_c", route="simple")

    # Add edge with tuple source (representing AND logic - both must complete)
    swarm.add_edge("agent_b", "agent_c", co_source=("agent_a", "agent_b"))

    # Serialize
    serialized = swarm_serializer(swarm)

    # Verify structure
    assert serialized["type"] == "swarm"
    assert serialized["swarm"]["name"] == "test_workflow"
    assert len(serialized["swarm"]["nodes"]) == 3
    assert len(serialized["swarm"]["edges"]) == 2

    # Verify edges include source, target, and attributes
    edges_dict = {(e["source"], e["target"]): e for e in serialized["swarm"]["edges"]}
    assert ("agent_a", "agent_c") in edges_dict
    assert ("agent_b", "agent_c") in edges_dict
