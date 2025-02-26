import networkx as nx
import pytest

from swarmx import Agent, Swarm


@pytest.fixture
def sample_agent():
    return Agent(name="TestAgent")


@pytest.fixture
def sample_swarm():
    return Swarm()


def test_add_agent_node(sample_agent, sample_swarm):
    sample_swarm.add_node(sample_agent)

    # Verify node exists in graph
    assert sample_agent.id in sample_swarm._G.nodes

    # Verify node attributes
    node_data = sample_swarm._G.nodes[sample_agent.id]
    assert node_data["type"] == "agent"
    assert node_data["name"] == "TestAgent"


def test_add_swarm_node(sample_swarm):
    sub_swarm = Swarm()
    sample_swarm.add_node(sub_swarm)

    # Verify node exists and has correct type
    assert sub_swarm.id in sample_swarm._G.nodes
    node_data = sample_swarm._G.nodes[sub_swarm.id]
    assert node_data["type"] == "swarm"

    # Verify subgraph data
    assert "nodes" in node_data
    assert "links" in node_data


def test_add_edge(sample_agent, sample_swarm):
    agent2 = Agent(name="TestAgent2")

    sample_swarm.add_node(sample_agent)
    sample_swarm.add_node(agent2)
    sample_swarm.add_edge(sample_agent, agent2)

    # Verify edge exists
    assert sample_swarm._G.has_edge(sample_agent.id, agent2.id)


def test_graph_structure(sample_swarm):
    agent1 = Agent(name="Agent1")
    agent2 = Agent(name="Agent2")
    swarm = Swarm()

    sample_swarm.add_node(agent1)
    sample_swarm.add_node(agent2)
    sample_swarm.add_node(swarm)

    sample_swarm.add_edge(agent1, swarm)
    sample_swarm.add_edge(swarm, agent2)

    # Verify graph structure
    assert sample_swarm._G.number_of_nodes() == 3
    assert sample_swarm._G.number_of_edges() == 2
    assert list(nx.edge_dfs(sample_swarm._G)) == [
        (agent1.id, swarm.id),
        (swarm.id, agent2.id),
    ]


def test_can_be_added_property(sample_swarm):
    agent = Agent()
    sample_swarm.add_node(agent)

    # Single node should be both root and leaf
    assert sample_swarm._can_be_added_as_node

    # Add second node and edge
    agent2 = Agent()
    sample_swarm.add_node(agent2)
    sample_swarm.add_edge(agent, agent2)

    # Now has one root and one leaf
    assert sample_swarm._can_be_added_as_node
