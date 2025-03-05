import networkx as nx
import pytest

from swarmx import Agent, Swarm


@pytest.fixture
def sample_agent():
    return Agent(name="TestAgent")


@pytest.fixture
def sample_swarm():
    return Swarm()


def test_add_agent_node(sample_agent: Agent, sample_swarm: Swarm):
    sample_swarm.add_node(0, sample_agent)

    # Verify node exists in graph0
    assert sample_swarm.nodes[0] is sample_agent


def test_add_swarm_node(sample_swarm: Swarm):
    sub_swarm = Swarm()
    sample_swarm.add_node(0, sub_swarm)

    # Verify node exists and has correct type
    assert sample_swarm.nodes[0] is sub_swarm


def test_add_edge(sample_agent: Agent, sample_swarm: Swarm):
    agent2 = Agent(name="TestAgent2")
    sample_swarm.add_node(0, sample_agent)
    sample_swarm.add_node(1, agent2)

    sample_swarm.add_edge(0, 1)

    # Verify edge exists
    assert sample_swarm.has_edge(0, 1)


def test_graph_structure(sample_swarm: Swarm):
    agent1 = Agent(name="Agent1")
    agent2 = Agent(name="Agent2")
    swarm = Swarm()

    sample_swarm.add_node(0, agent1)
    sample_swarm.add_node(1, agent2)
    sample_swarm.add_node(2, swarm)
    sample_swarm.add_edge(0, 2)
    sample_swarm.add_edge(2, 1)

    # Verify graph structure
    assert sample_swarm.number_of_nodes() == 3
    assert sample_swarm.number_of_edges() == 2
    assert list(nx.edge_dfs(sample_swarm)) == [
        (0, 2),
        (2, 1),
    ]


def test_agent_serialization():
    """Test Agent serialization and deserialization with model_dump and model_validate."""
    # Create an agent
    agent = Agent(
        name="TestAgent", model="gpt-4o", instructions="You are a test agent."
    )

    # Serialize to dict
    agent_dict = agent.model_dump(mode="json")

    # Verify key fields
    assert agent_dict["name"] == "TestAgent"
    assert agent_dict["model"] == "gpt-4o"
    assert agent_dict["instructions"] == "You are a test agent."

    # Deserialize back to agent
    new_agent = Agent.model_validate(agent_dict)

    # Verify the new agent matches the original
    assert new_agent.name == agent.name
    assert new_agent.model == agent.model
    assert new_agent.instructions == agent.instructions


def test_swarm_serialization():
    """Test Swarm serialization and deserialization with model_dump and model_validate."""
    # Create a swarm with agents
    swarm = Swarm()
    agent1 = Agent(name="Agent1", instructions="You are agent 1")
    agent2 = Agent(name="Agent2", instructions="You are agent 2")
    swarm.add_node(0, agent1)
    swarm.add_node(1, agent2)

    # Add nodes and edge
    swarm.add_edge(0, 1)

    # Serialize to dict
    swarm_dict = swarm.model_dump()

    # Deserialize back to swarm
    new_swarm = Swarm.model_validate(swarm_dict)

    # Verify nodes and edges were preserved
    assert len(new_swarm.nodes) == 2
    assert len(new_swarm.edges) == 1

    # Find nodes by comparing IDs
    node_names = [data.name for i, data in new_swarm.nodes(data=True)]
    assert agent1.name in node_names
    assert agent2.name in node_names
