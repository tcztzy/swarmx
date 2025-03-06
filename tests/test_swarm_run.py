import pytest

from swarmx import Agent, Swarm

pytestmark = pytest.mark.anyio


@pytest.fixture
def simple_swarm():
    swarm = Swarm()
    agent1 = Agent(name="Agent1", instructions="You are Agent 1")
    agent2 = Agent(name="Agent2", instructions="You are Agent 2")

    swarm.add_node(0, agent1)
    swarm.add_node(1, agent2)
    swarm.add_edge(0, 1)

    return swarm


def test_swarm_root(simple_swarm):
    assert simple_swarm.root == 0


def test_next_node(simple_swarm):
    assert simple_swarm._next_node == 2


def test_swarm_add_edge_validation():
    swarm = Swarm()
    agent1 = Agent(name="Agent1")
    swarm.add_node(0, agent1)

    # Test with non-existent source node
    with pytest.raises(ValueError, match="Node 999 not found"):
        swarm.add_edge(999, 0)

    # Test with non-existent target node
    with pytest.raises(ValueError, match="Node 999 not found"):
        swarm.add_edge(0, 999)
