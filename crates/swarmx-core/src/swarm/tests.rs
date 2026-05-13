use crate::agent::Agent;
use crate::edge::Edge;
use crate::swarm::{Swarm, SwarmNode};

#[test]
fn test_swarm_creation() {
    let swarm = Swarm::new("test_swarm", "agent_a");
    assert_eq!(swarm.name, "test_swarm");
    assert_eq!(swarm.root, "agent_a");
    assert!(swarm.nodes.is_empty());
}

#[test]
fn test_swarm_with_nodes() {
    let swarm = Swarm::new("test_swarm", "agent_a")
        .with_node(SwarmNode::Agent(Agent::new("agent_a")))
        .with_node(SwarmNode::Agent(Agent::new("agent_b")))
        .with_edge(Edge::new("agent_a", "agent_b"));

    assert_eq!(swarm.nodes.len(), 2);
    assert_eq!(swarm.edges.len(), 1);
}

#[test]
fn test_rebuild_graphs_valid() {
    let swarm = Swarm::new("test_swarm", "agent_a")
        .with_node(SwarmNode::Agent(Agent::new("agent_a")))
        .with_node(SwarmNode::Agent(Agent::new("agent_b")))
        .with_edge(Edge::new("agent_a", "agent_b"));

    let result = swarm.rebuild_graphs();
    assert!(result.is_ok());
}

#[test]
fn test_rebuild_graphs_cycle_detected() {
    let swarm = Swarm::new("test_swarm", "agent_a")
        .with_node(SwarmNode::Agent(Agent::new("agent_a")))
        .with_node(SwarmNode::Agent(Agent::new("agent_b")))
        .with_edge(Edge::new("agent_a", "agent_b"))
        .with_edge(Edge::new("agent_b", "agent_a"));

    let result = swarm.rebuild_graphs();
    assert!(result.is_err());
}

#[test]
fn test_rebuild_graphs_unknown_source() {
    let swarm = Swarm::new("test_swarm", "agent_a")
        .with_node(SwarmNode::Agent(Agent::new("agent_a")))
        .with_edge(Edge::new("unknown", "agent_a"));

    let result = swarm.rebuild_graphs();
    assert!(result.is_err());
}

#[tokio::test]
#[ignore = "requires OPENAI_API_KEY and Python swarmx installed"]
async fn test_swarm_execution_simple() {
    let swarm =
        Swarm::new("test_swarm", "agent_a").with_node(SwarmNode::Agent(Agent::new("agent_a")));

    let result = swarm
        .execute(
            serde_json::json!({"messages": [{"role": "user", "content": "hi"}]}),
            None,
        )
        .await;
    assert!(result.is_ok(), "Execution failed: {:?}", result.err());
}
