use crate::edge::Edge;

#[test]
fn test_edge_creation() {
    let edge = Edge::new("agent_a", "agent_b");
    assert_eq!(edge.source, "agent_a");
    assert_eq!(edge.target, "agent_b");
    assert!(edge.condition.is_none());
}

#[test]
fn test_edge_with_condition() {
    let edge = Edge::new("agent_a", "agent_b").with_condition("score > 0.5");
    assert_eq!(edge.condition, Some("score > 0.5".to_string()));
}

#[test]
fn test_edge_serde() {
    let edge = Edge::new("a", "b").with_condition("true");
    let json = serde_json::to_string(&edge).unwrap();
    assert!(json.contains("a"));
    assert!(json.contains("true"));
    let deserialized: Edge = serde_json::from_str(&json).unwrap();
    assert_eq!(deserialized, edge);
}
