use crate::agent::Agent;
use crate::hook::Hook;

#[test]
fn test_agent_creation() {
    let agent = Agent::new("test_agent")
        .with_model("gpt-4")
        .with_instructions("Be helpful.");
    assert_eq!(agent.name(), "test_agent");
    assert_eq!(agent.model, Some("gpt-4".to_string()));
    assert_eq!(agent.instructions, Some("Be helpful.".to_string()));
}

#[test]
fn test_agent_hooks() {
    let agent = Agent::new("hooked_agent").with_hook(Hook {
        on_start: Some("start_tool".to_string()),
        on_end: Some("end_tool".to_string()),
        ..Default::default()
    });
    assert_eq!(agent.hooks().len(), 1);
}

#[test]
fn test_agent_serde() {
    let agent = Agent::new("serde_test").with_model("gpt-4");
    let json = serde_json::to_string(&agent).unwrap();
    assert!(json.contains("serde_test"));
    let deserialized: Agent = serde_json::from_str(&json).unwrap();
    assert_eq!(deserialized.name(), "serde_test");
}
