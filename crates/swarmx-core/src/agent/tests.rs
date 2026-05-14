use super::{Agent, AgentBackend, AgentProcessOptions};
use crate::hook::Hook;
use std::collections::BTreeMap;

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
    assert!(!json.contains("process"));
    let deserialized: Agent = serde_json::from_str(&json).unwrap();
    assert_eq!(deserialized.name(), "serde_test");
    assert!(deserialized.process.is_default());
}

#[test]
fn test_agent_process_options_round_trip() {
    let mut env = BTreeMap::new();
    env.insert(
        "CODEX_CONFIG_DIR".to_string(),
        "/tmp/codex-config".to_string(),
    );
    env.insert("PATH".to_string(), "/usr/bin".to_string());

    let agent = Agent::new("codex")
        .with_backend(AgentBackend::Custom {
            program: "bun".to_string(),
            args: vec![
                "x".to_string(),
                "@agentclientprotocol/codex-acp".to_string(),
            ],
        })
        .with_process(AgentProcessOptions {
            current_dir: Some("/tmp/work".into()),
            env: env.clone(),
            clear_env: true,
        });

    let json = serde_json::to_string(&agent).unwrap();
    assert!(json.contains("process"));
    assert!(json.contains("currentDir"));
    let deserialized: Agent = serde_json::from_str(&json).unwrap();
    assert_eq!(deserialized.process.current_dir, Some("/tmp/work".into()));
    assert_eq!(deserialized.process.env, env);
    assert!(deserialized.process.clear_env);
}

#[test]
fn test_agent_acp_command_preserves_process_options() {
    let mut env = BTreeMap::new();
    env.insert(
        "CODEX_CONFIG_DIR".to_string(),
        "/tmp/codex-config".to_string(),
    );
    env.insert("PATH".to_string(), "/usr/bin".to_string());

    let agent = Agent::new("codex")
        .with_backend(AgentBackend::Custom {
            program: "bun".to_string(),
            args: vec![
                "x".to_string(),
                "@agentclientprotocol/codex-acp".to_string(),
            ],
        })
        .with_process(AgentProcessOptions {
            current_dir: Some("/tmp/work".into()),
            env: env.clone(),
            clear_env: true,
        });

    let command = agent.acp_command();
    assert_eq!(command.program, "bun");
    assert_eq!(
        command.args,
        vec![
            "x".to_string(),
            "@agentclientprotocol/codex-acp".to_string()
        ]
    );
    assert_eq!(
        command.process.current_dir.as_deref(),
        Some(std::path::Path::new("/tmp/work"))
    );
    assert_eq!(command.process.env, env);
    assert!(command.process.clear_env);
}
