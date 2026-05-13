use super::*;

#[test]
fn harness_label_all_variants() {
    assert_eq!(Harness::ClaudeCode.label(), "Claude Code");
    assert_eq!(Harness::Codex.label(), "Codex");
    assert_eq!(Harness::OpenCode.label(), "OpenCode");
    assert_eq!(Harness::Hermes.label(), "Hermes");
    assert_eq!(Harness::OpenClaw.label(), "OpenClaw");
    assert_eq!(Harness::SwarmX.label(), "SwarmX");
}

#[test]
fn harness_icon_all_variants() {
    // Every harness maps to its corresponding AgentIcon.
    assert_eq!(Harness::ClaudeCode.icon(), AgentIcon::ClaudeCode);
    assert_eq!(Harness::Codex.icon(), AgentIcon::Codex);
    assert_eq!(Harness::OpenCode.icon(), AgentIcon::OpenCode);
    assert_eq!(Harness::Hermes.icon(), AgentIcon::Hermes);
    assert_eq!(Harness::OpenClaw.icon(), AgentIcon::OpenClaw);
    assert_eq!(Harness::SwarmX.icon(), AgentIcon::SwarmX);
}

#[test]
fn compat_claude_code_only_anthropic() {
    assert_eq!(Harness::ClaudeCode.compatible(), &[ProviderKind::Anthropic]);
}

#[test]
fn compat_codex_openai() {
    let c = Harness::Codex.compatible();
    assert!(c.contains(&ProviderKind::OpenAIResponses));
    assert!(c.contains(&ProviderKind::OpenAIChat));
}

#[test]
fn compat_opencode_three_providers() {
    let c = Harness::OpenCode.compatible();
    assert!(c.contains(&ProviderKind::Anthropic));
    assert!(c.contains(&ProviderKind::OpenAIChat));
    assert!(c.contains(&ProviderKind::Ollama));
}

#[test]
fn passthrough_env_standard_keys() {
    let keys = Harness::ClaudeCode.passthrough_env();
    assert!(keys.contains(&"PATH"));
    assert!(keys.contains(&"HOME"));
    assert!(keys.contains(&"USER"));
}

#[test]
fn config_env_per_harness() {
    use std::path::Path;
    let dir = Path::new("/tmp/test-config");
    let vars = Harness::ClaudeCode.config_env(dir);
    assert_eq!(vars[0].0, "CLAUDE_CONFIG_DIR");
    assert_eq!(vars[0].1, std::ffi::OsString::from("/tmp/test-config"));

    let vars = Harness::Codex.config_env(dir);
    assert_eq!(vars[0].0, "CODEX_CONFIG_DIR");

    let vars = Harness::OpenCode.config_env(dir);
    assert_eq!(vars[0].0, "OPENCODE_CONFIG_DIR");
}

#[test]
fn base_command_program_names() {
    let cmd = Harness::ClaudeCode.base_command();
    let prog = cmd.get_program().to_string_lossy().into_owned();
    assert_eq!(prog, "bun");

    let cmd = Harness::OpenCode.base_command();
    let prog = cmd.get_program().to_string_lossy().into_owned();
    assert_eq!(prog, "opencode");
}

#[test]
fn spawn_command_env_clear_and_passthrough() {
    // Construct a minimal instance + provider for testing.
    let provider = ModelProvider {
        id: "p1".into(),
        label: "Test".into(),
        kind: ProviderKind::Anthropic,
        base_url: None,
        api_key_ref: None,
        default_model: "claude-test".into(),
        available_models: vec!["claude-test".into()],
    };
    let inst = AgentInstance {
        id: "i1".into(),
        label: "Test".into(),
        harness: Harness::ClaudeCode,
        provider_id: "p1".into(),
        model: "claude-test".into(),
        instructions: None,
        config_dir: std::path::PathBuf::from("/tmp/claude-config"),
        icon_override: None,
        default_cwd: None,
    };

    let cmd = Harness::ClaudeCode.spawn_command(&inst, &provider, std::path::Path::new("/tmp"));

    // Verify cwd is set.
    // Note: Command doesn't expose current_dir after construction in std,
    // so we test the env composition instead.
    let envs: Vec<_> = cmd.get_envs().collect();

    // ANTHROPIC_MODEL should be set.
    assert!(envs.iter().any(|(k, v)| {
        *k == std::ffi::OsStr::new("ANTHROPIC_MODEL")
            && v.as_ref()
                .is_some_and(|val| val.to_string_lossy() == "claude-test")
    }));

    // CLAUDE_CONFIG_DIR should be set.
    assert!(envs.iter().any(|(k, v)| {
        *k == std::ffi::OsStr::new("CLAUDE_CONFIG_DIR")
            && v.as_ref()
                .is_some_and(|val| val.to_string_lossy() == "/tmp/claude-config")
    }));

    // PATH should be passed through if set in the parent.
    if std::env::var("PATH").is_ok() {
        assert!(envs.iter().any(|(k, _)| *k == std::ffi::OsStr::new("PATH")));
    }
}

#[test]
fn spawn_command_no_api_key_when_ref_unset() {
    let provider = ModelProvider {
        id: "p1".into(),
        label: "Test".into(),
        kind: ProviderKind::Anthropic,
        base_url: None,
        api_key_ref: None,
        default_model: "claude-test".into(),
        available_models: vec!["claude-test".into()],
    };
    let inst = AgentInstance {
        id: "i1".into(),
        label: "Test".into(),
        harness: Harness::ClaudeCode,
        provider_id: "p1".into(),
        model: "claude-test".into(),
        instructions: None,
        config_dir: std::path::PathBuf::from("/tmp/claude-config"),
        icon_override: None,
        default_cwd: None,
    };

    let cmd = Harness::ClaudeCode.spawn_command(&inst, &provider, std::path::Path::new("/tmp"));
    let envs: Vec<_> = cmd.get_envs().collect();

    // ANTHROPIC_API_KEY should NOT be set when api_key_ref is None.
    assert!(
        !envs
            .iter()
            .any(|(k, _)| *k == std::ffi::OsStr::new("ANTHROPIC_API_KEY"))
    );
}

#[test]
fn serde_round_trip_all_variants() {
    for h in &[
        Harness::SwarmX,
        Harness::ClaudeCode,
        Harness::Codex,
        Harness::OpenCode,
        Harness::Hermes,
        Harness::OpenClaw,
    ] {
        let json = serde_json::to_string(h).unwrap();
        let back: Harness = serde_json::from_str(&json).unwrap();
        assert_eq!(*h, back);
    }
}
