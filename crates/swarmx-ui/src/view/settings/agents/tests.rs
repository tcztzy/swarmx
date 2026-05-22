//! Handler-level tests for agent CRUD.
//!
//! Build `App` manually via `bare_app()` helper to skip filesystem seeding.

use crate::app::{App, Message};
use crate::harness::Harness;
use crate::instance::{
    AgentInstance, ModelProvider, ProviderKind, default_models_for_kind, uuid_v7,
};
use crate::theme::ThemePreference;
use crate::tokens::{Density, DesignTokens};

fn seed_provider(kind: ProviderKind, id: &str) -> ModelProvider {
    ModelProvider {
        id: id.into(),
        label: format!("P-{id}"),
        kind,
        base_url: None,
        api_key_ref: None,
        default_model: default_models_for_kind(kind)
            .first()
            .cloned()
            .unwrap_or_default(),
        available_models: default_models_for_kind(kind),
    }
}

fn seed_instance(provider: &ModelProvider) -> AgentInstance {
    AgentInstance {
        id: uuid_v7(),
        label: "I".into(),
        harness: Harness::ClaudeCode,
        provider_id: provider.id.clone(),
        model: provider.default_model.clone(),
        instructions: None,
        config_dir: std::path::PathBuf::from("/tmp/x"),
        icon_override: None,
        default_cwd: None,
    }
}

fn bare_app(providers: Vec<ModelProvider>, instances: Vec<AgentInstance>) -> App {
    let provider_labels = providers.iter().map(|p| p.label.clone()).collect();
    let density = Density::Comfortable;
    App {
        theme: ThemePreference::Dark.shadcn_theme(),
        theme_preference: ThemePreference::Dark,
        density,
        tokens: DesignTokens::for_density(density),
        app_settings: crate::persistence::AppSettings::default(),
        providers,
        instances,
        sessions: Vec::new(),
        active_session: None,
        session_grouping: crate::persistence::SessionGrouping::Date,
        session_sort_by: crate::persistence::SessionSortBy::Recency,
        project_filter: None,

        sidebar_search: String::new(),
        group_collapsed: std::collections::HashSet::new(),
        renaming_session: None,
        renaming_remote_session: None,
        rename_buffer: String::new(),
        surface: crate::app::Surface::Welcome,
        input: String::new(),
        loading: false,
        error: None,
        env_checks: Vec::new(),
        env_installing: None,
        agent_statuses: Vec::new(),
        remote_sessions: Vec::new(),
        remote_title_overrides: std::collections::HashMap::new(),
        pending_launch_session_id: None,
        md_cache: std::collections::HashMap::new(),
        session_cache: crate::swr::SWRCache::new(60),
        remote_title_cache: crate::swr::SWRCache::new(60),
        message_cache: crate::swr::SWRCache::new(300),
        thinking_expanded: std::collections::HashSet::new(),
        tool_expanded: std::collections::HashSet::new(),
        sidebar_collapsed: false,
        sidebar_width: 240.0,
        sidebar_dragging: false,
        startup_loading: false,
        spinner_progress: 0.0,
        home_dir: String::new(),
        cancel_requested: false,
        pending_delete_provider: None,
        pending_delete_instance: None,
        provider_labels,
        persistence_enabled: false,
    }
}

#[test]
fn handler_fixture_app_does_not_persist_to_disk() {
    let app = bare_app(vec![], vec![]);
    assert!(!app.persistence_enabled);
}

#[test]
fn add_provider_appends_with_defaults() {
    let mut app = bare_app(vec![], vec![]);
    let _ = app.update(Message::AddProvider);
    assert_eq!(app.providers.len(), 1);
    assert_eq!(app.providers[0].kind, ProviderKind::Custom);
    assert_eq!(
        app.providers[0].available_models,
        default_models_for_kind(ProviderKind::Custom)
    );
}

#[test]
fn update_provider_label_persists() {
    let mut app = bare_app(vec![seed_provider(ProviderKind::Anthropic, "p1")], vec![]);
    let _ = app.update(Message::UpdateProviderLabel(0, "New".into()));
    assert_eq!(app.providers[0].label, "New");
}

#[test]
fn update_provider_kind_reseeds_models() {
    let mut app = bare_app(vec![seed_provider(ProviderKind::Custom, "p1")], vec![]);
    let _ = app.update(Message::UpdateProviderKind(0, ProviderKind::Anthropic));
    assert_eq!(app.providers[0].kind, ProviderKind::Anthropic);
    assert_eq!(
        app.providers[0].available_models,
        default_models_for_kind(ProviderKind::Anthropic)
    );
    assert!(
        app.providers[0]
            .available_models
            .contains(&app.providers[0].default_model)
    );
}

#[test]
fn set_api_key_empty_clears_ref() {
    let mut p = seed_provider(ProviderKind::Anthropic, "p1");
    p.api_key_ref = Some("p1".into());
    let mut app = bare_app(vec![p], vec![]);
    let _ = app.update(Message::SetProviderApiKey(0, "".into()));
    assert_eq!(app.providers[0].api_key_ref, None);
}

#[test]
fn delete_provider_two_step() {
    let mut app = bare_app(vec![seed_provider(ProviderKind::Anthropic, "p1")], vec![]);
    let _ = app.update(Message::DeleteProvider(0));
    assert_eq!(app.pending_delete_provider, Some(0));
    assert_eq!(app.providers.len(), 1);
    let _ = app.update(Message::ConfirmDeleteProvider(0));
    assert_eq!(app.pending_delete_provider, None);
    assert_eq!(app.providers.len(), 0);
}

#[test]
fn cancel_delete_provider_clears_pending() {
    let mut app = bare_app(vec![seed_provider(ProviderKind::Anthropic, "p1")], vec![]);
    let _ = app.update(Message::DeleteProvider(0));
    let _ = app.update(Message::CancelDeleteProvider);
    assert_eq!(app.pending_delete_provider, None);
    assert_eq!(app.providers.len(), 1);
}

#[test]
fn add_instance_requires_provider() {
    let mut app = bare_app(vec![], vec![]);
    let _ = app.update(Message::AddInstance);
    assert_eq!(app.instances.len(), 0);
}

#[test]
fn add_instance_uses_first_provider_defaults() {
    let p = seed_provider(ProviderKind::Anthropic, "p1");
    let mut app = bare_app(vec![p.clone()], vec![]);
    let _ = app.update(Message::AddInstance);
    assert_eq!(app.instances.len(), 1);
    assert_eq!(app.instances[0].provider_id, "p1");
    assert_eq!(app.instances[0].model, p.default_model);
    assert_eq!(app.instances[0].harness, Harness::ClaudeCode);
}

#[test]
fn update_instance_provider_resets_model_if_incompatible() {
    let p_openai = seed_provider(ProviderKind::OpenAIResponses, "openai");
    let p_anthropic = seed_provider(ProviderKind::Anthropic, "anthropic");
    let mut inst = seed_instance(&p_openai);
    inst.model = "gpt-4o".into();
    let mut app = bare_app(vec![p_openai, p_anthropic.clone()], vec![inst]);
    let _ = app.update(Message::UpdateInstanceProviderId(0, "anthropic".into()));
    assert_eq!(app.instances[0].provider_id, "anthropic");
    assert_eq!(app.instances[0].model, p_anthropic.default_model);
}

#[test]
fn delete_instance_two_step() {
    let p = seed_provider(ProviderKind::Anthropic, "p1");
    let inst = seed_instance(&p);
    let mut app = bare_app(vec![p], vec![inst]);
    let _ = app.update(Message::DeleteInstance(0));
    assert_eq!(app.pending_delete_instance, Some(0));
    assert_eq!(app.instances.len(), 1);
    let _ = app.update(Message::ConfirmDeleteInstance(0));
    assert_eq!(app.pending_delete_instance, None);
    assert_eq!(app.instances.len(), 0);
}
