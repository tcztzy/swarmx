use iced::{Element, Task};
use iced_shadcn::Theme;
use swarmx_core::{Agent, AgentBackend, Swarm, swarm::SwarmNode};

use lucide_icons;
use std::collections::HashSet;

use crate::data::{LoadSessionMeta, RemoteAgentSessions, RemoteSessionSource, Session};
use crate::environment::{
    self, AgentRuntime, AgentRuntimeStatus, DepStatus, RemoteAgentRef, RuntimeDep,
};
use crate::harness::Harness;
use crate::instance::{self, AgentInstance, ModelProvider};
use crate::persistence::{
    AppSettings, SessionGrouping, SessionSortBy, load_settings, save_settings,
};
use crate::swr::SWRCache;
use crate::theme::ThemePreference;
use crate::tokens::{Density, DesignTokens};
use crate::view::chat_message::{ChatMessage, MessageKind};
use crate::view::settings;

/// Run the SwarmX desktop application.
pub fn run() -> iced::Result {
    let icon = load_icon();
    iced::application(App::new, App::update, App::view)
        .font(lucide_icons::LUCIDE_FONT_BYTES)
        .window(iced::window::Settings {
            icon,
            ..Default::default()
        })
        .theme(|app: &App| app.theme_preference.iced_theme())
        .subscription(App::subscription)
        .run()
}

fn load_icon() -> Option<iced::window::Icon> {
    iced::window::icon::from_file_data(include_bytes!("../resources/icon.png"), None).ok()
}

// ── App ──────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Surface {
    Welcome,
    Conversation,
    Settings { tab: settings::Tab },
}

pub struct App {
    pub theme: Theme,
    pub theme_preference: ThemePreference,
    pub density: Density,
    pub tokens: DesignTokens,
    pub app_settings: AppSettings,
    // Agent instance + provider catalog (spec §4)
    pub providers: Vec<ModelProvider>,
    pub instances: Vec<AgentInstance>,
    // Session management
    pub sessions: Vec<Session>,
    pub active_session: Option<usize>,
    pub session_grouping: SessionGrouping,
    pub session_sort_by: SessionSortBy,
    pub project_filter: Option<String>,

    pub sidebar_search: String,
    pub group_collapsed: HashSet<String>,
    pub renaming_session: Option<usize>,
    pub rename_buffer: String,
    // Current surface
    pub surface: Surface,
    // Chat (active session)
    pub input: String,
    pub loading: bool,
    pub error: Option<String>,
    // Environment
    pub env_checks: Vec<DepStatus>,
    pub env_installing: Option<RuntimeDep>,
    // ACP agent detection
    pub agent_statuses: Vec<AgentRuntimeStatus>,
    /// Sessions fetched from ACP agents via session/list.
    pub remote_sessions: Vec<RemoteAgentSessions>,
    pub pending_launch_session_id: Option<String>,
    // Markdown render cache: session_id -> parsed Content per agent message
    pub md_cache: std::collections::HashMap<String, Vec<iced::widget::markdown::Content>>,
    // SWR cache for session list (60s TTL)
    pub session_cache: SWRCache<String, Vec<Session>>,
    /// SWR cache for session message content (300s TTL). Key = session_id.
    pub message_cache: SWRCache<String, Vec<ChatMessage>>,
    /// Indices of expanded thinking sections in the active session.
    pub thinking_expanded: HashSet<usize>,
    /// Indices of expanded tool call sections in the active session.
    pub tool_expanded: HashSet<usize>,
    // Sidebar
    pub sidebar_collapsed: bool,
    pub sidebar_width: f32,
    pub sidebar_dragging: bool,
    /// True until first AgentSessionsResult — prevents UI flicker.
    pub startup_loading: bool,
    /// Spinner rotation progress (0.0..1.0).
    pub spinner_progress: f32,
    /// User home directory path.
    pub home_dir: String,
    /// Set when user clicks Stop — in-flight Response will be discarded.
    pub cancel_requested: bool,
    /// Two-click confirmation state for provider deletion.
    pub pending_delete_provider: Option<usize>,
    /// Two-click confirmation state for instance deletion.
    pub pending_delete_instance: Option<usize>,
    /// Cached provider labels for select widget (must outlive view).
    pub provider_labels: Vec<String>,
    /// Handler tests construct `App` directly; keep those fixtures in memory.
    pub persistence_enabled: bool,
}

#[derive(Debug, Clone)]
pub enum Message {
    // Sidebar
    NewSession,
    CreateSession(String),
    SelectSession(usize),
    DeleteSession(usize),
    TogglePinSession(usize),
    StartRenameSession(usize),
    SessionRenameChanged(String),
    CommitSessionRename,
    CancelSessionRename,
    ArchiveSession(usize),
    ToggleUnreadSession(usize),
    OpenWorkingDirectory(String),
    OpenSessionInNewWindow(String),
    FilterChanged(FilterChange),

    ToggleGroup(String),
    SearchSidebar(String),
    ToggleSettings,
    // Chat
    InputChanged(String),
    SendMessage,
    Response(Result<Vec<serde_json::Value>, String>),
    UseSuggestion(String),
    LinkClicked(String),
    CopyToClipboard(String),
    SetThinkingOpen(usize, bool),
    SetToolOpen(usize, bool),
    StopGeneration,
    ModelChanged(usize, String),
    // Agents
    AddProvider,
    DeleteProvider(usize),
    ConfirmDeleteProvider(usize),
    CancelDeleteProvider,
    UpdateProviderLabel(usize, String),
    UpdateProviderKind(usize, crate::instance::ProviderKind),
    UpdateProviderBaseUrl(usize, String),
    SetProviderApiKey(usize, String),
    UpdateProviderDefaultModel(usize, String),
    AddInstance,
    DeleteInstance(usize),
    ConfirmDeleteInstance(usize),
    CancelDeleteInstance,
    UpdateInstanceLabel(usize, String),
    UpdateInstanceHarness(usize, Harness),
    UpdateInstanceProviderId(usize, String),
    UpdateInstanceModel(usize, String),
    UpdateInstanceInstructions(usize, String),
    UpdateInstanceDefaultCwd(usize, String),
    // Settings
    GoToSettingsTab(settings::Tab),
    SetTheme(ThemePreference),
    SetDensity(Density),
    // Environment
    InstallTool(RuntimeDep),
    InstallResult(RuntimeDep, Result<(), String>),
    ToggleSidebar,
    StartSidebarDrag,
    SidebarDrag(f32),
    EndSidebarDrag,
    // Animation
    Tick(iced::time::Instant),
    WindowFocused,
    // Remote ACP sessions
    RefreshAgentSessions,
    AgentSessionsResult(Result<Vec<RemoteAgentSessions>, String>),
    LoadRemoteSession(RemoteAgentRef, RemoteSessionSource, String, String),
    RemoteSessionLoaded(
        Result<(LoadSessionMeta, Vec<ChatMessage>), String>,
        RemoteAgentRef,
        RemoteSessionSource,
        String,
    ),
}

#[derive(Debug, Clone)]
pub enum FilterChange {
    Grouping(SessionGrouping),
    Sort(SessionSortBy),
    Project(Option<String>),
}

/// Build a swarmx_core::Agent from an instance + provider for ACP transport.
fn build_agent(inst: &AgentInstance, provider: &ModelProvider) -> Agent {
    match inst.harness {
        Harness::SwarmX => Agent::new("swarmx")
            .with_backend(AgentBackend::SwarmX)
            .with_instructions(
                inst.instructions
                    .as_deref()
                    .unwrap_or("You are a helpful AI assistant. Be concise and friendly."),
            ),
        _ => {
            let cwd = inst
                .default_cwd
                .as_deref()
                .unwrap_or_else(|| std::path::Path::new("."));
            let cmd = inst.harness.spawn_command(inst, provider, cwd);
            let program = cmd.get_program().to_string_lossy().into_owned();
            let args: Vec<String> = cmd
                .get_args()
                .map(|a| a.to_string_lossy().into_owned())
                .collect();
            Agent::new(&inst.id).with_backend(AgentBackend::Custom { program, args })
        }
    }
}

fn build_runtime_agent(runtime: AgentRuntime) -> Agent {
    let (program, args) = runtime.command();
    Agent::new(runtime.id()).with_backend(AgentBackend::Custom {
        program: program.to_string(),
        args: args.into_iter().map(ToString::to_string).collect(),
    })
}

fn runtime_for_harness(harness: Harness) -> Option<AgentRuntime> {
    match harness {
        Harness::ClaudeCode => Some(AgentRuntime::ClaudeAgentAcp),
        Harness::Codex => Some(AgentRuntime::CodexAcp),
        Harness::OpenCode => Some(AgentRuntime::OpenCode),
        Harness::Hermes => Some(AgentRuntime::Hermes),
        Harness::OpenClaw => Some(AgentRuntime::OpenClaw),
        Harness::SwarmX => None,
    }
}

#[derive(Clone, Debug, Default)]
struct LaunchOptions {
    session_id: Option<String>,
}

impl LaunchOptions {
    fn from_env() -> Self {
        let mut args = std::env::args().skip(1);
        let mut session_id = None;

        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--session" => {
                    session_id = args.next();
                }
                _ if arg.starts_with("swarmx://session/") => {
                    session_id = arg
                        .trim_start_matches("swarmx://session/")
                        .split(['?', '#'])
                        .next()
                        .filter(|id| !id.is_empty())
                        .map(ToString::to_string);
                }
                _ => {}
            }
        }

        Self { session_id }
    }
}

impl App {
    fn new() -> (Self, Task<Message>) {
        let mut app = Self::default();
        app.apply_launch_options(LaunchOptions::from_env());
        (app, Task::done(Message::RefreshAgentSessions))
    }
}

impl Default for App {
    fn default() -> Self {
        let app_settings = load_settings();
        let theme_preference = app_settings.theme;
        let density = app_settings.density;
        let tokens = DesignTokens::for_density(density);
        let home = dirs_next_home();
        let env_checks: Vec<_> = RuntimeDep::all()
            .iter()
            .map(|d| environment::check_one_sync(*d))
            .collect();
        let agent_statuses: Vec<_> = AgentRuntime::all()
            .iter()
            .map(|a| environment::check_agent_sync(*a))
            .collect();
        let providers = instance::load_providers();
        let providers_labels: Vec<String> = providers.iter().map(|p| p.label.clone()).collect();
        let instances = instance::load_instances();
        let sessions = load_sessions(&instances);
        let mut md_cache = std::collections::HashMap::new();
        for s in &sessions {
            let contents: Vec<_> = s
                .messages
                .iter()
                .filter(|m| !m.is_user)
                .map(|m| iced::widget::markdown::Content::parse(&m.content))
                .collect();
            md_cache.insert(s.id.clone(), contents);
        }
        let thinking_expanded = HashSet::new();
        let tool_expanded = HashSet::new();
        let session_grouping = app_settings.session_grouping;
        let session_sort_by = app_settings.session_sort_by;
        let project_filter = app_settings.project_filter.clone();
        let mut session_cache = SWRCache::new(60);
        session_cache.set("local".to_string(), sessions.clone());
        Self {
            theme: theme_preference.shadcn_theme(),
            theme_preference,
            density,
            tokens,
            app_settings,
            providers,
            instances,
            sessions,
            active_session: None,
            session_grouping,
            session_sort_by,
            project_filter,

            sidebar_search: String::new(),
            group_collapsed: HashSet::new(),
            renaming_session: None,
            rename_buffer: String::new(),
            surface: Surface::Welcome,
            input: String::new(),
            loading: false,
            error: None,
            env_checks,
            env_installing: None,
            agent_statuses,
            remote_sessions: Vec::new(),
            pending_launch_session_id: None,
            md_cache,
            session_cache,
            message_cache: SWRCache::new(300),
            thinking_expanded,
            tool_expanded,
            sidebar_collapsed: false,
            sidebar_width: 240.0,
            sidebar_dragging: false,
            startup_loading: true,
            spinner_progress: 0.0,
            home_dir: home,
            cancel_requested: false,
            pending_delete_provider: None,
            pending_delete_instance: None,
            provider_labels: providers_labels,
            persistence_enabled: true,
        }
    }
}

impl App {
    fn save_settings(&self) {
        if self.persistence_enabled {
            let _ = save_settings(&self.app_settings);
        }
    }

    fn save_providers(&self) {
        if self.persistence_enabled {
            let _ = instance::save_providers(&self.providers);
        }
    }

    fn save_instances(&self) {
        if self.persistence_enabled {
            let _ = instance::save_instances(&self.instances);
        }
    }
}

fn dirs_next_home() -> String {
    std::env::var("HOME")
        .or_else(|_| std::env::var("USERPROFILE"))
        .unwrap_or_else(|_| ".".into())
}

fn sessions_dir() -> std::path::PathBuf {
    let home = dirs_next_home();
    std::path::PathBuf::from(home)
        .join(".swarmx")
        .join("sessions")
}

fn session_path(session_id: &str) -> std::path::PathBuf {
    sessions_dir().join(format!("{}.json", session_id))
}

fn load_sessions(instances: &[AgentInstance]) -> Vec<Session> {
    let dir = sessions_dir();
    let _ = std::fs::create_dir_all(&dir);
    let mut sessions: Vec<Session> = Vec::new();
    let valid_instance_ids: std::collections::HashSet<String> =
        instances.iter().map(|i| i.id.clone()).collect();
    let default_instance_id = instances.first().map(|i| i.id.clone());
    if let Ok(entries) = std::fs::read_dir(&dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().is_some_and(|e| e == "json") {
                if let Ok(data) = std::fs::read_to_string(&path) {
                    if let Ok(mut session) = serde_json::from_str::<Session>(&data) {
                        if session.agent_runtime.is_none()
                            && (session.agent_instance_id.is_empty()
                                || !valid_instance_ids.contains(&session.agent_instance_id))
                        {
                            if let Some(ref id) = default_instance_id {
                                session.agent_instance_id = id.clone();
                            } else {
                                continue;
                            }
                        }
                        sessions.push(session);
                    }
                }
            }
        }
    }
    sessions.sort_by(|a, b| b.updated_at.cmp(&a.updated_at));
    sessions
}

fn save_session(session: &mut Session) {
    use std::time::{SystemTime, UNIX_EPOCH};
    let ts = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    session.updated_at = ts.to_string();
    write_session(session);
}

fn write_session(session: &Session) {
    let dir = sessions_dir();
    let _ = std::fs::create_dir_all(&dir);
    let path = session_path(&session.id);
    let _ = std::fs::write(
        &path,
        serde_json::to_string_pretty(session).unwrap_or_default(),
    );
}

fn delete_session_file(session_id: &str) {
    let path = session_path(session_id);
    let _ = std::fs::remove_file(&path);
}

fn session_title_for_edit(session: &Session) -> String {
    session
        .title
        .clone()
        .or_else(|| session.messages.first().map(|m| m.content.clone()))
        .unwrap_or_else(|| "New session".to_string())
}

fn open_working_directory(path: &str) -> Result<(), String> {
    let mut command = if cfg!(target_os = "macos") {
        let mut command = std::process::Command::new("open");
        command.arg(path);
        command
    } else if cfg!(target_os = "windows") {
        let mut command = std::process::Command::new("explorer");
        command.arg(path);
        command
    } else {
        let mut command = std::process::Command::new("xdg-open");
        command.arg(path);
        command
    };

    command
        .spawn()
        .map(|_| ())
        .map_err(|e| format!("Failed to open working directory: {e}"))
}

fn open_session_in_new_window(session_id: &str) -> Result<(), String> {
    let exe = std::env::current_exe()
        .map_err(|e| format!("Failed to resolve current executable: {e}"))?;
    std::process::Command::new(exe)
        .arg("--session")
        .arg(session_id)
        .spawn()
        .map(|_| ())
        .map_err(|e| format!("Failed to open session in new window: {e}"))
}

/// Update SWR cache with current sessions.
fn update_session_cache(cache: &mut SWRCache<String, Vec<Session>>, sessions: &[Session]) {
    let mut sorted = sessions.to_vec();
    sorted.sort_by(|a, b| b.updated_at.cmp(&a.updated_at));
    cache.set("local".to_string(), sorted);
}

/// Convert an ACP session message (serde_json::Value) to a ChatMessage.
fn convert_acp_message(v: serde_json::Value) -> Option<ChatMessage> {
    let role = v.get("role")?.as_str()?;
    let content = v
        .get("content")
        .and_then(|c| c.as_str())
        .unwrap_or("")
        .to_string();
    let kind_str = v.get("kind").and_then(|k| k.as_str()).unwrap_or("message");
    let kind = match kind_str {
        "thinking" => MessageKind::Thinking,
        "tool_call" => MessageKind::ToolCall,
        "tool_result" => MessageKind::ToolResult,
        _ => MessageKind::Message,
    };
    let is_user = role == "user";
    let tool_name = v
        .get("tool_name")
        .and_then(|t| t.as_str())
        .map(String::from);
    let tool_result = v
        .get("tool_result")
        .and_then(|t| t.as_str())
        .map(String::from);
    Some(ChatMessage {
        is_user,
        content,
        kind,
        tool_name,
        tool_result,
        duration_ms: None,
    })
}

fn sqlite_quote(value: &str) -> String {
    value.replace('\'', "''")
}

fn hermes_state_db() -> std::path::PathBuf {
    std::path::PathBuf::from(dirs_next_home())
        .join(".hermes")
        .join("state.db")
}

fn value_f64(value: &serde_json::Value, key: &str) -> Option<f64> {
    value.get(key).and_then(|v| v.as_f64())
}

fn first_nonempty_string<'a>(value: &'a serde_json::Value, keys: &[&str]) -> Option<&'a str> {
    keys.iter().find_map(|key| {
        value
            .get(*key)
            .and_then(|v| v.as_str())
            .filter(|s| !s.trim().is_empty())
    })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum HermesPathUse {
    ExactCwd,
    MentionedPath,
}

fn hermes_unknown_cwd() -> std::path::PathBuf {
    std::path::PathBuf::new()
}

fn hermes_session_cwd(row: &serde_json::Value) -> std::path::PathBuf {
    row.get("model_config")
        .and_then(|v| v.as_str())
        .and_then(hermes_model_config_cwd)
        .or_else(|| {
            row.get("tool_calls")
                .and_then(|v| v.as_str())
                .and_then(hermes_tool_calls_cwd)
        })
        .unwrap_or_else(hermes_unknown_cwd)
}

fn hermes_model_config_cwd(model_config: &str) -> Option<std::path::PathBuf> {
    let value: serde_json::Value = serde_json::from_str(model_config).ok()?;
    first_nonempty_string(&value, &["cwd", "working_dir", "workdir"])
        .and_then(|cwd| normalize_hermes_path(cwd, HermesPathUse::ExactCwd))
}

fn hermes_tool_calls_cwd(tool_calls_json: &str) -> Option<std::path::PathBuf> {
    let groups: Vec<serde_json::Value> = serde_json::from_str(tool_calls_json).ok()?;
    let mut candidates = Vec::new();

    for group in groups {
        if let Some(group_json) = group.as_str() {
            if let Ok(value) = serde_json::from_str::<serde_json::Value>(group_json) {
                collect_hermes_path_candidates(&value, &mut candidates);
            } else {
                collect_hermes_text_path_candidates(group_json, 20, &mut candidates);
            }
        } else {
            collect_hermes_path_candidates(&group, &mut candidates);
        }
    }

    best_hermes_cwd_candidate(candidates)
}

fn collect_hermes_path_candidates(
    value: &serde_json::Value,
    candidates: &mut Vec<(i32, std::path::PathBuf)>,
) {
    match value {
        serde_json::Value::Object(map) => {
            for (key, value) in map {
                let key = key.to_ascii_lowercase();
                if let Some(text) = value.as_str() {
                    if matches!(key.as_str(), "cwd" | "workdir" | "working_dir") {
                        push_hermes_path_candidate(candidates, text, 90, HermesPathUse::ExactCwd);
                    } else if key.contains("path") || key.ends_with("_dir") {
                        push_hermes_path_candidate(
                            candidates,
                            text,
                            50,
                            HermesPathUse::MentionedPath,
                        );
                    }

                    if key == "arguments" {
                        if let Ok(arguments) = serde_json::from_str::<serde_json::Value>(text) {
                            collect_hermes_path_candidates(&arguments, candidates);
                        }
                    }
                    collect_hermes_text_path_candidates(text, 20, candidates);
                } else {
                    collect_hermes_path_candidates(value, candidates);
                }
            }
        }
        serde_json::Value::Array(values) => {
            for value in values {
                collect_hermes_path_candidates(value, candidates);
            }
        }
        serde_json::Value::String(text) => {
            collect_hermes_text_path_candidates(text, 20, candidates);
        }
        _ => {}
    }
}

fn collect_hermes_text_path_candidates(
    text: &str,
    score: i32,
    candidates: &mut Vec<(i32, std::path::PathBuf)>,
) {
    for token in extract_hermes_path_tokens(text) {
        push_hermes_path_candidate(candidates, &token, score, HermesPathUse::MentionedPath);
    }
}

fn push_hermes_path_candidate(
    candidates: &mut Vec<(i32, std::path::PathBuf)>,
    raw: &str,
    score: i32,
    path_use: HermesPathUse,
) {
    if let Some(path) = normalize_hermes_path(raw, path_use) {
        candidates.push((score, path));
    }
}

fn best_hermes_cwd_candidate(
    candidates: Vec<(i32, std::path::PathBuf)>,
) -> Option<std::path::PathBuf> {
    let mut best_by_path: std::collections::HashMap<std::path::PathBuf, i32> =
        std::collections::HashMap::new();
    for (score, path) in candidates {
        best_by_path
            .entry(path)
            .and_modify(|existing| *existing = (*existing).max(score))
            .or_insert(score);
    }

    best_by_path
        .into_iter()
        .max_by(|(left_path, left_score), (right_path, right_score)| {
            left_score
                .cmp(right_score)
                .then_with(|| {
                    left_path
                        .components()
                        .count()
                        .cmp(&right_path.components().count())
                })
                .then_with(|| {
                    left_path
                        .to_string_lossy()
                        .cmp(&right_path.to_string_lossy())
                })
        })
        .map(|(path, _)| path)
}

fn normalize_hermes_path(raw: &str, path_use: HermesPathUse) -> Option<std::path::PathBuf> {
    let token = clean_hermes_path_token(raw)?;
    let token = strip_hermes_glob_suffix(&token);
    let path = expand_hermes_home_path(&token)?;
    if !path.is_absolute() {
        return None;
    }

    let dir = if path.is_dir() {
        path
    } else if path.is_file() {
        path.parent()?.to_path_buf()
    } else if path_use == HermesPathUse::ExactCwd {
        path
    } else {
        return None;
    };

    let path = if path_use == HermesPathUse::MentionedPath {
        hermes_project_root(&dir)
    } else {
        dir
    };

    (!is_ignored_hermes_path(&path)).then_some(path)
}

fn clean_hermes_path_token(raw: &str) -> Option<String> {
    let trimmed = raw
        .trim()
        .trim_start_matches(['"', '\'', '`', '(', '[', '{', '<'])
        .trim_end_matches(['"', '\'', '`', ',', ';', ':', ')', ']', '}', '>'])
        .trim_end_matches('/');
    (!trimmed.is_empty()).then(|| trimmed.to_string())
}

fn strip_hermes_glob_suffix(raw: &str) -> String {
    raw.find(['*', '?', '['])
        .map(|index| raw[..index].trim_end_matches('/').to_string())
        .unwrap_or_else(|| raw.to_string())
}

fn expand_hermes_home_path(raw: &str) -> Option<std::path::PathBuf> {
    if raw == "~" {
        Some(std::path::PathBuf::from(dirs_next_home()))
    } else if let Some(rest) = raw.strip_prefix("~/") {
        Some(std::path::PathBuf::from(dirs_next_home()).join(rest))
    } else if raw.starts_with('/') && !raw.starts_with("//") {
        Some(std::path::PathBuf::from(raw))
    } else {
        None
    }
}

fn extract_hermes_path_tokens(text: &str) -> Vec<String> {
    let mut tokens = Vec::new();
    let mut iter = text.char_indices().peekable();

    while let Some((start, ch)) = iter.next() {
        let starts_path = ch == '/' || (ch == '~' && text[start..].starts_with("~/"));
        if !starts_path {
            continue;
        }

        let mut end = text.len();
        while let Some(&(index, next)) = iter.peek() {
            if is_hermes_path_delimiter(next) {
                end = index;
                break;
            }
            iter.next();
        }

        if let Some(token) = clean_hermes_path_token(&text[start..end]) {
            tokens.push(token);
        }
    }

    tokens
}

fn is_hermes_path_delimiter(ch: char) -> bool {
    ch.is_whitespace()
        || matches!(
            ch,
            '"' | '\''
                | '`'
                | '<'
                | '>'
                | '('
                | ')'
                | '['
                | ']'
                | '{'
                | '}'
                | '|'
                | '&'
                | ';'
                | ','
        )
}

fn hermes_project_root(path: &std::path::Path) -> std::path::PathBuf {
    for ancestor in path.ancestors() {
        if ancestor.join(".git").exists() {
            return ancestor.to_path_buf();
        }
    }

    const MARKERS: &[&str] = &[
        "Cargo.toml",
        "pyproject.toml",
        "package.json",
        "pnpm-workspace.yaml",
        "package-lock.json",
        "yarn.lock",
        "uv.lock",
        "go.mod",
        "flake.nix",
    ];

    path.ancestors()
        .find(|ancestor| MARKERS.iter().any(|marker| ancestor.join(marker).exists()))
        .unwrap_or(path)
        .to_path_buf()
}

fn is_ignored_hermes_path(path: &std::path::Path) -> bool {
    let home = std::path::PathBuf::from(dirs_next_home());
    if path == home {
        return true;
    }

    for ignored in [
        home.join(".hermes"),
        home.join(".local"),
        home.join(".cache"),
        home.join(".cargo"),
        home.join(".rustup"),
        home.join(".ssh"),
        home.join("Library"),
    ] {
        if path.starts_with(ignored) {
            return true;
        }
    }

    [
        "/tmp",
        "/private/tmp",
        "/var/folders",
        "/usr",
        "/bin",
        "/sbin",
        "/System",
        "/Library",
        "/Applications",
        "/dev",
        "/etc",
    ]
    .iter()
    .any(|ignored| path.starts_with(ignored))
}

async fn hermes_native_sessions() -> Result<Vec<agent_client_protocol::schema::SessionInfo>, String>
{
    let db = hermes_state_db();
    if !db.exists() {
        return Ok(Vec::new());
    }

    let query = r#"
        SELECT
            s.id,
            COALESCE(
                NULLIF(s.title, ''),
                substr((
                    SELECT content
                    FROM messages
                    WHERE session_id = s.id
                        AND role = 'user'
                        AND COALESCE(content, '') <> ''
                    ORDER BY timestamp
                    LIMIT 1
                ), 1, 80),
                s.id
            ) AS title,
            COALESCE(
                (SELECT max(timestamp) FROM messages WHERE session_id = s.id),
                s.ended_at,
                s.started_at
            ) AS updated_at,
            COALESCE(s.model_config, '') AS model_config,
            COALESCE((
                SELECT json_group_array(tool_calls)
                FROM (
                    SELECT tool_calls
                    FROM messages
                    WHERE session_id = s.id
                        AND COALESCE(tool_calls, '') <> ''
                    ORDER BY timestamp DESC
                    LIMIT 30
                )
            ), '[]') AS tool_calls
        FROM sessions s
        ORDER BY updated_at DESC
        LIMIT 200;
    "#;

    let output = tokio::process::Command::new("sqlite3")
        .arg("-json")
        .arg(db)
        .arg(query)
        .output()
        .await
        .map_err(|e| format!("Failed to read Hermes sessions: {e}"))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(format!("Failed to read Hermes sessions: {stderr}"));
    }

    let rows: Vec<serde_json::Value> = serde_json::from_slice(&output.stdout)
        .map_err(|e| format!("Failed to parse Hermes sessions: {e}"))?;
    Ok(rows
        .into_iter()
        .filter_map(|row| {
            let id = row.get("id")?.as_str()?.to_string();
            let cwd = hermes_session_cwd(&row);
            let updated_at = value_f64(&row, "updated_at")
                .map(|ts| (ts as i64).to_string())
                .unwrap_or_default();
            let title = row
                .get("title")
                .and_then(|v| v.as_str())
                .filter(|s| !s.trim().is_empty())
                .map(ToString::to_string);
            let mut session = agent_client_protocol::schema::SessionInfo::new(id, cwd.clone())
                .updated_at(updated_at);
            if let Some(title) = title {
                session = session.title(title);
            }
            Some(session)
        })
        .collect())
}

async fn load_hermes_native_session(session_id: &str) -> Result<Vec<ChatMessage>, String> {
    let db = hermes_state_db();
    let session_id = sqlite_quote(session_id);
    let query = format!(
        r#"
        SELECT role, content, tool_name, timestamp
        FROM messages
        WHERE session_id = '{session_id}'
        ORDER BY timestamp;
        "#
    );

    let output = tokio::process::Command::new("sqlite3")
        .arg("-json")
        .arg(db)
        .arg(query)
        .output()
        .await
        .map_err(|e| format!("Failed to load Hermes session: {e}"))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(format!("Failed to load Hermes session: {stderr}"));
    }

    let rows: Vec<serde_json::Value> = serde_json::from_slice(&output.stdout)
        .map_err(|e| format!("Failed to parse Hermes session: {e}"))?;
    if rows.is_empty() {
        return Err("Hermes session not found".to_string());
    }

    Ok(rows
        .into_iter()
        .filter_map(|row| {
            let role = first_nonempty_string(&row, &["role"])?;
            let content = row
                .get("content")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
            if content.is_empty() {
                return None;
            }
            match role {
                "user" => Some(ChatMessage::new_user(content)),
                "tool" => Some(ChatMessage::new_tool_result(
                    row.get("tool_name")
                        .and_then(|v| v.as_str())
                        .unwrap_or("tool")
                        .to_string(),
                    content,
                )),
                _ => Some(ChatMessage::new_assistant(content)),
            }
        })
        .collect())
}

// ── Update ──────────────────────────────────────────────────────────────────

impl App {
    fn apply_launch_options(&mut self, options: LaunchOptions) {
        let Some(session_id) = options.session_id else {
            return;
        };
        let Some(index) = self.sessions.iter().position(|session| {
            session.id == session_id
                || session.acp_session_id.as_deref() == Some(session_id.as_str())
        }) else {
            self.pending_launch_session_id = Some(session_id);
            return;
        };

        self.pending_launch_session_id = None;
        self.active_session = Some(index);
        self.surface = Surface::Conversation;
        self.message_cache.set(
            self.sessions[index].id.clone(),
            self.sessions[index].messages.clone(),
        );
    }

    pub fn update(&mut self, message: Message) -> Task<Message> {
        match message {
            Message::NewSession => {
                let Some(inst) = self.instances.first() else {
                    return Task::none();
                };
                let cwd = inst
                    .default_cwd
                    .as_deref()
                    .map(|p| p.display().to_string())
                    .unwrap_or_else(|| self.home_dir.clone());
                let id = instance::uuid_v7();
                let session = Session::new(&id, &inst.id, &cwd, None);
                self.sessions.insert(0, session);
                self.active_session = Some(0);
                self.surface = Surface::Conversation;
                self.error = None;
                save_session(&mut self.sessions[0]);
                Task::none()
            }
            Message::CreateSession(ref instance_id) => {
                let Some(inst) = self.instances.iter().find(|i| &i.id == instance_id) else {
                    return Task::none();
                };
                let cwd = inst
                    .default_cwd
                    .as_deref()
                    .map(|p| p.display().to_string())
                    .unwrap_or_else(|| self.home_dir.clone());
                let id = instance::uuid_v7();
                let session = Session::new(&id, instance_id, &cwd, None);
                self.sessions.insert(0, session);
                self.active_session = Some(0);
                self.surface = Surface::Conversation;
                self.error = None;
                save_session(&mut self.sessions[0]);
                update_session_cache(&mut self.session_cache, &self.sessions);
                Task::none()
            }
            Message::SelectSession(i) => {
                if i < self.sessions.len() {
                    self.active_session = Some(i);
                    self.surface = Surface::Conversation;
                    self.error = None;
                    self.renaming_session = None;
                    self.rename_buffer.clear();
                    self.thinking_expanded.clear();
                    self.tool_expanded.clear();
                    if self.sessions[i].unread {
                        self.sessions[i].unread = false;
                        write_session(&self.sessions[i]);
                    }
                    // Update SWR message cache so subsequent selects hit cache.
                    let sid = self.sessions[i].id.clone();
                    self.message_cache
                        .set(sid, self.sessions[i].messages.clone());
                }
                Task::none()
            }
            Message::DeleteSession(i) => {
                if i < self.sessions.len() {
                    let session_id = self.sessions[i].id.clone();
                    self.sessions.remove(i);
                    self.active_session = match self.active_session {
                        Some(active) if active == i => None,
                        Some(active) if active > i => Some(active - 1),
                        other => other,
                    };
                    delete_session_file(&session_id);
                    update_session_cache(&mut self.session_cache, &self.sessions);
                    if self.active_session.is_none() {
                        self.surface = Surface::Welcome;
                    }
                }
                Task::none()
            }
            Message::TogglePinSession(i) => {
                if let Some(session) = self.sessions.get_mut(i) {
                    session.pinned = !session.pinned;
                    write_session(session);
                    update_session_cache(&mut self.session_cache, &self.sessions);
                }
                Task::none()
            }
            Message::StartRenameSession(i) => {
                if let Some(session) = self.sessions.get(i) {
                    self.renaming_session = Some(i);
                    self.rename_buffer = session_title_for_edit(session);
                }
                Task::none()
            }
            Message::SessionRenameChanged(value) => {
                self.rename_buffer = value;
                Task::none()
            }
            Message::CommitSessionRename => {
                if let Some(i) = self.renaming_session.take()
                    && let Some(session) = self.sessions.get_mut(i)
                {
                    let title = self.rename_buffer.trim();
                    session.title = if title.is_empty() {
                        None
                    } else {
                        Some(title.to_string())
                    };
                    write_session(session);
                    update_session_cache(&mut self.session_cache, &self.sessions);
                }
                self.rename_buffer.clear();
                Task::none()
            }
            Message::CancelSessionRename => {
                self.renaming_session = None;
                self.rename_buffer.clear();
                Task::none()
            }
            Message::ArchiveSession(i) => {
                if let Some(session) = self.sessions.get_mut(i) {
                    session.archived = true;
                    write_session(session);
                    if self.active_session == Some(i) {
                        self.active_session = None;
                        self.surface = Surface::Welcome;
                    }
                    if self.renaming_session == Some(i) {
                        self.renaming_session = None;
                        self.rename_buffer.clear();
                    }
                    update_session_cache(&mut self.session_cache, &self.sessions);
                }
                Task::none()
            }
            Message::ToggleUnreadSession(i) => {
                if let Some(session) = self.sessions.get_mut(i) {
                    session.unread = !session.unread;
                    write_session(session);
                    update_session_cache(&mut self.session_cache, &self.sessions);
                }
                Task::none()
            }
            Message::OpenWorkingDirectory(path) => {
                if let Err(e) = open_working_directory(&path) {
                    self.error = Some(e);
                }
                Task::none()
            }
            Message::OpenSessionInNewWindow(session_id) => {
                if let Err(e) = open_session_in_new_window(&session_id) {
                    self.error = Some(e);
                }
                Task::none()
            }
            Message::FilterChanged(change) => {
                match change {
                    FilterChange::Grouping(g) => {
                        self.session_grouping = g;
                        self.app_settings.session_grouping = g;
                    }
                    FilterChange::Sort(s) => {
                        self.session_sort_by = s;
                        self.app_settings.session_sort_by = s;
                    }
                    FilterChange::Project(p) => {
                        self.project_filter = p.clone();
                        self.app_settings.project_filter = p;
                    }
                }
                self.save_settings();
                Task::none()
            }

            Message::ToggleGroup(name) => {
                if self.group_collapsed.contains(&name) {
                    self.group_collapsed.remove(&name);
                } else {
                    self.group_collapsed.insert(name);
                }
                Task::none()
            }
            Message::SearchSidebar(q) => {
                self.sidebar_search = q;
                Task::none()
            }
            Message::ToggleSettings => {
                self.surface = match self.surface {
                    Surface::Settings { .. } => {
                        if self.active_session.is_some() {
                            Surface::Conversation
                        } else {
                            Surface::Welcome
                        }
                    }
                    _ => Surface::Settings {
                        tab: settings::Tab::Appearance,
                    },
                };
                Task::none()
            }
            Message::StopGeneration => {
                self.cancel_requested = true;
                self.loading = false;
                Task::none()
            }
            Message::ModelChanged(idx, model) => {
                if let Some(inst) = self.instances.get_mut(idx) {
                    inst.model = model;
                    self.save_instances();
                }
                Task::none()
            }
            Message::AddProvider => {
                let p = ModelProvider {
                    id: instance::uuid_v7(),
                    label: "New Provider".into(),
                    kind: crate::instance::ProviderKind::Custom,
                    base_url: None,
                    api_key_ref: None,
                    default_model: "custom".into(),
                    available_models: instance::default_models_for_kind(
                        crate::instance::ProviderKind::Custom,
                    ),
                };
                self.providers.push(p);
                self.save_providers();
                Task::none()
            }
            Message::DeleteProvider(i) => {
                self.pending_delete_provider = Some(i);
                Task::none()
            }
            Message::ConfirmDeleteProvider(i) => {
                if i < self.providers.len() {
                    self.providers.remove(i);
                }
                self.pending_delete_provider = None;
                self.save_providers();
                Task::none()
            }
            Message::CancelDeleteProvider => {
                self.pending_delete_provider = None;
                Task::none()
            }
            Message::UpdateProviderLabel(i, s) => {
                if let Some(p) = self.providers.get_mut(i) {
                    p.label = s;
                }
                self.save_providers();
                Task::none()
            }
            Message::UpdateProviderKind(i, k) => {
                if let Some(p) = self.providers.get_mut(i) {
                    p.kind = k;
                    p.available_models = instance::default_models_for_kind(k);
                    if !p.available_models.contains(&p.default_model) {
                        p.default_model = p.available_models.first().cloned().unwrap_or_default();
                    }
                }
                self.save_providers();
                Task::none()
            }
            Message::UpdateProviderBaseUrl(i, s) => {
                if let Some(p) = self.providers.get_mut(i) {
                    p.base_url = if s.is_empty() { None } else { Some(s) };
                }
                self.save_providers();
                Task::none()
            }
            Message::SetProviderApiKey(i, key) => {
                let persistence_enabled = self.persistence_enabled;
                if let Some(p) = self.providers.get_mut(i) {
                    if key.is_empty() {
                        p.api_key_ref = None;
                    } else if !persistence_enabled || instance::set_api_key(&p.id, &key).is_ok() {
                        p.api_key_ref = Some(p.id.clone());
                    }
                }
                self.save_providers();
                Task::none()
            }
            Message::UpdateProviderDefaultModel(i, s) => {
                if let Some(p) = self.providers.get_mut(i) {
                    p.default_model = s;
                }
                self.save_providers();
                Task::none()
            }
            Message::AddInstance => {
                let Some(prov) = self.providers.first() else {
                    return Task::none();
                };
                let id = instance::uuid_v7();
                let inst = AgentInstance {
                    id: id.clone(),
                    label: "New Agent".into(),
                    harness: Harness::ClaudeCode,
                    provider_id: prov.id.clone(),
                    model: prov.default_model.clone(),
                    instructions: None,
                    config_dir: dirs_next::home_dir()
                        .unwrap_or_else(|| std::path::PathBuf::from("."))
                        .join(".swarmx")
                        .join("instances")
                        .join(&id),
                    icon_override: None,
                    default_cwd: None,
                };
                self.instances.push(inst);
                self.save_instances();
                Task::none()
            }
            Message::DeleteInstance(i) => {
                self.pending_delete_instance = Some(i);
                Task::none()
            }
            Message::ConfirmDeleteInstance(i) => {
                if i < self.instances.len() {
                    self.instances.remove(i);
                }
                self.pending_delete_instance = None;
                self.save_instances();
                Task::none()
            }
            Message::CancelDeleteInstance => {
                self.pending_delete_instance = None;
                Task::none()
            }
            Message::UpdateInstanceLabel(i, s) => {
                if let Some(x) = self.instances.get_mut(i) {
                    x.label = s;
                }
                self.save_instances();
                Task::none()
            }
            Message::UpdateInstanceHarness(i, h) => {
                if let Some(x) = self.instances.get_mut(i) {
                    x.harness = h;
                }
                self.save_instances();
                Task::none()
            }
            Message::UpdateInstanceProviderId(i, pid) => {
                if let Some(x) = self.instances.get_mut(i) {
                    x.provider_id = pid.clone();
                    if let Some(p) = self.providers.iter().find(|p| p.id == pid) {
                        if !p.available_models.contains(&x.model) {
                            x.model = p.default_model.clone();
                        }
                    }
                }
                self.save_instances();
                Task::none()
            }
            Message::UpdateInstanceModel(i, m) => {
                if let Some(x) = self.instances.get_mut(i) {
                    x.model = m;
                }
                self.save_instances();
                Task::none()
            }
            Message::UpdateInstanceInstructions(i, s) => {
                if let Some(x) = self.instances.get_mut(i) {
                    x.instructions = if s.is_empty() { None } else { Some(s) };
                }
                self.save_instances();
                Task::none()
            }
            Message::UpdateInstanceDefaultCwd(i, s) => {
                if let Some(x) = self.instances.get_mut(i) {
                    x.default_cwd = if s.is_empty() { None } else { Some(s.into()) };
                }
                self.save_instances();
                Task::none()
            }
            Message::GoToSettingsTab(tab) => {
                self.surface = Surface::Settings { tab };
                Task::none()
            }
            Message::SetTheme(pref) => {
                self.theme_preference = pref;
                self.theme = pref.shadcn_theme();
                self.app_settings.theme = pref;
                self.save_settings();
                Task::none()
            }
            Message::SetDensity(d) => {
                self.density = d;
                self.tokens = DesignTokens::for_density(d);
                self.app_settings.density = d;
                self.save_settings();
                Task::none()
            }
            Message::InputChanged(s) => {
                self.input = s;
                Task::none()
            }
            Message::SendMessage => {
                let text = self.input.trim().to_string();
                if text.is_empty() || self.loading {
                    return Task::none();
                }
                let Some(active) = self.active_session else {
                    return Task::none();
                };
                if active >= self.sessions.len() {
                    return Task::none();
                }

                self.input.clear();
                self.error = None;
                self.sessions[active]
                    .messages
                    .push(ChatMessage::new(true, text.clone()));
                {
                    let contents: Vec<_> = self.sessions[active]
                        .messages
                        .iter()
                        .filter(|m| !m.is_user)
                        .map(|m| iced::widget::markdown::Content::parse(&m.content))
                        .collect();
                    self.md_cache
                        .insert(self.sessions[active].id.clone(), contents);
                }
                self.loading = true;
                self.cancel_requested = false;

                // Update SWR cache with user message included.
                let sid = self.sessions[active].id.clone();
                self.message_cache
                    .set(sid, self.sessions[active].messages.clone());

                let agent = if let Some(runtime) = self.sessions[active].agent_runtime {
                    build_runtime_agent(runtime)
                } else {
                    let instance_id = self.sessions[active].agent_instance_id.clone();
                    let Some(inst) = self.instances.iter().find(|i| i.id == instance_id) else {
                        self.loading = false;
                        return Task::none();
                    };
                    let Some(provider) = self.providers.iter().find(|p| p.id == inst.provider_id)
                    else {
                        self.loading = false;
                        return Task::none();
                    };
                    build_agent(inst, provider)
                };
                let agent_name = agent.name.clone();
                let all_messages: Vec<_> = self.sessions[active]
                    .messages
                    .iter()
                    .map(|m| {
                        if m.is_user {
                            serde_json::json!({"role": "user", "content": m.content})
                        } else {
                            serde_json::json!({"role": "assistant", "content": m.content})
                        }
                    })
                    .collect();

                Task::perform(
                    async move {
                        let swarm =
                            Swarm::new("chat", &agent_name).with_node(SwarmNode::Agent(agent));
                        swarm
                            .execute(serde_json::json!({"messages": all_messages}), None)
                            .await
                            .map(|msgs| {
                                msgs.into_iter()
                                    .filter(|v| {
                                        v.get("content")
                                            .and_then(|c| c.as_str())
                                            .is_some_and(|c| !c.is_empty())
                                    })
                                    .collect::<Vec<_>>()
                            })
                            .map_err(|e| e.to_string())
                    },
                    Message::Response,
                )
            }
            Message::Response(Ok(updates)) => {
                if self.cancel_requested {
                    self.cancel_requested = false;
                    // Append a system note to the active session
                    if let Some(active) = self.active_session {
                        if active < self.sessions.len() {
                            self.sessions[active]
                                .messages
                                .push(ChatMessage::new_assistant("[Generation stopped.]".into()));
                        }
                    }
                    return Task::none();
                }
                let Some(active) = self.active_session else {
                    self.loading = false;
                    return Task::none();
                };
                if active >= self.sessions.len() {
                    self.loading = false;
                    return Task::none();
                }
                for u in &updates {
                    let content = u.get("content").and_then(|c| c.as_str()).unwrap_or("");
                    if content.is_empty() {
                        continue;
                    }
                    let kind_str = u.get("kind").and_then(|k| k.as_str()).unwrap_or("message");
                    let tool_name = u
                        .get("tool_name")
                        .and_then(|n| n.as_str())
                        .map(|s| s.to_string());
                    match kind_str {
                        "tool_call" => {
                            self.sessions[active]
                                .messages
                                .push(ChatMessage::new_tool_call(
                                    tool_name.unwrap_or_default(),
                                    content.to_string(),
                                ));
                        }
                        "tool_result" => {
                            self.sessions[active]
                                .messages
                                .push(ChatMessage::new_tool_result(
                                    tool_name.unwrap_or_default(),
                                    content.to_string(),
                                ));
                        }
                        "thinking" => {
                            self.sessions[active]
                                .messages
                                .push(ChatMessage::new_thinking(content.to_string()));
                        }
                        _ => {
                            self.sessions[active]
                                .messages
                                .push(ChatMessage::new_assistant(content.to_string()));
                        }
                    }
                }
                self.loading = false;
                self.thinking_expanded.clear();
                self.tool_expanded.clear();
                {
                    let contents: Vec<_> = self.sessions[active]
                        .messages
                        .iter()
                        .filter(|m| !m.is_user)
                        .map(|m| iced::widget::markdown::Content::parse(&m.content))
                        .collect();
                    self.md_cache
                        .insert(self.sessions[active].id.clone(), contents);
                }
                save_session(&mut self.sessions[active]);
                let sid = self.sessions[active].id.clone();
                self.message_cache
                    .set(sid, self.sessions[active].messages.clone());
                Task::none()
            }
            Message::Response(Err(e)) => {
                self.error = Some(e);
                self.loading = false;
                Task::none()
            }
            Message::UseSuggestion(text) => {
                self.input = text;
                Task::none()
            }
            Message::LinkClicked(_url) => {
                // TODO: open in browser
                Task::none()
            }
            Message::CopyToClipboard(text) => iced::clipboard::write(text),
            Message::SetThinkingOpen(idx, open) => {
                if open {
                    self.thinking_expanded.insert(idx);
                } else {
                    self.thinking_expanded.remove(&idx);
                }
                Task::none()
            }
            Message::SetToolOpen(idx, open) => {
                if open {
                    self.tool_expanded.insert(idx);
                } else {
                    self.tool_expanded.remove(&idx);
                }
                Task::none()
            }
            Message::Tick(_instant) => {
                self.spinner_progress = (self.spinner_progress + 0.03) % 1.0;
                Task::none()
            }
            Message::WindowFocused => Task::done(Message::RefreshAgentSessions),
            Message::RefreshAgentSessions => {
                // Configured instances win because they carry user-selected provider/env.
                // Available discovered runtimes are queried when no instance covers them yet.
                let mut configured_runtimes = std::collections::HashSet::new();
                let mut triples: Vec<(RemoteAgentRef, String, Agent, Option<AgentRuntime>)> =
                    Vec::new();
                for inst in self
                    .instances
                    .iter()
                    .filter(|inst| inst.harness != Harness::SwarmX)
                {
                    if let Some(runtime) = runtime_for_harness(inst.harness) {
                        configured_runtimes.insert(runtime);
                    }
                    let Some(provider) = self.providers.iter().find(|p| p.id == inst.provider_id)
                    else {
                        continue;
                    };
                    triples.push((
                        RemoteAgentRef::Instance(inst.id.clone()),
                        inst.label.clone(),
                        build_agent(inst, provider),
                        runtime_for_harness(inst.harness),
                    ));
                }
                for status in self
                    .agent_statuses
                    .iter()
                    .filter(|status| status.available)
                    .filter(|status| !configured_runtimes.contains(&status.agent))
                {
                    triples.push((
                        RemoteAgentRef::Runtime(status.agent),
                        status.agent.session_label().to_string(),
                        build_runtime_agent(status.agent),
                        Some(status.agent),
                    ));
                }
                update_session_cache(&mut self.session_cache, &self.sessions);
                Task::perform(
                    async move {
                        let mut all = Vec::new();
                        for (agent_ref, agent_label, agent, runtime) in &triples {
                            match agent.list_sessions(None).await {
                                Ok(resp) => {
                                    let mut sessions = resp.sessions;
                                    let mut source = RemoteSessionSource::Acp;
                                    if sessions.is_empty() && *runtime == Some(AgentRuntime::Hermes)
                                    {
                                        match hermes_native_sessions().await {
                                            Ok(native_sessions) if !native_sessions.is_empty() => {
                                                sessions = native_sessions;
                                                source = RemoteSessionSource::HermesNative;
                                            }
                                            Ok(_) => {}
                                            Err(e) => {
                                                eprintln!(
                                                    "Failed to list Hermes native sessions: {}",
                                                    e
                                                );
                                            }
                                        }
                                    }
                                    all.push(RemoteAgentSessions {
                                        agent_name: agent_label.clone(),
                                        agent_ref: agent_ref.clone(),
                                        source,
                                        sessions,
                                    });
                                }
                                Err(e) => {
                                    eprintln!("Failed to list sessions for {}: {}", agent_label, e);
                                }
                            }
                        }
                        Ok(all)
                    },
                    Message::AgentSessionsResult,
                )
            }
            Message::AgentSessionsResult(result) => {
                match result {
                    Ok(mut sessions) => {
                        let loaded_ids: std::collections::HashSet<String> = self
                            .sessions
                            .iter()
                            .filter_map(|s| s.acp_session_id.clone())
                            .collect();
                        for ag in &mut sessions {
                            ag.sessions
                                .retain(|s| !loaded_ids.contains(&s.session_id.to_string()));
                        }
                        let launch_remote =
                            self.pending_launch_session_id
                                .as_deref()
                                .and_then(|pending| {
                                    sessions.iter().find_map(|ag| {
                                        ag.sessions.iter().find_map(|session| {
                                            (session.session_id.to_string() == pending).then(|| {
                                                (
                                                    ag.agent_ref.clone(),
                                                    ag.source,
                                                    pending.to_string(),
                                                    session.cwd.display().to_string(),
                                                )
                                            })
                                        })
                                    })
                                });
                        if launch_remote.is_some() {
                            self.pending_launch_session_id = None;
                        }
                        self.remote_sessions = sessions;
                        self.startup_loading = false;
                        update_session_cache(&mut self.session_cache, &self.sessions);
                        if let Some((agent_ref, source, session_id, cwd)) = launch_remote {
                            return Task::done(Message::LoadRemoteSession(
                                agent_ref, source, session_id, cwd,
                            ));
                        }
                    }
                    Err(e) => {
                        eprintln!("Agent session refresh failed: {}", e);
                        self.startup_loading = false;
                    }
                }
                Task::none()
            }
            Message::LoadRemoteSession(ref agent_ref, source, session_id, cwd) => {
                let instances = self.instances.clone();
                let providers = self.providers.clone();
                let agent_ref1 = agent_ref.clone();
                let agent_ref2 = agent_ref.clone();
                let session_id2 = session_id.clone();
                Task::perform(
                    async move {
                        let (agent, agent_instance_id, agent_runtime) = match &agent_ref1 {
                            RemoteAgentRef::Instance(instance_id) => {
                                let inst = instances
                                    .iter()
                                    .find(|i| i.id == *instance_id)
                                    .ok_or_else(|| "Instance not found".to_string())?;
                                let provider = providers
                                    .iter()
                                    .find(|p| p.id == inst.provider_id)
                                    .ok_or_else(|| "Provider not found".to_string())?;
                                (build_agent(inst, provider), inst.id.clone(), None)
                            }
                            RemoteAgentRef::Runtime(runtime) => (
                                build_runtime_agent(*runtime),
                                runtime.id().to_string(),
                                Some(*runtime),
                            ),
                        };
                        let chat_msgs: Vec<ChatMessage> =
                            if source == RemoteSessionSource::HermesNative {
                                load_hermes_native_session(&session_id).await?
                            } else {
                                let (_resp, messages) = agent
                                    .load_session(&session_id, std::path::Path::new(&cwd))
                                    .await
                                    .map_err(|e| e.to_string())?;
                                messages
                                    .into_iter()
                                    .filter_map(convert_acp_message)
                                    .collect()
                            };
                        let title = chat_msgs
                            .iter()
                            .find(|m| m.is_user)
                            .map(|m| {
                                let t: String = m.content.chars().take(40).collect();
                                if m.content.chars().count() > 40 {
                                    format!("{t}...")
                                } else {
                                    t
                                }
                            })
                            .unwrap_or_default();
                        Ok((
                            LoadSessionMeta {
                                session_id,
                                title,
                                cwd,
                                agent_instance_id,
                                agent_runtime,
                            },
                            chat_msgs,
                        ))
                    },
                    move |res| Message::RemoteSessionLoaded(res, agent_ref2, source, session_id2),
                )
            }
            Message::RemoteSessionLoaded(result, _agent_ref, _source, _session_id) => {
                match result {
                    Ok((meta, messages)) => {
                        let id = instance::uuid_v7();
                        let mut session = Session::new(
                            &id,
                            &meta.agent_instance_id,
                            &meta.cwd,
                            Some(meta.session_id.clone()),
                        );
                        session.agent_runtime = meta.agent_runtime;
                        if !meta.title.is_empty() {
                            session.title = Some(meta.title);
                        }
                        session.messages = messages;
                        let contents: Vec<_> = session
                            .messages
                            .iter()
                            .filter(|m| !m.is_user)
                            .map(|m| iced::widget::markdown::Content::parse(&m.content))
                            .collect();
                        self.md_cache.insert(id.clone(), contents);
                        save_session(&mut session);
                        // Remove loaded session from remote list to prevent duplicate
                        if let Some(ref acp_id) = session.acp_session_id {
                            for ag in &mut self.remote_sessions {
                                ag.sessions.retain(|s| s.session_id.to_string() != *acp_id);
                            }
                        }
                        self.sessions.push(session);
                        self.active_session = Some(self.sessions.len() - 1);
                        self.surface = Surface::Conversation;
                    }
                    Err(e) => {
                        self.error = Some(format!("Failed to load session: {}", e));
                    }
                }
                Task::none()
            }
            Message::InstallTool(dep) => {
                if self.env_installing.is_some() {
                    return Task::none();
                }
                self.env_installing = Some(dep);
                Task::perform(
                    async move { environment::install(dep).await.map_err(|e| e.to_string()) },
                    move |res| Message::InstallResult(dep, res),
                )
            }
            Message::InstallResult(dep, result) => {
                self.env_installing = None;
                match result {
                    Ok(()) => {
                        // Re-check the just-installed dep
                        if let Some(status) = self.env_checks.iter_mut().find(|s| s.dep == dep) {
                            *status = environment::check_one_sync(dep);
                        }
                        // Chain: bun installed → check claude → prompt if missing
                        // Chain: claude installed → re-scan claude-agent-acp
                        match dep {
                            RuntimeDep::Bun => {
                                // Re-scan all bun-dependent agents (now have bun, maybe not claude)
                                for status in &mut self.agent_statuses {
                                    if status.agent.needs_bun() {
                                        *status = environment::check_agent_sync(status.agent);
                                    }
                                }
                                // If claude CLI still missing, auto-start its install
                                let claude_missing = self
                                    .env_checks
                                    .iter()
                                    .any(|s| s.dep == RuntimeDep::Claude && !s.installed);
                                if claude_missing && self.env_installing.is_none() {
                                    self.env_installing = Some(RuntimeDep::Claude);
                                    return Task::perform(
                                        async {
                                            environment::install(RuntimeDep::Claude)
                                                .await
                                                .map_err(|e| e.to_string())
                                        },
                                        move |res| Message::InstallResult(RuntimeDep::Claude, res),
                                    );
                                }
                                return Task::done(Message::RefreshAgentSessions);
                            }
                            RuntimeDep::Claude => {
                                // Claude CLI installed — re-scan claude-agent-acp
                                for status in &mut self.agent_statuses {
                                    if status.agent == AgentRuntime::ClaudeAgentAcp {
                                        *status = environment::check_agent_sync(
                                            AgentRuntime::ClaudeAgentAcp,
                                        );
                                    }
                                }
                                return Task::done(Message::RefreshAgentSessions);
                            }
                            _ => {}
                        }
                    }
                    Err(e) => {
                        self.error = Some(format!("Failed to install {}: {}", dep.label(), e));
                    }
                }
                Task::none()
            }
            Message::ToggleSidebar => {
                self.sidebar_collapsed = !self.sidebar_collapsed;
                Task::none()
            }
            Message::StartSidebarDrag => {
                self.sidebar_dragging = true;
                Task::none()
            }
            Message::SidebarDrag(x) => {
                if self.sidebar_dragging {
                    self.sidebar_width = x.clamp(180.0, 480.0);
                }
                Task::none()
            }
            Message::EndSidebarDrag => {
                self.sidebar_dragging = false;
                Task::none()
            }
        }
    }

    pub fn subscription(&self) -> iced::Subscription<Message> {
        let tick = iced::time::every(std::time::Duration::from_millis(16)).map(Message::Tick);
        let drag = if self.sidebar_dragging {
            iced::event::listen_with(|event, _status, _app| match event {
                iced::Event::Mouse(iced::mouse::Event::CursorMoved { position }) => {
                    Some(Message::SidebarDrag(position.x))
                }
                iced::Event::Mouse(iced::mouse::Event::ButtonReleased(_)) => {
                    Some(Message::EndSidebarDrag)
                }
                _ => None,
            })
        } else {
            iced::Subscription::none()
        };
        let focus = iced::event::listen_with(|event, _status, _app| match event {
            iced::Event::Window(iced::window::Event::Focused) => Some(Message::WindowFocused),
            _ => None,
        });
        iced::Subscription::batch([tick, drag, focus])
    }
}

// ── View ────────────────────────────────────────────────────────────────────

impl App {
    pub fn view(&self) -> Element<'_, Message> {
        crate::view::main_view(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tool_calls_json(arguments: serde_json::Value) -> String {
        serde_json::json!([serde_json::json!([
            {
                "type": "function",
                "function": {
                    "name": "terminal",
                    "arguments": arguments.to_string()
                }
            }
        ])
        .to_string()])
        .to_string()
    }

    #[test]
    fn hermes_model_config_cwd_uses_recorded_cwd() {
        let cwd = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let model_config = serde_json::json!({ "cwd": cwd.display().to_string() }).to_string();

        assert_eq!(hermes_model_config_cwd(&model_config), Some(cwd));
    }

    #[test]
    fn hermes_tool_calls_cwd_uses_explicit_workdir() {
        let cwd = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let tool_calls = tool_calls_json(serde_json::json!({
            "command": "cargo check",
            "workdir": cwd.display().to_string()
        }));

        assert_eq!(hermes_tool_calls_cwd(&tool_calls), Some(cwd));
    }

    #[test]
    fn hermes_tool_calls_cwd_ignores_tmp_workdir() {
        let tool_calls = tool_calls_json(serde_json::json!({
            "command": "python3 script.py",
            "workdir": "/tmp"
        }));

        assert_eq!(hermes_tool_calls_cwd(&tool_calls), None);
    }

    #[test]
    fn hermes_session_cwd_falls_back_to_unknown() {
        let row = serde_json::json!({
            "model_config": "",
            "tool_calls": "[]"
        });

        assert_eq!(hermes_session_cwd(&row), std::path::PathBuf::new());
    }
}
