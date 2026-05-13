use agent_client_protocol::schema::SessionInfo as AcpSessionInfo;

use crate::environment::{AgentRuntime, RemoteAgentRef};
use crate::view::chat_message::ChatMessage;

fn is_false(value: &bool) -> bool {
    !*value
}

// ── Session ──────────────────────────────────────────────────────────────────

#[derive(Clone, serde::Serialize, serde::Deserialize)]
pub struct Session {
    pub id: String,
    #[serde(default)]
    pub agent_instance_id: String,
    pub working_dir: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model_override: Option<String>,
    #[serde(default)]
    pub acp_session_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub agent_runtime: Option<AgentRuntime>,
    pub messages: Vec<ChatMessage>,
    pub created_at: String,
    pub updated_at: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,
    #[serde(default, skip_serializing_if = "is_false")]
    pub pinned: bool,
    #[serde(default, skip_serializing_if = "is_false")]
    pub archived: bool,
    #[serde(default, skip_serializing_if = "is_false")]
    pub unread: bool,
}

impl Session {
    pub fn new(
        id: &str,
        agent_instance_id: &str,
        working_dir: &str,
        acp_session_id: Option<String>,
    ) -> Self {
        use std::time::{SystemTime, UNIX_EPOCH};
        let ts = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        Self {
            id: id.to_string(),
            agent_instance_id: agent_instance_id.to_string(),
            working_dir: working_dir.to_string(),
            model_override: None,
            acp_session_id,
            agent_runtime: None,
            messages: Vec::new(),
            created_at: ts.to_string(),
            updated_at: ts.to_string(),
            title: None,
            pinned: false,
            archived: false,
            unread: false,
        }
    }
}

// ── Agent entry ──────────────────────────────────────────────────────────────

/// Icon identifier for each agent backend type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum AgentIcon {
    SwarmX,
    ClaudeCode,
    Codex,
    OpenCode,
    Hermes,
    OpenClaw,
}

impl AgentIcon {
    /// SVG bytes adapted for the current theme.
    /// OpenCode and Hermes use `fill="currentColor"` which iced doesn't resolve —
    /// we swap it to white on dark backgrounds.
    pub fn svg_bytes(self, is_dark: bool) -> std::borrow::Cow<'static, [u8]> {
        let raw: &[u8] = match self {
            AgentIcon::SwarmX => include_bytes!("../resources/swarmx.svg"),
            AgentIcon::ClaudeCode => include_bytes!("../resources/claudecode-color.svg"),
            AgentIcon::Codex => include_bytes!("../resources/codex-color.svg"),
            AgentIcon::OpenCode => include_bytes!("../resources/opencode.svg"),
            AgentIcon::Hermes => include_bytes!("../resources/hermes.svg"),
            AgentIcon::OpenClaw => include_bytes!("../resources/openclaw.svg"),
        };

        if is_dark && matches!(self, AgentIcon::OpenCode | AgentIcon::Hermes) {
            let modified = std::str::from_utf8(raw)
                .unwrap_or("")
                .replace("currentColor", "#ffffff");
            std::borrow::Cow::Owned(modified.into_bytes())
        } else {
            std::borrow::Cow::Borrowed(raw)
        }
    }
}

// ── Remote Sessions (ACP) ───────────────────────────────────────────────────

/// Sessions from a single ACP agent, using native ACP SessionInfo types.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RemoteSessionSource {
    Acp,
    HermesNative,
}

/// Sessions from a single remote agent.
#[derive(Debug, Clone)]
pub struct RemoteAgentSessions {
    pub agent_name: String,
    pub agent_ref: RemoteAgentRef,
    pub source: RemoteSessionSource,
    pub sessions: Vec<AcpSessionInfo>,
}

/// Lightweight metadata from a loaded ACP session — used before the local Session is created.
#[derive(Debug, Clone)]
pub struct LoadSessionMeta {
    pub session_id: String,
    pub title: String,
    pub cwd: String,
    pub agent_instance_id: String,
    pub agent_runtime: Option<AgentRuntime>,
}
