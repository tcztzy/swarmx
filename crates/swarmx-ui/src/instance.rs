//! Agent instance & model provider data model (spec §4).
//!
//! Persisted to `~/.swarmx/instances.json` and `~/.swarmx/providers.json`.
//! Seeds one Anthropic provider + one Claude instance on first launch.

use std::io;
use std::path::PathBuf;

use crate::data::AgentIcon;
use crate::harness::Harness;

// ── Provider Kind ──────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProviderKind {
    Anthropic,
    #[serde(rename = "openai_responses")]
    OpenAIResponses,
    #[serde(rename = "openai_chat")]
    OpenAIChat,
    Ollama,
    Custom,
}

impl ProviderKind {
    pub fn all() -> [ProviderKind; 5] {
        [
            Self::Anthropic,
            Self::OpenAIResponses,
            Self::OpenAIChat,
            Self::Ollama,
            Self::Custom,
        ]
    }

    pub fn label(&self) -> &'static str {
        match self {
            Self::Anthropic => "Anthropic",
            Self::OpenAIResponses => "OpenAI (Responses)",
            Self::OpenAIChat => "OpenAI (Chat)",
            Self::Ollama => "Ollama",
            Self::Custom => "Custom",
        }
    }
}

// ── Model Provider ─────────────────────────────────────────────────────────────

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ModelProvider {
    pub id: String,
    pub label: String,
    pub kind: ProviderKind,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub base_url: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub api_key_ref: Option<String>,
    pub default_model: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub available_models: Vec<String>,
}

impl ModelProvider {
    /// Resolve the API key from the OS keychain, if `api_key_ref` is set.
    fn resolve_api_key(&self) -> Option<String> {
        let ref_name = self.api_key_ref.as_ref()?;
        let service = format!("swarmx.provider.{}", ref_name);
        keyring::Entry::new(&service, "default")
            .and_then(|e| e.get_password())
            .ok()
    }

    /// Environment variables for this provider + model combination.
    pub fn env_vars(&self, model: &str) -> Vec<(&'static str, String)> {
        let key = self.resolve_api_key();
        match self.kind {
            ProviderKind::Anthropic => {
                let mut vars = vec![("ANTHROPIC_MODEL", model.to_string())];
                if let Some(k) = key {
                    vars.push(("ANTHROPIC_API_KEY", k));
                }
                if let Some(ref url) = self.base_url {
                    vars.push(("ANTHROPIC_BASE_URL", url.clone()));
                }
                vars
            }
            ProviderKind::OpenAIResponses => {
                let mut vars = vec![("OPENAI_RESPONSES_MODEL", model.to_string())];
                if let Some(k) = key {
                    vars.push(("OPENAI_API_KEY", k));
                }
                if let Some(ref url) = self.base_url {
                    vars.push(("OPENAI_BASE_URL", url.clone()));
                }
                vars
            }
            ProviderKind::OpenAIChat => {
                let mut vars = vec![("OPENAI_CHAT_MODEL", model.to_string())];
                if let Some(k) = key {
                    vars.push(("OPENAI_API_KEY", k));
                }
                if let Some(ref url) = self.base_url {
                    vars.push(("OPENAI_BASE_URL", url.clone()));
                }
                vars
            }
            ProviderKind::Ollama => {
                let mut vars = vec![("OLLAMA_MODEL", model.to_string())];
                if let Some(ref url) = self.base_url {
                    vars.push(("OLLAMA_HOST", url.clone()));
                }
                if let Some(k) = key {
                    vars.push(("OLLAMA_API_KEY", k));
                }
                vars
            }
            ProviderKind::Custom => {
                let mut vars = vec![("CUSTOM_MODEL", model.to_string())];
                if let Some(k) = key {
                    vars.push(("CUSTOM_API_KEY", k));
                }
                if let Some(ref url) = self.base_url {
                    vars.push(("CUSTOM_BASE_URL", url.clone()));
                }
                vars
            }
        }
    }

    /// Seed default: one Anthropic provider.
    pub fn anthropic_seed() -> Self {
        Self {
            id: uuid_v7(),
            label: "Anthropic".into(),
            kind: ProviderKind::Anthropic,
            base_url: None,
            api_key_ref: None,
            default_model: "claude-sonnet-4-5".into(),
            available_models: vec![
                "claude-sonnet-4-5".into(),
                "claude-opus-4-5".into(),
                "claude-haiku-4-5".into(),
            ],
        }
    }

    /// Seed default: one OpenAI provider.
    pub fn openai_seed() -> Self {
        Self {
            id: uuid_v7(),
            label: "OpenAI".into(),
            kind: ProviderKind::OpenAIResponses,
            base_url: None,
            api_key_ref: None,
            default_model: "gpt-4o".into(),
            available_models: vec!["gpt-4o".into(), "gpt-4.1".into(), "gpt-4.1-mini".into()],
        }
    }
}

// ── Agent Instance ─────────────────────────────────────────────────────────────

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct AgentInstance {
    pub id: String,
    pub label: String,
    pub harness: Harness,
    pub provider_id: String,
    pub model: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub instructions: Option<String>,
    pub config_dir: PathBuf,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub icon_override: Option<AgentIcon>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub default_cwd: Option<PathBuf>,
}

impl AgentInstance {
    /// Seed default: one Claude Code instance pointing at the Anthropic seed provider.
    pub fn claude_seed(provider_id: &str) -> Self {
        Self {
            id: uuid_v7(),
            label: "Claude".into(),
            harness: Harness::ClaudeCode,
            provider_id: provider_id.to_string(),
            model: "claude-sonnet-4-5".into(),
            instructions: None,
            config_dir: dirs_next::home_dir()
                .unwrap_or_else(|| PathBuf::from("."))
                .join(".claude"),
            icon_override: None,
            default_cwd: None,
        }
    }
}

// ── UUID v7 (simple) ───────────────────────────────────────────────────────────

pub fn uuid_v7() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let ms = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64;
    let rand: u64 = (ms.wrapping_mul(6364136223846793005))
        .wrapping_add(1442695040888963407)
        .rotate_left(17);
    // time_hi (32 bits from ms), time_mid (16 bits), version+rand, variant+rand, rand
    let time_hi = (ms >> 16) as u32;
    let time_mid = (ms & 0xFFFF) as u16;
    format!(
        "{time_hi:08x}-{time_mid:04x}-{:04x}-{:04x}-{:012x}",
        ((rand >> 16) as u16 & 0x0FFF) | 0x7000,
        ((rand >> 32) as u16 & 0x3FFF) | 0x8000,
        rand >> 48
    )
}

// ── Persistence ────────────────────────────────────────────────────────────────

fn swarmx_dir() -> PathBuf {
    dirs_next::home_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join(".swarmx")
}

pub fn instances_path() -> PathBuf {
    swarmx_dir().join("instances.json")
}

pub fn providers_path() -> PathBuf {
    swarmx_dir().join("providers.json")
}

pub fn load_instances() -> Vec<AgentInstance> {
    let path = instances_path();
    let instances: Vec<AgentInstance> = std::fs::read_to_string(&path)
        .ok()
        .and_then(|s| serde_json::from_str(&s).ok())
        .unwrap_or_default();

    if instances.is_empty() {
        let providers = load_providers();
        let provider_id = providers
            .first()
            .map(|p| p.id.clone())
            .unwrap_or_else(|| ModelProvider::anthropic_seed().id);
        let seed = AgentInstance::claude_seed(&provider_id);
        let _ = save_instances(std::slice::from_ref(&seed));
        return vec![seed];
    }
    instances
}

pub fn save_instances(instances: &[AgentInstance]) -> io::Result<()> {
    let path = instances_path();
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let json = serde_json::to_string_pretty(instances)?;
    std::fs::write(&path, json)
}

pub fn default_models_for_kind(kind: ProviderKind) -> Vec<String> {
    match kind {
        ProviderKind::Anthropic => vec![
            "claude-sonnet-4-5".into(),
            "claude-opus-4-5".into(),
            "claude-haiku-4-5".into(),
        ],
        ProviderKind::OpenAIResponses | ProviderKind::OpenAIChat => {
            vec!["gpt-4o".into(), "gpt-4.1".into(), "gpt-4.1-mini".into()]
        }
        ProviderKind::Ollama => vec!["llama3.2".into()],
        ProviderKind::Custom => vec!["custom".into()],
    }
}

pub fn load_providers() -> Vec<ModelProvider> {
    let path = providers_path();
    let mut providers: Vec<ModelProvider> = std::fs::read_to_string(&path)
        .ok()
        .and_then(|s| serde_json::from_str(&s).ok())
        .unwrap_or_default();

    // Upgrade: populate available_models if empty on existing providers
    for p in &mut providers {
        if p.available_models.is_empty() {
            p.available_models = default_models_for_kind(p.kind);
        }
    }

    if providers.is_empty() {
        let seed = ModelProvider::anthropic_seed();
        let _ = save_providers(std::slice::from_ref(&seed));
        return vec![seed];
    }
    providers
}

pub fn save_providers(providers: &[ModelProvider]) -> io::Result<()> {
    let path = providers_path();
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let json = serde_json::to_string_pretty(providers)?;
    std::fs::write(&path, json)
}

/// Write an API key to the OS keychain for the given provider.
/// Phase 3 only reads; Phase 8 (settings agents tab) does writes.
pub fn set_api_key(provider_id: &str, key: &str) -> Result<(), keyring::Error> {
    keyring::Entry::new(&format!("swarmx.provider.{}", provider_id), "default")?.set_password(key)
}

// ── Tests ──────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests;
