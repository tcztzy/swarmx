//! Agent harness catalog (spec §4.2).
//!
//! Each `Harness` variant knows how to spawn its agent binary with the correct
//! environment (config dir, passthrough vars, provider credentials).

use std::ffi::OsString;
use std::path::Path;
use std::process::Command;

use crate::data::AgentIcon;
use crate::instance::{AgentInstance, ModelProvider, ProviderKind};

// ── Harness ────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum Harness {
    SwarmX,
    ClaudeCode,
    Codex,
    OpenCode,
    Hermes,
    OpenClaw,
}

impl Harness {
    pub fn all() -> [Harness; 6] {
        [
            Self::SwarmX,
            Self::ClaudeCode,
            Self::Codex,
            Self::OpenCode,
            Self::Hermes,
            Self::OpenClaw,
        ]
    }

    pub fn label(&self) -> &'static str {
        match self {
            Harness::SwarmX => "SwarmX",
            Harness::ClaudeCode => "Claude Code",
            Harness::Codex => "Codex",
            Harness::OpenCode => "OpenCode",
            Harness::Hermes => "Hermes",
            Harness::OpenClaw => "OpenClaw",
        }
    }

    pub fn icon(&self) -> AgentIcon {
        match self {
            Harness::SwarmX => AgentIcon::SwarmX,
            Harness::ClaudeCode => AgentIcon::ClaudeCode,
            Harness::Codex => AgentIcon::Codex,
            Harness::OpenCode => AgentIcon::OpenCode,
            Harness::Hermes => AgentIcon::Hermes,
            Harness::OpenClaw => AgentIcon::OpenClaw,
        }
    }

    /// Which provider kinds this harness can use (spec §4.2).
    pub fn compatible(&self) -> &'static [ProviderKind] {
        match self {
            Harness::ClaudeCode => &[ProviderKind::Anthropic],
            Harness::Codex => &[ProviderKind::OpenAIResponses, ProviderKind::OpenAIChat],
            Harness::OpenCode => &[
                ProviderKind::Anthropic,
                ProviderKind::OpenAIChat,
                ProviderKind::Ollama,
            ],
            Harness::Hermes => &[ProviderKind::OpenAIChat, ProviderKind::Ollama],
            Harness::OpenClaw => &[ProviderKind::Anthropic],
            Harness::SwarmX => &[
                ProviderKind::Anthropic,
                ProviderKind::OpenAIChat,
                ProviderKind::Ollama,
            ],
        }
    }

    /// Standard Unix environment variables passed through to the child process.
    pub fn passthrough_env(&self) -> &'static [&'static str] {
        &["PATH", "HOME", "LANG", "USER", "SHELL", "TERM"]
    }

    /// Harness-specific config-dir environment variable (spec §4.4).
    pub fn config_env(&self, config_dir: &Path) -> Vec<(&'static str, OsString)> {
        let key = match self {
            Harness::SwarmX => "SWARMX_CONFIG_DIR",
            Harness::ClaudeCode => "CLAUDE_CONFIG_DIR",
            Harness::Codex => "CODEX_CONFIG_DIR",
            Harness::OpenCode => "OPENCODE_CONFIG_DIR",
            Harness::Hermes => "HERMES_CONFIG_DIR",
            Harness::OpenClaw => "OPENCLAW_CONFIG_DIR",
        };
        vec![(key, OsString::from(config_dir))]
    }

    /// Base binary + args for this harness.
    pub fn base_command(&self) -> Command {
        let mut cmd = Command::new(match self {
            Harness::SwarmX => "python3",
            Harness::ClaudeCode | Harness::Codex => "bun",
            Harness::OpenCode => "opencode",
            Harness::Hermes => "hermes",
            Harness::OpenClaw => "openclaw",
        });
        match self {
            Harness::SwarmX => {
                cmd.args(["-m", "swarmx.acp_server"]);
            }
            Harness::ClaudeCode => {
                cmd.args(["x", "@agentclientprotocol/claude-agent-acp"]);
            }
            Harness::Codex => {
                cmd.args(["x", "@agentclientprotocol/codex-acp"]);
            }
            Harness::OpenCode => {
                cmd.arg("acp");
            }
            Harness::Hermes => {
                cmd.arg("acp");
            }
            Harness::OpenClaw => {
                cmd.arg("acp");
            }
        }
        cmd
    }

    /// Full spawn command: base binary + cwd + env_clear + explicit envs.
    pub fn spawn_command(
        &self,
        inst: &AgentInstance,
        provider: &ModelProvider,
        cwd: &Path,
    ) -> Command {
        let mut cmd = self.base_command();
        cmd.current_dir(cwd);
        cmd.env_clear();

        // Passthrough standard env vars.
        for key in self.passthrough_env() {
            if let Ok(val) = std::env::var(key) {
                cmd.env(key, val);
            }
        }

        // Config dir isolation.
        for (k, v) in self.config_env(&inst.config_dir) {
            cmd.env(k, v);
        }

        // Provider credentials + model.
        for (k, v) in provider.env_vars(&inst.model) {
            cmd.env(k, v);
        }

        cmd
    }
}

// ── Tests ──────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests;
