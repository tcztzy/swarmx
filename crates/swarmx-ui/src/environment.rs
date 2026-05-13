//! Runtime environment detection.
//!
//! Checks for required tools (uv, bun, rustup, rg) and offers one-click install.

use std::fmt;

/// Runtime dependency the app may need.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RuntimeDep {
    /// Python package manager — SwarmX agent backend
    Uv,
    /// JavaScript runtime — Claude Code agent backend
    Bun,
    /// Claude Code CLI — default agent, required for claude-agent-acp
    Claude,
    /// Rust toolchain manager — custom agent builds, provides cargo
    Rustup,
    /// Fast code search — used by agents for repository exploration
    Ripgrep,
}

/// Status of a single runtime dependency.
#[derive(Debug, Clone)]
pub struct DepStatus {
    pub dep: RuntimeDep,
    pub installed: bool,
    pub path: Option<String>,
}

impl RuntimeDep {
    /// All dependencies the app cares about.
    pub fn all() -> &'static [RuntimeDep] {
        &[
            RuntimeDep::Uv,
            RuntimeDep::Bun,
            RuntimeDep::Claude,
            RuntimeDep::Rustup,
            RuntimeDep::Ripgrep,
        ]
    }

    /// Required deps — core agent runtime chain.
    pub fn required() -> &'static [RuntimeDep] {
        &[RuntimeDep::Uv, RuntimeDep::Bun, RuntimeDep::Claude]
    }

    /// Optional deps — nice-to-have but not blocking.
    pub fn optional() -> &'static [RuntimeDep] {
        &[RuntimeDep::Rustup, RuntimeDep::Ripgrep]
    }

    /// Binary name used for `which` lookup.
    pub fn binary(self) -> &'static str {
        match self {
            RuntimeDep::Uv => "uv",
            RuntimeDep::Bun => "bun",
            RuntimeDep::Claude => "claude",
            RuntimeDep::Rustup => "rustup",
            RuntimeDep::Ripgrep => "rg",
        }
    }

    /// Human-readable name.
    pub fn label(self) -> &'static str {
        match self {
            RuntimeDep::Uv => "uv",
            RuntimeDep::Bun => "Bun",
            RuntimeDep::Claude => "Claude Code",
            RuntimeDep::Rustup => "rustup",
            RuntimeDep::Ripgrep => "ripgrep",
        }
    }

    /// Why this dep is needed.
    pub fn why(self) -> &'static str {
        match self {
            RuntimeDep::Uv => "Python agent backend (SwarmX)",
            RuntimeDep::Bun => "Claude Code agent backend",
            RuntimeDep::Claude => "Default AI agent — drives claude-agent-acp",
            RuntimeDep::Rustup => "Custom Rust agent builds",
            RuntimeDep::Ripgrep => "Fast code search for agents",
        }
    }

    /// Shell one-liner to install.
    pub fn install_cmd(self) -> &'static str {
        match self {
            RuntimeDep::Uv => "curl -LsSf https://astral.sh/uv/install.sh | sh",
            RuntimeDep::Bun => "curl -fsSL https://bun.sh/install | bash",
            RuntimeDep::Claude => "bun install -g @anthropic-ai/claude-code",
            RuntimeDep::Rustup => {
                "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y"
            }
            RuntimeDep::Ripgrep => "cargo install ripgrep",
        }
    }
}

impl fmt::Display for RuntimeDep {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.label())
    }
}

/// Check if a single binary exists in PATH (synchronous, for startup).
pub fn check_one_sync(dep: RuntimeDep) -> DepStatus {
    let bin = dep.binary();
    match std::process::Command::new("which")
        .arg(bin)
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
    {
        Ok(status) if status.success() => DepStatus {
            dep,
            installed: true,
            path: Some(format!("/usr/bin/{bin}")), // approximate
        },
        _ => DepStatus {
            dep,
            installed: false,
            path: None,
        },
    }
}

/// Check if a single binary exists in PATH.
pub async fn check_one(dep: RuntimeDep) -> DepStatus {
    let bin = dep.binary();
    match tokio::process::Command::new("which")
        .arg(bin)
        .output()
        .await
    {
        Ok(out) if out.status.success() => {
            let path = String::from_utf8_lossy(&out.stdout).trim().to_string();
            DepStatus {
                dep,
                installed: true,
                path: Some(path),
            }
        }
        _ => DepStatus {
            dep,
            installed: false,
            path: None,
        },
    }
}

/// Check all runtime dependencies.
pub async fn check_all() -> Vec<DepStatus> {
    let mut results = Vec::new();
    for dep in RuntimeDep::all() {
        results.push(check_one(*dep).await);
    }
    results
}

/// Run an install command for a dependency.
pub async fn install(dep: RuntimeDep) -> Result<(), String> {
    let cmd = dep.install_cmd();
    let status = tokio::process::Command::new("sh")
        .arg("-c")
        .arg(cmd)
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .status()
        .await
        .map_err(|e| e.to_string())?;

    if !status.success() {
        return Err(format!("install script exited with {status}"));
    }
    Ok(())
}

// ── ACP Agent Runtime Detection ──────────────────────────────────────────────

/// Third-party ACP agents that SwarmX can drive.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AgentRuntime {
    ClaudeAgentAcp,
    CodexAcp,
    OpenCode,
    Hermes,
    OpenClaw,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum RemoteAgentRef {
    Instance(String),
    Runtime(AgentRuntime),
}

/// Detection result for a single agent runtime.
#[derive(Debug, Clone)]
pub struct AgentRuntimeStatus {
    pub agent: AgentRuntime,
    pub available: bool,
    pub version: Option<String>,
}

impl AgentRuntime {
    /// All known third-party ACP agents.
    pub fn all() -> &'static [AgentRuntime] {
        &[
            AgentRuntime::ClaudeAgentAcp,
            AgentRuntime::CodexAcp,
            AgentRuntime::OpenCode,
            AgentRuntime::Hermes,
            AgentRuntime::OpenClaw,
        ]
    }

    /// Human-readable label.
    pub fn label(&self) -> &'static str {
        match self {
            AgentRuntime::ClaudeAgentAcp => "Claude Code (ACP)",
            AgentRuntime::CodexAcp => "Codex (ACP)",
            AgentRuntime::OpenCode => "OpenCode",
            AgentRuntime::Hermes => "Hermes",
            AgentRuntime::OpenClaw => "OpenClaw",
        }
    }

    /// Stable identifier used for runtime-backed sessions.
    pub fn id(&self) -> &'static str {
        match self {
            AgentRuntime::ClaudeAgentAcp => "claude-agent-acp",
            AgentRuntime::CodexAcp => "codex-acp",
            AgentRuntime::OpenCode => "opencode",
            AgentRuntime::Hermes => "hermes",
            AgentRuntime::OpenClaw => "openclaw",
        }
    }

    /// Label to show in session groupings.
    pub fn session_label(&self) -> &'static str {
        match self {
            AgentRuntime::ClaudeAgentAcp => "Claude Code",
            AgentRuntime::CodexAcp => "Codex",
            AgentRuntime::OpenCode => "OpenCode",
            AgentRuntime::Hermes => "Hermes",
            AgentRuntime::OpenClaw => "OpenClaw",
        }
    }

    /// Binary name for `which` lookup. Bun-based agents check bun instead.
    pub fn binary(&self) -> &'static str {
        match self {
            AgentRuntime::ClaudeAgentAcp | AgentRuntime::CodexAcp => "bun",
            AgentRuntime::OpenCode => "opencode",
            AgentRuntime::Hermes => "hermes",
            AgentRuntime::OpenClaw => "openclaw",
        }
    }

    /// Minimum CLI version required (only for non-bun agents).
    pub fn min_version(&self) -> Option<&'static str> {
        match self {
            AgentRuntime::OpenCode => Some("1.1.30"),
            _ => None,
        }
    }

    /// Whether this agent requires bun to run (uses `bun x`).
    pub fn needs_bun(&self) -> bool {
        matches!(self, AgentRuntime::ClaudeAgentAcp | AgentRuntime::CodexAcp)
    }

    /// Underlying CLI binary this agent wraps (for bun-based agents).
    /// e.g. ClaudeAgentAcp wraps `claude`, CodexAcp wraps `codex`.
    pub fn underlying_cli(&self) -> Option<&'static str> {
        match self {
            AgentRuntime::ClaudeAgentAcp => Some("claude"),
            AgentRuntime::CodexAcp => Some("codex"),
            _ => None,
        }
    }

    /// Whether this is a default agent — always shown, always checked.
    pub fn is_default(&self) -> bool {
        matches!(self, AgentRuntime::ClaudeAgentAcp)
    }

    /// Default agents that SwarmX considers required.
    pub fn defaults() -> &'static [AgentRuntime] {
        &[AgentRuntime::ClaudeAgentAcp]
    }

    /// Optional agents — user opts in.
    pub fn optionals() -> &'static [AgentRuntime] {
        &[
            AgentRuntime::CodexAcp,
            AgentRuntime::OpenCode,
            AgentRuntime::Hermes,
            AgentRuntime::OpenClaw,
        ]
    }

    /// The `(program, args)` tuple for `AgentBackend::Custom`.
    pub fn command(&self) -> (&'static str, Vec<&'static str>) {
        match self {
            AgentRuntime::ClaudeAgentAcp => {
                ("bun", vec!["x", "@agentclientprotocol/claude-agent-acp"])
            }
            AgentRuntime::CodexAcp => ("bun", vec!["x", "@agentclientprotocol/codex-acp"]),
            AgentRuntime::OpenCode => ("opencode", vec!["acp"]),
            AgentRuntime::Hermes => ("hermes", vec!["acp"]),
            AgentRuntime::OpenClaw => ("openclaw", vec!["acp"]),
        }
    }

    /// One-click install command. Bun-based agents just need bun.
    pub fn install_label(&self) -> Option<&'static str> {
        if self.needs_bun() {
            Some("Install Bun")
        } else {
            None
        }
    }
}

/// Synchronous check for an ACP agent runtime.
pub fn check_agent_sync(agent: AgentRuntime) -> AgentRuntimeStatus {
    let bin = agent.binary();
    let found = std::process::Command::new("which")
        .arg(bin)
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .map(|s| s.success())
        .unwrap_or(false);

    if !found {
        return AgentRuntimeStatus {
            agent,
            available: false,
            version: None,
        };
    }

    // For bun-based agents, need both bun AND the underlying CLI.
    if agent.needs_bun() {
        let cli_ok = agent.underlying_cli().is_none_or(|cli| {
            std::process::Command::new("which")
                .arg(cli)
                .stdout(std::process::Stdio::null())
                .stderr(std::process::Stdio::null())
                .status()
                .map(|s| s.success())
                .unwrap_or(false)
        });
        return AgentRuntimeStatus {
            agent,
            available: cli_ok,
            version: None,
        };
    }

    // For opencode, check version.
    let version = std::process::Command::new(bin)
        .arg("--version")
        .output()
        .ok()
        .and_then(|out| {
            let s = String::from_utf8_lossy(&out.stdout).trim().to_string();
            // Strip leading 'v' if present
            s.strip_prefix('v').map(|v| v.to_string()).or(Some(s))
        });

    let meets_min = version
        .as_ref()
        .is_some_and(|v| agent.min_version().is_none_or(|min| version_ge(v, min)));

    AgentRuntimeStatus {
        agent,
        available: meets_min,
        version,
    }
}

/// Simple semver-like comparison: "1.1.30" >= "1.1.30" → true, "1.0.0" >= "1.1.30" → false.
fn version_ge(actual: &str, required: &str) -> bool {
    let parse =
        |s: &str| -> Vec<u32> { s.split('.').filter_map(|p| p.parse::<u32>().ok()).collect() };
    let a = parse(actual);
    let r = parse(required);
    let n = a.len().max(r.len());
    for i in 0..n {
        let av = a.get(i).copied().unwrap_or(0);
        let rv = r.get(i).copied().unwrap_or(0);
        match av.cmp(&rv) {
            std::cmp::Ordering::Greater => return true,
            std::cmp::Ordering::Less => return false,
            std::cmp::Ordering::Equal => continue,
        }
    }
    true // equal
}
